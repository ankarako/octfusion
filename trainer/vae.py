from typing import Dict, Any, List
from trainer.registry import register_trainer

from datasets.registry import get_dataset
from datasets.sampler import InfSampler

import torch
from torch.utils.data import DataLoader

from models.networks.dualoctree_networks.graph_vae import GraphVAE
from models.networks.dualoctree_networks.loss import geometry_loss
import ocnn
from ocnn.dataset import CollateBatch

import utils.state
import utils.log as log
from utils.util_dualoctree import calc_sdf
from utils.mesh import mcubes
from tqdm import tqdm
import copy
import os
from datetime import datetime
import trimesh        

@register_trainer
class VAETrainer:
    def __init__(
        self,
        exp_name: str,
        output_dir: str,
        seed: int,
        device: List[int],
        nepochs: int,
        chkp_iters: int,
        chkp_filepath: str,
        log_iters: int,
        trainset_conf: Dict[str, Any],
        trainloader_kwargs: Dict[str, Any],
        model_kwargs: Dict[str, Any],
        optim_kwargs: Dict[str, Any],
        w_kl: float=0.1,
        sdf_res: int=256
    ):
        """
        Instantiate a VAETrainer object.

        :param exp_name The experiment's name
        :param output_dir The directory to save checkpoints, and intermediate meshes
            during training.
        :param seed A global seed for reproducibility.
        :param device A list of integers indicating the GPU ids to train on (only single GPU supported :P).
        :param nepochs The number of training epochs.
        :param chkp_iters Checkpoint saving interval in iterations.
        :param chkp_filepath A path to a checkpoint file for resuming training.
        :param log_iters Logging interval in iterations (basically intermediate output saving).
        :param trainset_conf A dictionary holding the configuration for the dataset to load.
        :param trainloader_kwargs A dictionary holding the training DataLoader's keyword arguments.
        :param model_kwargs A dictionary holding the GraphVAE's keyword arguments.
        :param optim_kwargs A dictionary holding the AdamW optimizer's keyword arguments.
        :param w_kl KL divergence weight for loss calculation.
        :param sdf_res The sdf grid resolution (used only for normalizing the output meshes.)
        """
        # training state
        self.exp_name = exp_name
        self.nepochs = nepochs
        self.sdf_res = sdf_res

        # create output directory
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        
        now = datetime.now().strftime("%d-%m-%y-%H-%M-%S")
        experiment_folder = exp_name + f"-{now}"
        self.experiment_dir = os.path.join(output_dir, experiment_folder)
        if not os.path.exists(self.experiment_dir):
            os.mkdir(self.experiment_dir)

        # more training state
        utils.state.seed(seed)
        self.device = utils.state.get_device(device)
        
        # training dataset
        self.trainset_conf = trainset_conf
        self.trainloader_kwargs = trainloader_kwargs
        self.trainset, collate_fn = get_dataset(trainset_conf.key, **trainset_conf.kwargs, device=self.device)
        sampler = InfSampler(self.trainset, shuffle=True)
        self.trainloader = DataLoader(
            self.trainset, sampler=sampler, **trainloader_kwargs, collate_fn=collate_fn
        )
        self.train_iter = iter(self.trainloader)

        # instantiate vae
        self.model_kwargs = model_kwargs
        self.autoencoder = GraphVAE(**model_kwargs).to(self.device)
        
        # instatiate optimizer
        self.optim = torch.optim.AdamW([p for p in self.autoencoder.parameters() if p.requires_grad == True], **optim_kwargs)

        # I really don't like doing this stuff, but it will do for now
        def poly(epoch: int, lr_power: float=0.9):
            return (1 - epoch / self.nepochs) ** lr_power
        self.sched = torch.optim.lr_scheduler.LambdaLR(self.optim, poly)

        # more training state
        self.epoch_len = len(self.trainloader)
        self.total_iters = self.nepochs * self.epoch_len
        self.iter_range = range(self.total_iters)
        self.curr_iter = -1
        self.curr_epoch = -1

        # loss weights
        self.w_kl = w_kl

        # checkpoint state
        self.chkp_dir = os.path.join(self.experiment_dir, 'chkp')
        self.chkp_iters = chkp_iters
        if not os.path.exists(self.chkp_dir):
            os.mkdir(self.chkp_dir)
        
        # load from checkpoint if specified
        if chkp_filepath is not None and os.path.exists(chkp_filepath):
            log.INFO(f"Loading checkpoint from: {chkp_filepath}")
            self.load_chkp(chkp_filepath)
        
        # logging state
        self.log_iters = log_iters
    
    def infer(self, batch: Dict[str, Any]) -> None:
        """
        Perform one inference step and save the output
        """
        autoencoder_out = self.autoencoder(
            batch['octree'], batch['octree_gt'], batch['pos']
        )
        bbmin, bbmax = -0.9, 0.9
        sdf_batched = calc_sdf(
            autoencoder_out['neural_mpu'], self.trainloader.batch_size, bbmin=bbmin, bbmax=bbmax
        )

        bsize = sdf_batched.shape[0]
        for i in range(bsize):
            sdf = sdf_batched[i].cpu().numpy()
            v_pos, t_pos_idx = mcubes(sdf, 0.0)
            if v_pos.size == 0 or t_pos_idx.size == 0:
                log.WARN(f"Marching cubes returned empty mesh (iter: {self.curr_iter}, batch: {i})")
                continue
            v_pos = v_pos * ((bbmax - bbmin) / self.sdf_res) + bbmin
            mesh = trimesh.Trimesh(v_pos, t_pos_idx)
            filename = f"infer-{self.exp_name}-iter{self.curr_iter}-b{i}.obj"
            filepath = os.path.join(self.experiment_dir, filename)
            mesh.export(filepath)
    
    def get_lr(self):
        """
        Get the current learning rate from the optimizer.
        This will work for our case as all the optimized 
        parameters are trained under the same learning rate.

        :return The optimizer's current learning rate.
        """
        lr = 0.0
        for param_group in self.optim.param_groups:
            lr = param_group['lr']
            break
        return lr

    def train(self):
        """
        Run the trainer
        """
        log.INFO(f"Running: {repr(self)}")
        loop = tqdm(self.iter_range, total=len(self.iter_range), ncols=100)
        self.autoencoder.train()
        for iter in loop:
            # handle training state
            self.curr_epoch = iter // self.epoch_len
            self.curr_iter = iter
            loop.set_description_str(f"e: {self.curr_epoch} | training")

            # load next batch
            data = next(self.train_iter)
            # process input
            # create octrees of each set of points
            octrees = []
            for pts in data['points']:
                oc = ocnn.octree.Octree(
                    depth=self.autoencoder.depth_out, 
                    full_depth=self.autoencoder.full_depth
                )
                oc.build_octree(pts)
                octrees += [oc]
            octree = ocnn.octree.merge_octrees(octrees)
            octree.construct_all_neigh()
            octree_gt = copy.deepcopy(octree)
            data['octree'] = octree
            data['octree_gt'] = octree_gt
            data['pos'].requires_grad = True
            data['pos'] = data['pos']

            # perform a forward step
            autoencoder_out = self.autoencoder(data['octree'], data['octree_gt'], data['pos'])

            # calculate losses
            output = geometry_loss(data, autoencoder_out, 'sdf_reg_loss', kl_weight=self.w_kl)
            losses = [val for key, val in output.items() if 'loss' in key]
            output['loss'] = torch.sum(torch.stack(losses))
            output['code_max'] = autoencoder_out['code_max']
            output['code_min'] = autoencoder_out['code_min']

            # optimize params
            self.optim.zero_grad()
            output['loss'].backward()
            self.optim.step()
            if self.curr_iter % self.epoch_len == 0 and self.curr_iter > 0:
                self.sched.step()

            loop.set_postfix({
                'loss': output['loss'].detach().item(),
                'lr': self.get_lr()
            })
            
            # log
            if (self.curr_iter % self.log_iters == 0):
                # want to save iteration 0 and use it as
                # a sanity check for exporting
                loop.set_description_str(f"e: {self.curr_epoch} | saving output")
                self.autoencoder.eval()
                self.infer(data)
                self.autoencoder.train()
            

            # save checkpoint
            if (self.curr_iter % self.chkp_iters == 0):
                # want to save iteration 0 and use it as 
                # sanity check for saving checkpoints
                loop.set_description_str(f"e: {self.curr_epoch} | saving checkpoint")
                self.save_chkp()
        log.INFO("Training terminated.")
        log.INFO("Saving checkpoint...")
        chkp_fpath = self.save_chkp()
        log.INFO(f"Checkpoint saved at: {chkp_fpath}")
    
    def __repr__(self) -> str:
        """
        Get a string representation of the trainer's state

        :return A string containing the trainer's state.
        """
        _repr = f"{VAETrainer.__name__}(\n"
        _repr += f"\texp_name={self.exp_name},\n"
        _repr += f"\tdevice={self.device},\n"
        _repr += f"\tnepochs={self.nepochs}\n"
        _repr += f"\ttrainset={self.trainset_conf},\n"
        _repr += f"\ttrainloader={self.trainloader_kwargs},\n"
        _repr += f"\tmodel=GraphVAE,\n"
        _repr += f"\tmodel_kwargs={self.model_kwargs},\n"
        _repr += f")"
        return _repr

    def save_chkp(self) -> None:
        """
        Save a checkpoint of the current training state.

        :return The filepath of the saved checkpoint
        """
        filename = f"{self.exp_name}-iter{self.curr_iter}.ckpt"
        filepath = os.path.join(self.chkp_dir, filename)
        state_dict = {
            'model': self.autoencoder.state_dict(),
            'optim': self.optim.state_dict(),
            'epoch': self.curr_epoch,
            'iter': self.curr_iter
        }
        torch.save(state_dict, filepath)
        return filepath
    
    def load_chkp(self, filepath) -> None:
        """
        Load a checkpoint from the specified filepath.

        :param filepath The path to the checkpoint file to load.
        """
        state_dict = torch.load(filepath)
        self.autoencoder.load_state_dict(state_dict['model'])
        self.optim.load_state_dict(state_dict['optim'])
        self.curr_epoch = state_dict['epoch']
        self.curr_iter = state_dict['iter']
        self.total_iters = self.total_iters - self.curr_iter
        self.iter_range = range(self.curr_iter, self.total_iters)
        