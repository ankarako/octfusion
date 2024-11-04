from typing import Dict, Any, List
import functools
from trainer.registry import register_trainer

from datasets.registry import get_dataset
from datasets.sampler import InfSampler

import torch
import numpy as np
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
import trimesh



def collate_fn(data):
    octree_collate_fn = CollateBatch(merge_points=False)
    output = octree_collate_fn(data)
    if 'pos' in output:
        batch_idx = torch.cat([
            torch.ones([pos.size(0), 1], device=pos.device) * i for i, pos in enumerate(output['pos'])
        ], dim=0)
        pos = torch.cat(output['pos'], dim=0)
        output['pos'] = torch.cat([pos, batch_idx], dim=1)

    for key in ['grad', 'sdf']:
        if key in output:
            output[key] = torch.cat(output[key], dim=0)
    return output
        

@register_trainer
class VAETrainer:
    def __init__(
        self,
        exp_name: str,
        seed: int,
        device: List[int],
        nepochs: int,
        chkp_dir: str,
        chkp_iters: int,
        chkp_filepath: str,
        log_dir: str,
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
        """
        # training state
        self.exp_name = exp_name
        self.nepochs = nepochs
        self.sdf_res = sdf_res

        utils.state.seed(seed)
        self.device = utils.state.get_device(device)
        
        # training dataset
        self.trainset_conf = trainset_conf
        self.trainloader_kwargs = trainloader_kwargs
        self.trainset = get_dataset(trainset_conf.key, **trainset_conf.kwargs, device=self.device)
        sampler = InfSampler(self.trainset, shuffle=True)
        self.trainloader = DataLoader(
            self.trainset, sampler=sampler, **trainloader_kwargs, collate_fn=collate_fn
        )
        self.train_iter = iter(self.trainloader)

        # instantiate vae
        self.model_kwargs = model_kwargs
        self.autoencoder = GraphVAE(**model_kwargs).to(self.device)
        
        # instatiate optim
        self.optim = torch.optim.AdamW(self.autoencoder.parameters(), **optim_kwargs)

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
        self.chkp_dir = chkp_dir
        self.chkp_iters = chkp_iters
        if not os.path.exists(self.chkp_dir):
            os.mkdir(chkp_dir)
        
        # load from checkpoint if specified
        if chkp_filepath is not None and os.path.exists(chkp_filepath):
            log.INFO(f"Loading checkpoint from: {chkp_filepath}")
            self.load_chkp(chkp_filepath)
        
        # logging state
        self.log_dir = log_dir
        self.log_iters = log_iters
    
    def infer(self, data: Dict[str, Any]) -> None:
        """
        Perform one inference step and save the output
        """
        autoencoder_out = self.autoencoder(
            data['octree'], data['octree_gt'], data['pos']
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
            filepath = os.path.join(self.log_dir, filename)
            mesh.export(filepath)
    
    def get_lr(self):
        lr = 0.0
        for param_group in self.optim.param_groups:
            lr = param_group['lr']
            break
        return lr

    def run(self):
        """
        Run the trainer
        """
        log.INFO(f"Running: {repr(self)}")
        loop = tqdm(self.iter_range, total=self.total_iters, ncols=100)
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

            # perform a forward step
            autoencoder_out = self.autoencoder(
                data['octree'], data['octree_gt'], data['pos']
            )

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
            if self.curr_iter % self.epoch_len == 0:
                self.sched.step()

            loop.set_postfix({
                'loss': output['loss'].detach().item(),
                'lr': self.get_lr()
            })
            
            # log
            if (self.curr_iter % self.log_iters == 0) and (self.curr_iter > 0):
                loop.set_description_str(f"e: {self.curr_epoch} | infering")
                self.autoencoder.eval()
                self.infer(data)
                self.autoencoder.train()
                loop.set_description_str(f"e: {self.curr_epoch} | training")
            

            # save checkpoint
            if (self.curr_iter % self.chkp_iters == 0) and (self.curr_iter > 0):
                loop.set_description_str(f"e: {self.curr_epoch} | saving checkpoint")
                self.save_chkp()
                loop.set_description_str(f"e: {self.curr_epoch} | training")
    
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
        Save a checkpoint
        """
        filename = f"{self.exp_name}_iter{self.curr_iter}.ckpt"
        filepath = os.path.join(self.chkp_dir, filename)
        state_dict = {
            'model': self.autoencoder.state_dict(),
            'optim': self.optim.state_dict(),
            'epoch': self.curr_epoch,
            'iter': self.curr_iter
        }
        torch.save(state_dict, filepath)
    
    def load_chkp(self, filepath) -> None:
        """
        Load a checkpoint from the specified filepath
        """
        state_dict = torch.load(filepath)
        self.autoencoder.load_state_dict(state_dict['model'])
        self.optim.load_state_dict(state_dict['optim'])
        self.curr_epoch = state_dict['epoch']
        self.curr_iter = state_dict['iter']
        self.total_iters = self.total_iters - self.curr_iter
        self.iter_range = range(self.curr_iter, self.total_iters)
        