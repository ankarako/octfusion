from typing import Dict, Any, List
from trainer.registry import register_trainer
from datasets.registry import get_dataset
from datasets.sampler import InfSampler

import torch
from torch.utils.data import DataLoader

from models.networks.dualoctree_networks.graph_vae import GraphVAE
from models.networks.diffusion_networks.graph_unet_lr import UNet3DModel
from models.networks.diffusion_networks.ldm_diffusion_util import (
    beta_linear_log_snr, 
    log_snr_to_alpha_sigma, 
    right_pad_dims_to, 
    EMA, 
    set_requires_grad,
    update_moving_average
)

import ocnn
import utils
import utils.log as log
from utils.util_dualoctree import octree2split_small


from tqdm import tqdm
import copy
import os
from datetime import datetime
import trimesh
from enum import Enum


class Stage(Enum):
    first = 0
    second = 1


@register_trainer
class FirstStageTrainer:
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
        ema_rate: float,
        stage: str,
        trainset_conf: Dict[str, Any],
        trainloader_kwargs: Dict[str, Any],
        vae_kwargs: Dict[str, Any],
        vae_chkp_filepath: str,
        df_kwargs: Dict[str, Any],
        optim_kwargs: Dict[str, Any]
    ):
        """
        Instantiate a FirstStageTrainer object
        """
        # training state
        self.exp_name = exp_name
        self.nepochs = nepochs
        assert stage in ['first', 'second'], f"The specified stage is invalid. Expected one of [first, second], got {stage}"
        self.stage = Stage[stage.lower()]

        # create output directory
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        
        now = datetime.now().strftime("%d-%m-%y-%H-%M-%S")
        experiment_folder = exp_name + f"-{now}"
        self.experiment_dir = os.path.join(output_dir, experiment_folder)
        if not os.path.exists(self.experiment_dir):
            os.mkdir(self.experiment_dir)

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
        if not os.path.exists(vae_chkp_filepath):
            log.ERROR(f"You need to specified GraphVAE's checkpoint")
        self.vae_kwargs = vae_kwargs
        self.autoencoder = GraphVAE(**vae_kwargs).to(self.device)
        vae_dict = torch.load(vae_chkp_filepath)
        self.autoencoder.load_state_dict(vae_dict['model'])

        # instantiate diffusion model
        self.unet_type = "lr" if self.stage == Stage.first else "hr"
        self.df_kwargs = df_kwargs
        self.df = UNet3DModel(**df_kwargs).to(self.device)
        self.split_channel = 8
        self.code_channel = self.autoencoder.embed_dim
        z_sp_dim = 2 ** self.autoencoder.full_depth
        self.z_shape = (self.split_channel, z_sp_dim, z_sp_dim, z_sp_dim)

        self.ema_rate = ema_rate
        self.df_ema = copy.deepcopy(self.df)
        self.df_ema.to(self.device)
        self.ema_updater = EMA(self.ema_rate)
        
        # disable parameter update in ema copy
        set_requires_grad(self.df_ema, False)
        self.reset_ema_params()
        
        # noise configuration
        self.noise_schedule = "linear"

        # first stage trains only the low-resolution part
        if self.stage == Stage.first:
            set_requires_grad(self.df.unet_hr, False)
        elif self.stage == Stage.second:
            set_requires_grad(self.df.unet_lr, False)

        # instantiate optim
        self.optim = torch.optim.AdamW([p for p in self.df.parameters() if p.requires_grad == True], **optim_kwargs)

        # more training state
        self.epoch_len = len(self.trainloader)
        self.total_iters = self.nepochs * self.epoch_len
        self.iter_range = range(self.total_iters)
        self.curr_iter = -1
        self.curr_iter = -1

        # chechpoint state
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

    def reset_ema_params(self):
        """
        Reset EMA unet parameters.
        """
        self.df_ema.load_state_dict(self.df.state_dict())

    def process_input(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process the input batch. Create octree from the 
        specified Points objects, and split

        :param batch The input batch to process.
        :return The processed batch.
        """
        octrees = []
        for pts in batch['points']:
            oc = ocnn.octree.Octree(
                depth=self.autoencoder.depth_out,
                full_depth=self.autoencoder.full_depth
            )
            oc.build_octree(pts)
            octrees += [oc]
        octree = ocnn.octree.merge_octrees(octrees)
        octree.construct_all_neigh()
        batch['octree_in'] = octree
        batch['split_small'] = octree2split_small(batch['octree_in'], self.autoencoder.full_depth)
        return batch
    
    def train(self):
        """
        Start training
        """
        log.INFO(f"Running: {repr(self)}")
        loop = tqdm(self.iter_range, total=len(self.iter_range), ncols=100)
        self.autoencoder.eval()
        self.df.train()
        for iter in loop:
            # handle training state
            self.curr_epoch = iter // self.epoch_len
            self.curr_iter = iter
            loop.set_description_str(f"e: {self.curr_epoch} | training")

            # load next batch
            input_batch = next(self.train_iter)
            input_batch = self.process_input(input_batch)

            # perform a forward step
            df_loss = torch.tensor(0.0, device=self.device)
            batch_id = torch.arange(0, self.trainloader.batch_size, device=self.device, dtype=torch.long)

            times = torch.zeros([self.trainloader.batch_size,], device=self.device).float().uniform_(0, 1)
            noise = torch.randn_like(input_batch['split_small'])
            noise_level = beta_linear_log_snr(times)
            alpha, sigma = log_snr_to_alpha_sigma(noise_level)
            batch_alpha = right_pad_dims_to(input_batch['split_small'], alpha[batch_id])
            batch_sigma = right_pad_dims_to(input_batch['split_small'], sigma[batch_id])
            noised_data = batch_alpha * input_batch['split_small'] + batch_sigma * noise

            output = self.df(
                unet_type=self.unet_type, x=noised_data, doctree=None, lr=None, timesteps=noise_level, label=None
            )

            # calc loss
            df_loss = torch.nn.functional.mse_loss(output, input_batch['split_small'])
            
            # optimize params
            self.optim.zero_grad()
            df_loss.backward()
            self.optim.step()
            update_moving_average(self.df_ema, self.df, self.ema_updater)

            # log
            if (self.curr_iter % self.log_iters == 0) and (self.curr_iter > 0):
                loop.set_description_str(f"e: {self.curr_epoch} | infering")
            
            # save checkpoint
            if (self.curr_iter % self.chkp_iters == 0) and (self.curr_iter > 0):
                loop.set_description_str(f"e: {self.curr_epoch} | saving checkpoint")
                self.save_chkp()
        log.INFO("Training terminated.")
        log.INFO("Saving checkpoint...")
        self.save_chkp()
    
    def __repr__(self):
        _repr = f"{FirstStageTrainer.__name__}(\n"
        _repr += f"\texp_name={self.exp_name}\n"
        _repr += f"\tdevice={self.device}\n"
        _repr += f"\tnepochs={self.nepochs}\n"
        _repr += f"\ttrainset={self.trainset_conf},\n"
        _repr += f"\ttrainloader={self.trainloader_kwargs},\n"
        _repr += f"\tvae=GraphVAE,\n"
        _repr += f"\tvae_kwargs={self.vae_kwargs}"
        _repr += f"\tunet=UNet3D,\n"
        _repr += f"\tunet_kwargs={self.df_kwargs},\n"
        _repr = ")"
        return _repr
    
    def save_chkp(self) -> None:
        """
        Save a checkpoint
        """
        filename = f"{self.exp_name}_iter{self.curr_iter}.ckpt"
        filepath = os.path.join(self.chkp_dir, filename)
        state_dict = {
            'df': self,
            'vae': self.autoencoder.state_dict(),
            'optim': self.optim.state_dict(),
            'epoch': self.curr_epoch,
            'iter': self.curr_iter
        }
        torch.save(state_dict, filepath)

    
    def load_chkp(self, filepath: str) -> None:
        """
        Load a checkpoint from the specified filepath.

        :param filepath The path of the checkpoint file to load.
        """
        state_dict = torch.load(filepath)
        self.autoencoder.load_state_dict(state_dict['vae'])
        self.df.load_state_dict(state_dict['df'])
        self.optim.load_state_dict(state_dict['optim'])
        self.curr_epoch = state_dict['epoch']
        self.curr_iter = state_dict['iter']
        self.total_iters = self.total_iters - self.curr_iter
        self.iter_range = range(self.curr_iter, self.total_iters)