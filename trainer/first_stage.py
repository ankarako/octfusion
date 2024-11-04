from typing import Dict, Any, List
from trainer.registry import register_trainer
from datasets.registry import get_dataset
from datasets.sampler import InfSampler

import torch
from torch.utils.data import DataLoader

from models.networks.dualoctree_networks.graph_vae import GraphVAE
from models.networks.diffusion_networks.graph_unet_lr import UNet3DModel
from models.networks.diffusion_networks.ldm_diffusion_util import (
    beta_linear_log_snr, log_snr_to_alpha_sigma, right_pad_dims_to
)

import ocnn
from ocnn.dataset import CollateBatch

import utils
import utils.log as log
from utils.util_dualoctree import octree2split_small


from tqdm import tqdm
import copy
import os
import trimesh


@register_trainer
class FirstStageTrainer:
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
        vae_kwargs: Dict[str, Any],
        vae_chkp_filepath: str,
        unet_kwargs: Dict[str, Any],
        optim_kwargs: Dict[str, Any]
    ):
        """
        Instantiate a FirstStageTrainer object
        """
        # training state
        self.exp_name = exp_name
        self.nepochs = nepochs

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
        self.unet_kwargs = unet_kwargs
        self.unet = UNet3DModel(**unet_kwargs).to(self.device)

        # instantiate optim
        self.optim = torch.optim.AdamW([p for p in self.unet.parameters() if p.requires_grad == True], **optim_kwargs)

        # more training state
        self.epoch_len = len(self.trainloader)
        self.total_iters = self.nepochs * self.epoch_len
        self.iter_range = range(self.total_iters)
        self.curr_iter = -1
        self.curr_iter = -1

        # chechpoint state
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

    
    def train(self):
        """
        Start training
        """
        log.INFO(f"Running: {repr(self)}")
        loop = tqdm(self.iter_range, total=len(self.iter_range), ncols=100)
        self.autoencoder.eval()
        self.unet.train()
        for iter in loop:
            # handle training state
            self.curr_epoch = iter // self.epoch_len
            self.curr_iter = iter
            loop.set_description_str(f"e: {self.curr_epoch} | training")

            # load next batch
            data = next(self.train_iter)

            # process input
            # create octrees for each set of points
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
            data['octree_in'] = octree
            data['split_small'] = octree2split_small(data['octree_in'], self.autoencoder.full_depth)

            # perform a forward step
            df_lr_loss = torch.tensor(0.0, device=self.device)
            batch_id = torch.arange(0, self.trainloader.batch_size, device=self.device, dtype=torch.long)

            times = torch.zeros([self.trainloader.batch_size,], device=self.device).float().uniform_(0, 1)
            noise = torch.randn_like(data['split_small'])
            noise_level = beta_linear_log_snr(times)
            alpha, sigma = log_snr_to_alpha_sigma(noise_level)
            batch_alpha = right_pad_dims_to(data['split_small'], alpha[batch_id])
            batch_sigma = right_pad_dims_to(data['split_small'], sigma[batch_id])
            noised_data = batch_alpha * data['split_small'] + batch_sigma * noise

            output = self.unet(
                unet_type='lr', x=noised_data, doctree=None, lr=None, timesteps=noise_level, label=None
            )

            # calc loss
            df_lr_loss = torch.nn.functional.mse_loss(output, data['split_small'])
            
            # optimize params
            self.optim.zero_grad()
            df_lr_loss.backward()
            self.optim.step()
            # TODO: UPDATE EMA

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
        _repr += f"\tunet_kwargs={self.unet_kwargs},\n"
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
        self.unet.load_state_dict(state_dict['df'])
        self.optim.load_state_dict(state_dict['optim'])
        self.curr_epoch = state_dict['epoch']
        self.curr_iter = state_dict['iter']
        self.total_iters = self.total_iters - self.curr_iter
        self.iter_range = range(self.curr_iter, self.total_iters)