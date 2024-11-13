from typing import Dict, Any, List
from functools import partial
from trainer.registry import register_trainer
from datasets.registry import get_dataset
from datasets.sampler import InfSampler

import torch
from torch.utils.data import DataLoader
from torch.special import expm1

from models.networks.dualoctree_networks.graph_vae import GraphVAE, dual_octree
from models.networks.diffusion_networks.graph_unet_union import UNet3DModel
from models.networks.diffusion_networks.ldm_diffusion_util import (
    beta_linear_log_snr, 
    log_snr_to_alpha_sigma, 
    right_pad_dims_to, 
    EMA, 
    set_requires_grad,
    update_moving_average,
    voxel2mesh
)

from einops import repeat

import ocnn

import utils
from utils.mesh import mcubes
import utils.log as log
from utils.util_dualoctree import (
    octree2split_small, split2octree_small, octree2voxel, calc_sdf
)

from sampler import sample

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
class StageTrainer:
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
        sample_iters: int,
        ema_rate: float,
        stage: str,
        sdf_res: int,
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
        self.sdf_res = sdf_res
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

        self.seed = seed
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
            log.ERROR(f"You need to specify GraphVAE's checkpoint")
        self.vae_kwargs = vae_kwargs
        self.autoencoder = GraphVAE(**vae_kwargs).to(self.device)
        vae_dict = torch.load(vae_chkp_filepath)
        self.autoencoder.load_state_dict(vae_dict['model'])

        # instantiate diffusion model
        self.unet_type = "lr" if self.stage == Stage.first else "hr"
        self.df_kwargs = df_kwargs
        self.df = UNet3DModel(**df_kwargs, stage_flag=self.unet_type, use_checkpoint=False).to(self.device)
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

        # chechpoint state
        self.chkp_dir = os.path.join(self.experiment_dir, 'chkp')
        self.chkp_iters = chkp_iters
        if not os.path.exists(self.chkp_dir):
            os.mkdir(self.chkp_dir)

        # first stage trains only the low-resolution part
        if self.stage == Stage.second:
            set_requires_grad(self.df.unet_lr, False)

        # instantiate optim
        self.optim = torch.optim.AdamW([p for p in self.df.parameters() if p.requires_grad == True], **optim_kwargs)

        # more training state
        self.epoch_len = len(self.trainloader)
        self.total_iters = self.nepochs * self.epoch_len
        self.iter_range = range(self.total_iters)
        self.curr_iter = -1
        self.curr_iter = -1


        # logging state
        self.log_iters = log_iters
        self.sample_iters = sample_iters
        # for exporting eval data
        self.small_depth = 6
        self.large_depth = 8

        # load from checkpoint if specified
        if chkp_filepath is not None and os.path.exists(chkp_filepath):
            log.INFO(f"Loading checkpoint from: {chkp_filepath}")
            self.load_chkp(chkp_filepath)

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
            
            if self.stage == Stage.first:
                batch_id = torch.arange(0, self.trainloader.batch_size, device=self.device, dtype=torch.long)
                input_data = input_batch['split_small']
            elif self.stage == Stage.second:
                with torch.no_grad():
                    input_data, doctree_in = self.autoencoder.extract_code(input_batch['octree_in'])
                    batch_id = doctree_in.batch_id(self.small_depth)

            times = torch.zeros([self.trainloader.batch_size,], device=self.device).float().uniform_(0, 1)
            noise = torch.randn_like(input_data)
            noise_level = beta_linear_log_snr(times)
            alpha, sigma = log_snr_to_alpha_sigma(noise_level)
            batch_alpha = right_pad_dims_to(input_data, alpha[batch_id])
            batch_sigma = right_pad_dims_to(input_data, sigma[batch_id])
            noised_data = batch_alpha * input_data + batch_sigma * noise

            if self.stage == Stage.first:
                output = self.df(
                    unet_type=self.unet_type, x=noised_data, doctree=None, lr=None, timesteps=noise_level, label=None
                )
                # calc loss
                df_loss = torch.nn.functional.mse_loss(output, input_batch['split_small'])
            elif self.stage == Stage.second:
                output = self.df(
                    unet_type=self.unet_type, x=noised_data, doctree=doctree_in, unet_lr=self.df.unet_lr, timesteps=noise_level, label=None
                )
                # calc loss
                df_loss = torch.nn.functional.mse_loss(output, noise)

            
            # optimize params
            self.optim.zero_grad()
            df_loss.backward()
            self.optim.step()
            update_moving_average(self.df_ema, self.df, self.ema_updater)
            
            # log
            if (self.curr_iter % self.log_iters == 0):
                loop.set_postfix({
                  "loss": {f"{df_loss.detach().item():.4f}"}
                })
            
            # sample (eval)
            if (self.curr_iter % self.sample_iters == 0):
                # explicitly save iter 0 output for sanity check
                loop.set_description_str(f"e: {self.curr_epoch} | sampling")
                self.sample()
            
            # save checkpoint
            if (self.curr_iter % self.chkp_iters == 0):
                # explicitly save iter 0 checkpoint for sanity check
                loop.set_description_str(f"e: {self.curr_epoch} | saving checkpoint")
                self.save_chkp()
        log.INFO("Training terminated.")
        log.INFO("Saving checkpoint...")
        chkp_path = self.save_chkp()
        log.INFO(f"Checkpoint saved at: {chkp_path}")
    

    def export_octree(self, octree: ocnn.octree.Octree, depth: int, filepath: str) -> None:
        """
        Export a mesh from the specified octree

        :param octree The octree to export
        :param depth The depth of the octree to export
        :param path The output filepath
        """
        batch_id = octree.batch_id(depth=depth, nempty=False)
        data = torch.ones([len(batch_id), 1], device=self.device)
        data = octree2voxel(data=data, octree=octree, depth=depth, nempty=False)
        data = data.permute(0, 4, 1, 2, 3).contiguous()

        batch_size = octree.batch_size
        for i in range(batch_size):
            voxel = data[i].squeeze().cpu().numpy()
            mesh = voxel2mesh(voxel)
            if batch_size == 1:
                mesh.export(filepath)


    @torch.no_grad()
    def sample(self, ddim_steps: int=200, truncated_index: float=0.7):
        """
        """
        # sampling batch size is always 1
        out_filename = f"octfusion-stage{self.stage.name}-e{self.curr_epoch}-iter{self.curr_iter}.obj"
        out_filepath = os.path.join(self.experiment_dir, out_filename)
        octree, code = sample(
            self.seed,
            batch_size=1,
            z_shape=self.z_shape,
            ddim_steps=ddim_steps,
            octree_depth=self.autoencoder.depth,
            octree_full_depth=self.autoencoder.full_depth,
            stage=self.stage.name,
            code_channel=self.autoencoder.embed_dim,
            truncated_index=truncated_index,
            unet_type=self.unet_type,
            df_ema=self.df_ema,
            device=self.device
        )

        if self.stage == Stage.first:
            self.export_octree(octree, self.small_depth, out_filepath)
        else:
            # get sdfs
            output = self.autoencoder.decode_code(code, octree)
            bbmin, bbmax = -0.9, 0.9
            sdf_batched = calc_sdf(
                output['neural_mpu'], 1, bbmin=bbmin, bbmax=bbmax
            )
            for i in range(1):
                sdf = sdf_batched[i].cpu().numpy()
                v_pos, t_pos_idx = mcubes(sdf, 0.0)
                if v_pos.size == 0 or t_pos_idx.size == 0:
                    log.WARN(f"Marching cubes returned empty mesh (iter: {self.curr_iter}, batch: {i})")
                    continue
                v_pos = v_pos * ((bbmax - bbmin) / self.sdf_res) + bbmin
                mesh = trimesh.Trimesh(v_pos, t_pos_idx)
                mesh.export(out_filepath)

    
    def __repr__(self):
        _repr = f"{StageTrainer.__name__}(\n"
        _repr += f"\texp_name={self.exp_name}\n"
        _repr += f"\tdevice={self.device}\n"
        _repr += f"\tnepochs={self.nepochs}\n"
        _repr += f"\ttrainset={self.trainset_conf},\n"
        _repr += f"\ttrainloader={self.trainloader_kwargs},\n"
        _repr += f"\tvae=GraphVAE,\n"
        _repr += f"\tvae_kwargs={self.vae_kwargs}"
        _repr += f"\tunet=UNet3D,\n"
        _repr += f"\tunet_kwargs={self.df_kwargs},\n"
        _repr += ")"
        return _repr
    
    def save_chkp(self) -> None:
        """
        Save a checkpoint
        """
        filename = f"{self.exp_name}_iter{self.curr_iter}.ckpt"
        filepath = os.path.join(self.chkp_dir, filename)
        state_dict = {
            'vae': self.autoencoder.state_dict(),
            'optim': self.optim.state_dict(),
            'epoch': self.curr_epoch,
            'iter': self.curr_iter
        }
        state_dict['df_lr'] = self.df.unet_lr.state_dict()
        state_dict['df_ema_lr'] = self.df_ema.unet_lr.state_dict()
        if self.stage == Stage.second:
            state_dict['df_hr'] = self.df.unet_hr.state_dict()
            state_dict['df_ema_hr'] = self.df_ema.unet_hr.state_dict()
        torch.save(state_dict, filepath)
        return filepath

    
    def load_chkp(self, filepath: str) -> None:
        """
        Load a checkpoint from the specified filepath.

        :param filepath The path of the checkpoint file to load.
        """
        state_dict = torch.load(filepath)
        self.autoencoder.load_state_dict(state_dict['vae'])
        self.df.unet_lr.load_state_dict(state_dict['df_lr'])
        self.df_ema.unet_lr.load_state_dict(state_dict['df_ema_lr'])
            
        if self.stage == Stage.second:
            if ("df_hr" in state_dict) and ('df_ema_hr' in state_dict):    
                self.df.unet_hr.load_state_dict(state_dict['df_hr'])
                self.df_ema.unet_hr.load_state_dict(state_dict['df_ema_hr'])
                self.optim.load_state_dict(state_dict['optim'])
                self.curr_epoch = state_dict['epoch']
                self.curr_iter = state_dict['iter']
                self.total_iters = self.total_iters - self.curr_iter
                self.iter_range = range(self.curr_iter, self.total_iters)