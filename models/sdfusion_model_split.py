# Reference: diffusion is borrowed from the LDM repo: https://github.com/CompVis/latent-diffusion
# Specifically, functions from: https://github.com/CompVis/latent-diffusion/blob/main/ldm/models/diffusion/ddpm.py

import os
from collections import OrderedDict
from functools import partial
import copy

import numpy as np
from omegaconf import OmegaConf
from termcolor import colored, cprint
from einops import rearrange, repeat
from tqdm import tqdm
from random import random
import ocnn
from ocnn.nn import octree2voxel, octree_pad
from ocnn.octree import Octree, Points
from models.networks.dualoctree_networks import dual_octree


import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.special import expm1

import torchvision.utils as vutils
import torchvision.transforms as transforms

from models.base_model import BaseModel
from models.networks.diffusion_networks.network import DiffusionUNet
from models.networks.diffusion_networks.ldm_diffusion_util import *

from models.networks.diffusion_networks.samplers.ddim_new import DDIMSampler

# distributed
from utils.distributed import reduce_loss_dict

# rendering
from utils.util_3d import init_mesh_renderer, render_sdf, render_sdf_dualoctree
from utils.util_dualoctree import calc_sdf

TRUNCATED_TIME = 0.7

class SDFusionModel(BaseModel):
    def name(self):
        return 'SDFusion-Model'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain
        self.model_name = self.name()
        self.device = opt.device
        self.gradient_clip_val = 1.
        self.start_iter = opt.start_iter


        ######## START: Define Networks ########
        assert opt.df_cfg is not None
        assert opt.vq_cfg is not None

        # init df
        df_conf = OmegaConf.load(opt.df_cfg)
        vq_conf = OmegaConf.load(opt.vq_cfg)

        self.vq_conf = vq_conf
        self.solver = self.vq_conf.solver
        self.depth = self.vq_conf.model.depth
        self.full_depth = self.vq_conf.model.full_depth
        self.voxel_size = 2 ** self.depth

        # init diffusion networks
        df_model_params = df_conf.model.params
        unet_params = df_conf.unet.params
        self.conditioning_key = df_model_params.conditioning_key
        self.num_timesteps = df_model_params.timesteps
        self.thres = 0.5
        if self.conditioning_key == 'adm':
            self.num_classes = unet_params.num_classes
        elif self.conditioning_key == 'None':
            self.num_classes = 1
        self.df = DiffusionUNet(unet_params, conditioning_key=self.conditioning_key)
        self.df.to(self.device)

        # record z_shape
        code_channel = 8
        z_sp_dim = 2 ** self.full_depth
        self.z_shape = (code_channel, z_sp_dim, z_sp_dim, z_sp_dim)

        self.ema_df = copy.deepcopy(self.df)
        self.ema_df.to(self.device)
        if opt.isTrain:
            self.ema_rate = opt.ema_rate
            self.ema_updater = EMA(self.ema_rate)
            self.reset_parameters()
            set_requires_grad(self.ema_df, False)

        self.noise_schedule = 'cosine'
        if self.noise_schedule == "linear":
            self.log_snr = beta_linear_log_snr
        elif self.noise_schedule == "cosine":
            self.log_snr = alpha_cosine_log_snr
        else:
            raise ValueError(f'invalid noise schedule {self.noise_schedule}')

        ######## END: Define Networks ########

        if self.isTrain:

            # initialize optimizers
            self.optimizer = optim.AdamW([p for p in self.df.parameters() if p.requires_grad == True], lr=opt.lr)
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, 1000, 0.9)

            self.optimizers = [self.optimizer]
            self.schedulers = [self.scheduler]

            self.print_networks(verbose=False)

        if opt.ckpt is not None:
            self.load_ckpt(opt.ckpt, load_opt=self.isTrain)
            if self.isTrain:
                self.optimizers = [self.optimizer]
            # self.schedulers = [self.scheduler]


        # setup renderer
        if 'snet' in opt.dataset_mode:
            dist, elev, azim = 1.7, 20, 20
        elif 'pix3d' in opt.dataset_mode:
            dist, elev, azim = 1.7, 20, 20
        elif opt.dataset_mode == 'buildingnet':
            dist, elev, azim = 1.0, 20, 20

        self.renderer = init_mesh_renderer(image_size=256, dist=dist, elev=elev, azim=azim, device=self.device)

        # for distributed training
        if self.opt.distributed:
            self.make_distributed(opt)
            self.df_module = self.df.module

        else:
            self.df_module = self.df

        self.ddim_steps = 200
        if self.opt.debug == "1":
            # NOTE: for debugging purpose
            self.ddim_steps = 7
        cprint(f'[*] setting ddim_steps={self.ddim_steps}', 'blue')

    def reset_parameters(self):
        self.ema_df.load_state_dict(self.df.state_dict())

    def make_distributed(self, opt):
        self.df = nn.parallel.DistributedDataParallel(
            self.df,
            device_ids=[opt.local_rank],
            output_device=opt.local_rank,
            broadcast_buffers=False,
        )

    ############################ START: init diffusion params ############################

    def batch_to_cuda(self, batch):
        def points2octree(points):
            octree = ocnn.octree.Octree(depth = 6, full_depth = 4)
            octree.build_octree(points)
            return octree

        points = [pts.cuda(non_blocking=True) for pts in batch['points']]
        octrees = [points2octree(pts) for pts in points]
        octree = ocnn.octree.merge_octrees(octrees)
        octree.construct_all_neigh()
        batch['octree_in'] = octree

        batch['split'] = self.octree2split(batch['octree_in'])

    def set_input(self, input=None):
        self.batch_to_cuda(input)
        self.x = input['split']
        self.octree_in = input['octree_in']
        self.label = input['label']

    def switch_train(self):
        self.df.train()

    def switch_eval(self):
        self.df.eval()


    # check: ddpm.py, line 871 forward
    # check: p_losses
    # check: q_sample, apply_model


    def points2octree(self, points):
        points_in = Points(points = points.float())
        points_in.clip(min=-1, max=1)
        octree = Octree(self.depth, self.full_depth)
        octree.build_octree(points_in)
        return octree

    def forward(self):

        self.df.train()

        c = None

        split = self.x

        batch = split.shape[0]
        times = torch.zeros((batch,), device = self.device).float().uniform_(0, 1)

        # times = torch.zeros((batch,), device = self.device) + 0.9

        noise = torch.randn_like(split)

        noise_level = self.log_snr(times)
        padded_noise_level = right_pad_dims_to(split, noise_level)
        alpha, sigma = log_snr_to_alpha_sigma(padded_noise_level)
        noised_split = alpha * split + sigma * noise

        noised_octree = self.split2octree(noised_split)

        # print(times)
        # self.export_octree(self.octree_in, save_dir = 'airplane_gt')
        # self.export_octree(noised_octree, save_dir = 'noised_airplane')

        noised_doctree = dual_octree.DualOctree(noised_octree)
        noised_doctree.post_processing_for_docnn()
        doctree_in = noised_doctree

        input_data = torch.zeros((doctree_in.total_num,1), device = self.device)

        doctree_gt = dual_octree.DualOctree(self.octree_in)
        doctree_gt.post_processing_for_docnn()

        output_data = torch.zeros((doctree_gt.total_num,1), device = self.device)

        out, logits, doctree_out = self.df(input_data, doctree_in = noised_doctree, doctree_out = doctree_gt, t = noise_level)

        # out, logits, doctree_out = self.df(input_data, doctree_in = noised_doctree, doctree_out = None, t = noise_level)
        # self.export_octree(doctree_out.octree, save_dir = 'pred_airplane')

        self.df_loss = F.mse_loss(out, output_data)

        self.logit_loss = 0.
        for d in logits.keys():
            logitd = logits[d]
            label_gt = self.octree_in.nempty_mask(d).long()
            self.logit_loss += F.cross_entropy(logitd, label_gt)

        self.loss = self.df_loss + self.logit_loss

    def get_sampling_timesteps(self, batch, device, steps):
        times = torch.linspace(1., 0., steps + 1, device=device)
        times = repeat(times, 't -> b t', b=batch)
        times = torch.stack((times[:, :-1], times[:, 1:]), dim=0)
        times = times.unbind(dim=-1)
        return times

    @torch.no_grad()
    def uncond(self, batch_size=16, steps=200, category = 0, ema = True, truncated_index: float = 0.0, save_dir = None, index = 0):

        if ema:
            self.ema_df.eval()
        else:
            self.df.eval()

        shape = (batch_size, *self.z_shape)

        time_pairs = self.get_sampling_timesteps(
            batch_size, device=self.device, steps=steps)

        split = torch.randn(shape, device = self.device)
        x_start = None
        label = torch.randint(0, self.num_classes,(batch_size,), device = self.device)

        _iter = tqdm(time_pairs, desc='sampling loop time step')

        for time, time_next in _iter:

            log_snr = self.log_snr(time)
            log_snr_next = self.log_snr(time_next)
            log_snr, log_snr_next = map(
                partial(right_pad_dims_to, split), (log_snr, log_snr_next))

            alpha, sigma = log_snr_to_alpha_sigma(log_snr)
            alpha_next, sigma_next = log_snr_to_alpha_sigma(log_snr_next)

            noise_cond = self.log_snr(time)

            noised_octree = self.split2octree(split)

            noised_doctree = dual_octree.DualOctree(noised_octree)
            noised_doctree.post_processing_for_docnn()

            input_data = torch.zeros((noised_doctree.total_num, 1), device = self.device)

            if ema:
                _,_, doctree_out = self.ema_df(input_data, doctree_in = noised_doctree, doctree_out = None, t = noise_cond)
            else:
                _,_, doctree_out = self.df(input_data, doctree_in = noised_doctree, doctree_out = None, t = noise_cond)

            self.export_octree(doctree_out.octree, save_dir = 'pred_airplane', index = time.item())

            x_start = self.octree2split(doctree_out.octree)

            c = -expm1(log_snr - log_snr_next)

            mean = alpha_next * (split * (1 - c) / alpha + c * x_start)
            variance = (sigma_next ** 2) * c

            split = mean + torch.sqrt(variance) * torch.randn_like(split)

            # mean = alpha_next * (split * (1 - c) / alpha + c * x_start)
            # variance = (sigma_next ** 2) * c
            # noise = torch.where(
            #     rearrange(time_next > truncated_index, 'b -> b 1 1 1 1'),
            #     torch.randn_like(split),
            #     torch.zeros_like(split)
            # )
            # split = mean + torch.sqrt(variance) * noise

        print(split.max())
        print(split.min())

        octree_out = self.split2octree(split)
        self.export_octree(octree_out, save_dir, index)

        doctree_out = dual_octree.DualOctree(octree_out)
        doctree_out.post_processing_for_docnn()

        return doctree_out

    def get_doctree_data(self, doctree):

        data = torch.zeros([doctree.total_num,2])

        num_full_depth = doctree.lnum[self.full_depth]
        num_full_depth_p1 = doctree.lnum[self.full_depth + 1]

        data[:num_full_depth] = torch.tensor([-1,-1])
        data[num_full_depth : num_full_depth + num_full_depth_p1] = torch.tensor([1,-1])
        data[num_full_depth + num_full_depth_p1 :] = torch.tensor([1,1])
        data = data.to(self.device)
        return data

    def octree2split(self, octree):

        child_full_p1 = octree.children[self.full_depth + 1]
        split_full_p1 = (child_full_p1 >= 0)
        split_full_p1 = split_full_p1.reshape(-1, 8)
        split_full = octree_pad(data = split_full_p1, octree = octree, depth = self.full_depth)
        split_full = octree2voxel(data=split_full, octree=octree, depth = self.full_depth)
        split_full = split_full.permute(0,4,1,2,3).contiguous()

        split_full = split_full.float()
        split_full = 2 * split_full - 1

        # scale to [-2, 2]
        split_full = split_full * 4

        return split_full

    def split2octree(self, split):

        split[split > 0] = 1
        split[split < 0] = 0

        batch_size = split.shape[0]
        octree_out = create_full_octree(depth = self.depth, full_depth = self.full_depth, batch_size = batch_size, device = self.device)
        split_sum = torch.sum(split, dim = 1)
        nempty_mask_voxel = (split_sum > 0)
        x, y, z, b = octree_out.xyzb(self.full_depth)
        nempty_mask = nempty_mask_voxel[b,x,y,z]
        label = nempty_mask.long()
        octree_out.octree_split(label, self.full_depth)
        octree_out.octree_grow(self.full_depth + 1)
        octree_out.depth += 1

        x, y, z, b = octree_out.xyzb(depth = self.full_depth, nempty = True)
        nempty_mask_p1 = split[b,:,x,y,z]
        nempty_mask_p1 = nempty_mask_p1.reshape(-1)
        label_p1 = nempty_mask_p1.long()
        octree_out.octree_split(label_p1, self.full_depth + 1)
        octree_out.octree_grow(self.full_depth + 2)
        octree_out.depth += 1

        return octree_out

    def export_octree(self, octree, save_dir = None, index = 0):

        if not os.path.exists(save_dir): os.makedirs(save_dir)

        batch_id = octree.batch_id(depth = self.depth, nempty = False)
        data = torch.ones((len(batch_id), 1), device = self.device)
        data = octree2voxel(data = data, octree = octree, depth = self.depth, nempty = False)
        data = data.permute(0,4,1,2,3).contiguous()

        batch_size = octree.batch_size

        for i in tqdm(range(batch_size)):
            voxel = data[i].squeeze().cpu().numpy()
            mesh = voxel2mesh(voxel)
            if batch_size == 1:
                mesh.export(os.path.join(save_dir, f'{index}.obj'))
            else:
                mesh.export(os.path.join(save_dir, f'{i}.obj'))


    def get_sdfs(self, neural_mpu, batch_size, bbox):
        # bbox used for marching cubes
        if bbox is not None:
            self.bbmin, self.bbmax = bbox[:3], bbox[3:]
        else:
            sdf_scale = self.solver.sdf_scale
            self.bbmin, self.bbmax = -sdf_scale, sdf_scale    # sdf_scale = 0.9

        self.sdfs = calc_sdf(neural_mpu, batch_size, size = self.solver.resolution, bbmin = self.bbmin, bbmax = self.bbmax)

    def get_mesh(self,ngen, save_dir, level = 0):
        sdf_values = self.sdfs
        size = self.solver.resolution
        bbmin = self.bbmin
        bbmax = self.bbmax
        mesh_scale=self.vq_conf.data.test.point_scale
        for i in range(ngen):
            filename = os.path.join(save_dir, f'{i}.obj')
            sdf_value = sdf_values[i].cpu().numpy()
            vtx, faces = np.zeros((0, 3)), np.zeros((0, 3))
            try:
                vtx, faces, _, _ = skimage.measure.marching_cubes(sdf_value, level)
            except:
                pass
            if vtx.size == 0 or faces.size == 0:
                print('Warning from marching cubes: Empty mesh!')
                return
            vtx = vtx * ((bbmax - bbmin) / size) + bbmin   # [0,sz]->[bbmin,bbmax]  把vertex放缩到[bbmin, bbmax]之间
            vtx = vtx * mesh_scale
            mesh = trimesh.Trimesh(vtx, faces)  # 利用Trimesh创建mesh并存储为obj文件。
            mesh.export(filename)

    @torch.no_grad()
    def eval_metrics(self, dataloader, thres=0.0, global_step=0):
        self.eval()

        ret = OrderedDict([
            ('dummy_metrics', 0.0),
        ])
        self.train()
        return ret

    def backward(self):

        self.loss.backward()

    def update_EMA(self):
        update_moving_average(self.ema_df, self.df, self.ema_updater)

    def optimize_parameters(self):

        self.set_requires_grad([self.df], requires_grad=True)

        self.forward()
        self.optimizer.zero_grad()
        self.backward()
        self.optimizer.step()
        self.update_EMA()

    def get_current_errors(self):

        ret = OrderedDict([
            ('diffusion', self.df_loss.data),
            ('logit', self.logit_loss.data),
            ('total', self.loss.data),
        ])

        if hasattr(self, 'loss_gamma'):
            ret['gamma'] = self.loss_gamma.data

        return ret

    def get_current_visuals(self):

        with torch.no_grad():
            self.img_gen_df = render_sdf_dualoctree(self.renderer, self.sdfs, level=0,
                                                bbmin = self.bbmin, bbmax = self.bbmax,
                                                mesh_scale = self.vq_conf.data.test.point_scale, render_all = True)
            # self.img_gen_df = render_sdf(self.renderer, self.gen_df)

        vis_tensor_names = [
            'img_gen_df',
        ]

        vis_ims = self.tnsrs2ims(vis_tensor_names)
        visuals = zip(vis_tensor_names, vis_ims)

        return OrderedDict(visuals)

    def save(self, label, global_iter, save_opt=True):

        state_dict = {
            'df': self.df_module.state_dict(),
            'ema_df': self.ema_df.state_dict(),
            'opt': self.optimizer.state_dict(),
            'global_step': global_iter,
        }

        # if save_opt:
        #     state_dict['opt'] = self.optimizer.state_dict()

        save_filename = 'df_%s.pth' % (label)
        save_path = os.path.join(self.opt.ckpt_dir, save_filename)

        ckpts = os.listdir(self.opt.ckpt_dir)
        ckpts = [ck for ck in ckpts if ck!='df_steps-latest.pth']
        ckpts.sort(key=lambda x: int(x[9:-4]))
        if len(ckpts) > self.opt.ckpt_num:
            for ckpt in ckpts[:-self.opt.ckpt_num]:
                os.remove(os.path.join(self.opt.ckpt_dir, ckpt))

        torch.save(state_dict, save_path)

    def load_ckpt(self, ckpt, load_opt=True):
        map_fn = lambda storage, loc: storage
        if type(ckpt) == str:
            state_dict = torch.load(ckpt, map_location=map_fn)
        else:
            state_dict = ckpt

        # self.vqvae.load_state_dict(state_dict['vqvae'])
        self.df.load_state_dict(state_dict['df'])
        self.ema_df.load_state_dict(state_dict['ema_df'])
        self.start_iter = state_dict['global_step']
        print(colored('[*] weight successfully load from: %s' % ckpt, 'blue'))

        if load_opt:
            self.optimizer.load_state_dict(state_dict['opt'])
            print(colored('[*] optimizer successfully restored from: %s' % ckpt, 'blue'))
