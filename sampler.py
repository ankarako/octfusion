from typing import List
from functools import partial
import torch
from torch.special import expm1

from einops import repeat
from models.networks.dualoctree_networks.graph_vae import dual_octree
from models.networks.diffusion_networks.ldm_diffusion_util import (
    beta_linear_log_snr,
    log_snr_to_alpha_sigma,
    right_pad_dims_to,
    voxel2mesh
)
import ocnn

import utils
from utils.util_dualoctree import (
    octree2split_small, split2octree_small, octree2voxel
)
from tqdm import tqdm

@torch.no_grad()
def sample_loop(
    batch_size: int,
    shape: List[int],
    doctree_lr=None,
    ddim_steps: int=200,
    label: int=None,
    unet_type: str='lr',
    df_ema: torch.nn.Module=None,
    unet_lr: torch.nn.Module=None,
    df_type: str='x0',
    truncated_index: float=0.0,
    device: torch.device=torch.device('cuda')
):
    """
    """
    # get sampling timesteps
    times = torch.linspace(1., 0., ddim_steps + 1, device=device)
    times = repeat(times, 't -> b t', b=batch_size)
    times = torch.stack([times[:, :-1], times[:, 1:]], dim=0)
    times = times.unbind(dim=-1)

    noised_data = torch.randn(shape, device=device)

    x_start = None
    loop = tqdm(times, desc="samnpling loop: small", ncols=100)
    for t, t_next in loop:
        log_snr = beta_linear_log_snr(t)
        log_snr_next = beta_linear_log_snr(t_next)
        noise_cond = log_snr

        output = df_ema(
            unet_type=unet_type, 
            x=noised_data, 
            doctree=doctree_lr, 
            timesteps=noise_cond,
            unet_lr=unet_lr,
            x_self_cond=x_start,
            label=None
        )

        if t[0] < truncated_index and unet_type == "lr":
            output.sign_()
        
        if df_type == 'x0':
            x_start = output
            padded_log_snr, padded_log_snr_next = map(
                partial(right_pad_dims_to, noised_data), (log_snr, log_snr_next)
            )
            alpha, sigma = log_snr_to_alpha_sigma(padded_log_snr)
            alpha_next, sigma_next = log_snr_to_alpha_sigma(padded_log_snr_next)

            c = -expm1(padded_log_snr - padded_log_snr_next)
            mean = alpha_next * (noised_data * (1 - c) / alpha + c * output)
            variance = (sigma_next ** 2) * c

            noise = torch.where(
                right_pad_dims_to(noised_data, t_next > truncated_index),
                torch.randn_like(noised_data),
                torch.zeros_like(noised_data)
            )
            noised_data = mean + torch.sqrt(variance) * noise
        elif df_type == 'eps':
            alpha, sigma = log_snr_to_alpha_sigma(log_snr)
            alpha_next, sigma_next = log_snr_to_alpha_sigma(log_snr_next)
            alpha, sigma, alpha_next, sigma_next = alpha[0], sigma[0], alpha_next[0], sigma_next[0]
            x_start = (noised_data - output * sigma) / alpha.clamp(min=1e-8)
            noised_data = x_start * alpha_next + output * sigma_next
    return noised_data


@torch.no_grad()
def sample(
    seed: int,
    batch_size: int, 
    z_shape: List[int],
    ddim_steps: int, 
    octree_depth: int,
    octree_full_depth: int,
    stage: str,
    code_channel: int,
    truncated_index: float=0.7,
    unet_type: str='lr',
    df_ema: torch.nn.Module=None,
    device: torch.device=torch.device('cuda')
):
    """
    Sample a data point from the specified 
    model.

    :param batch_size The batch size of the data point to sample.
    :param ddim_steps The number of denoising steps.
    :param truncated_index
    :param model The unet model to utilize.
    :param device
    """
    assert stage in ['first', 'second']
    df_ema.eval()

    shape = [batch_size, *z_shape]
    split_small = sample_loop(
        batch_size=batch_size,
        shape=shape,
        doctree_lr=None,
        ddim_steps=ddim_steps,
        label=None,
        unet_type='lr',
        unet_lr=None,
        df_ema=df_ema,
        df_type='x0',
        truncated_index=truncated_index,
        device=device
    )
    octree_small: ocnn.octree.Octree = split2octree_small(split_small, octree_depth, octree_full_depth)

    if stage == 'first':
        return octree_small, None
    
    doctree_small = dual_octree.DualOctree(octree_small)
    doctree_small.post_processing_for_docnn()
    doctree_small_num = doctree_small.total_num

    samples = sample_loop(
        batch_size=batch_size,
        doctree_lr=doctree_small,
        shape=[doctree_small_num, code_channel],
        df_ema=df_ema,
        ddim_steps=ddim_steps,
        label=None,
        unet_type='hr',
        unet_lr=df_ema.unet_lr,
        df_type='eps'
    )
    return doctree_small, samples

