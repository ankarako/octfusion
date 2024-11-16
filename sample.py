from typing import List, Dict, Any
import argparse
import utils
from sampler import sample
from datetime import datetime
import os
import utils
import copy

import torch

from models.networks.dualoctree_networks.graph_vae import GraphVAE
from models.networks.diffusion_networks.graph_unet_union import UNet3DModel
from utils.util_dualoctree import (
    octree2split_small, octree2voxel, calc_sdf
)
from utils.mesh import mcubes
from tqdm import tqdm
import trimesh

class SamplingState:
    """
    A wrapper around StageTrainer's initialization
    keyword arguments.
    Helps keeping the same configuration files for 
    stage training and sampling
    """
    def __init__(
        self,
        seed: int,
        device: List[int],
        chkp_filepath: str,
        sdf_res: int,
        vae_kwargs: Dict[str, Any],
        df_kwargs: Dict[str, Any],
        **kwargs
    ):
        self.seed = seed
        self.sdf_res = sdf_res
        self.sample_iters = 200
        utils.state.seed(seed)
        self.device = utils.state.get_device(device)

        # instantiate vae
        self.autoencoder = GraphVAE(**vae_kwargs).to(self.device)

        # instantiate diffusion model
        self.df = UNet3DModel(**df_kwargs, stage_flag="hr", use_checkpoint=False).to(self.device)
        self.split_channel = 8
        self.code_channel = self.autoencoder.embed_dim
        z_sp_dim = 2 ** self.autoencoder.full_depth
        self.z_shape = (self.split_channel, z_sp_dim, z_sp_dim, z_sp_dim)

        self.df_ema = copy.deepcopy(self.df)
        self.df_ema = self.df_ema.to(self.device)
        
        if not os.path.exists(chkp_filepath):
            utils.log.ERROR(f"You need to specify a diffusion model checkpoint.")
            return
        
        state_dict = torch.load(chkp_filepath)
        self.autoencoder.load_state_dict(state_dict['vae'])
        
        self.df.unet_lr.load_state_dict(state_dict['df_lr'])
        self.df_ema.unet_lr.load_state_dict(state_dict['df_ema_lr'])
        
        self.df.unet_hr.load_state_dict(state_dict['df_hr'])
        self.df_ema.unet_hr.load_state_dict(state_dict['df_ema_hr'])



if __name__ == "__main__":
    parser = argparse.ArgumentParser("Sample from a trained OctFusion model")
    parser.add_argument("--conf", type=str, help="Path to the configuration file to load.")
    parser.add_argument("--nsamples", type=int, help="The number of samples to generate")
    parser.add_argument("--out", type=str, help="The directory to save the sampled data.")
    args = parser.parse_args()

    # read configuration
    conf = utils.conf.read_conf(args.conf)

    # check and create output folder
    now = datetime.now().strftime("%d-%m-%y-%H-%M-%S")
    output_folder = f"samples-{now}"

    if not os.path.exists(args.out):
        utils.log.ERROR(f"The specified output directory is invalid: {args.out}")
    output_dir = os.path.join(args.out, output_folder)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # initialize configuration
    sampling_state = SamplingState(**conf.kwargs)


    # start sampling
    sampling_state.autoencoder.eval()
    sampling_state.df.eval()
    sampling_state.df_ema.eval()
    with torch.no_grad():
        sampling_loop = tqdm(range(args.nsamples), desc="Sampling", ncols=100)
        for sample_id in sampling_loop:
            out_filename = f"octfusion-sample-{sample_id}.obj"
            out_filepath = os.path.join(output_dir, out_filename)
            octree, code = sample(
                sampling_state.seed,
                batch_size=1,
                z_shape=sampling_state.z_shape,
                ddim_steps=sampling_state.sample_iters,
                octree_depth=sampling_state.autoencoder.depth,
                octree_full_depth=sampling_state.autoencoder.full_depth,
                stage='second',
                code_channel=sampling_state.autoencoder.embed_dim,
                truncated_index=0.7,
                unet_type="hr",
                df_ema=sampling_state.df_ema,
                device=sampling_state.device
            )

            output = sampling_state.autoencoder.decode_code(code, octree)
            bbmin, bbmax = -0.9, 0.9
            sdf_batched = calc_sdf(output['neural_mpu'], 1, bbmin=bbmin, bbmax=bbmax)
            sdf = sdf_batched[0].cpu().numpy()
            v_pos, t_pos_idx = mcubes(sdf, 0.0)
            if v_pos.size == 0 or t_pos_idx.size == 0:
                utils.log.WARN(f"Marching cubes returned empty mesh (iter: {sample_id})")
                continue
            v_pos = v_pos * ((bbmax - bbmin) / sampling_state.sdf_res) + bbmin
            mesh = trimesh.Trimesh(v_pos, t_pos_idx)
            mesh.export(out_filepath)