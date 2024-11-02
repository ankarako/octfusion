from typing import List, Tuple
import argparse
import os

import torch
import numpy as np
import mesh2sdf
import trimesh
import glob
import ocnn

from tqdm import tqdm
import utils.log as log

def parse_args():
    parser = argparse.ArgumentParser("Teeth3DS octree dataset preproc")
    parser.add_argument("--indir", type=str, help="The root Teeth3DS directory")
    parser.add_argument("--outdir", type=str, help="The output directory to save the preprocessed data.")
    parser.add_argument("--row", type=str, choices=['lower', 'upper'], help="The teeth row to load.")
    parser.add_argument("--sdf_res", type=int, default=128, help="")
    parser.add_argument("--mesh_scale", type=float, default=0.8, help="")
    parser.add_argument("--out_scale", type=float, default=0.5, help="")
    parser.add_argument("--nsamples", type=int, default=100_000, help="")
    args = parser.parse_args()
    return args.indir, args.outdir, args.row, args.sdf_res, args.mesh_scale, args.out_scale, args.nsamples


class State:
    """
    Holds the state of the preprocessing app.
    """
    def __init__(self):
        """
        Initialize a State object.
        """
        self.parser = argparse.ArgumentParser("Teeth3DS octree dataset preproc")
        self.parser.add_argument("--indir", type=str, help="The root Teeth3DS directory")
        self.parser.add_argument("--outdir", type=str, help="The output directory to save the preprocessed data.")
        self.parser.add_argument("--row", type=str, choices=['lower', 'upper'], help="The teeth row to load.")
        self.indir = None
        self.outdir = None
        self.row = None
        self.mesh_scale = 0.8
        self.sdf_res = 256
        self.level = 0.007 # 2/128
        self.output_scale = 0.5

        # point sampling
        self.nsamples = 100_000

    def parse_args(self) -> None:
        """
        Parse cli arguments
        """
        args = self.parser.parse_args()
        self.indir = args.indir
        self.row = args.row
        self.outdir = args.outdir

    def init_dirs(self) -> None:
        assert os.path.exists(self.indir), f"The specified input directory is invalid: {self.indir}."
        
        # create output directory
        if not os.path.exists(self.outdir):
            os.mkdir(self.outdir)

        self.outdir_processed = os.path.join(self.outdir, 'processed', self.row)
        if not os.path.exists(self.outdir_processed):
            os.makedirs(self.outdir_processed)


def process(shape_filename: str, output_folder, state: State, loop: tqdm) -> Tuple[str, str, str]:
        """
        Preprocess a raw .obj sample. Rescale mesh, sample SDF
        remesh according to sampled SDF.

        :return The output sdf, mesh, and bbox filepaths
        """
        # first check if output data exist so we
        # don't do heavy processing without a reason        
        out_sdf_filename = 'sdf.npy'
        out_bbox_filename = 'bbox.npz'
        out_obj_filename = 'mesh.obj'
        out_sdf_filepath = os.path.join(output_folder, out_sdf_filename)
        out_bbox_filepath = os.path.join(output_folder, out_bbox_filename)
        out_obj_filepath = os.path.join(output_folder, out_obj_filename)
        if os.path.exists(out_sdf_filepath) and os.path.exists(out_bbox_filepath) and os.path.exists(out_obj_filepath):
            return out_sdf_filepath, out_obj_filepath, out_bbox_filepath

        shape_filepath = os.path.join(row_dir, shape_filename)
        mesh = trimesh.load(shape_filepath, force='mesh')

        # although meshes are scaled in [-1, 1], perform a 
        # scaling step in [-mesh_scale, mesh_scale]
        # to fit nicely in a unit grid
        verts = mesh.vertices
        bbmin, bbmax = verts.min(0), verts.max(0)
        center = (bbmin + bbmax) * 0.5
        scale = 2.0 * state.mesh_scale / (bbmax - bbmin).max()
        verts = (verts - center) * scale

        # run mesh2sdf
        loop.set_description("sampling sdf")
        sdf, mesh_new = mesh2sdf.compute(
            verts, mesh.faces, state.sdf_res, fix=True, level=state.level, return_mesh=True
        )
        mesh_new.vertices = mesh_new.vertices * state.output_scale
        loop.set_description('processing')
        # save
        np.savez(out_bbox_filepath, bbmax=bbmax, bbmin=bbmin, mul=state.mesh_scale)
        np.save(out_sdf_filepath, sdf)
        mesh_new.export(out_obj_filepath)
        return out_sdf_filepath, out_obj_filepath, out_bbox_filepath


def generate_dataset_sample(
    output_folder: str,
    shape_filename: str, 
    sdf_filepath: str, 
    mesh_filepath: str, 
    bbox_filepath: str, 
    state: State, 
    loop: tqdm
) -> None:
    """
    Generate a dataset sample.
    """
    # first check if output data exist so 
    # we won't heavy calculations without a reason
    out_pnts_filename = "pointcloud.npz"
    out_pnts_filepath = os.path.join(output_folder, out_pnts_filename)

    # sampled sdf filepath
    out_sampled_sdf_filename = 'sdf.npz'
    out_sampled_sdf_filepath = os.path.join(output_folder, out_sampled_sdf_filename)
    if os.path.exists(out_pnts_filepath) and os.path.exists(out_sampled_sdf_filepath):
        return

    # sample points from original mesh
    mesh = trimesh.load(mesh_filepath, force='mesh')
    points, idx = trimesh.sample.sample_surface(mesh, state.nsamples)
    normals = mesh.face_normals[idx]

    # save points
    np.savez(out_pnts_filepath, points=points.astype(np.float16), normals=normals.astype(np.float16))

    # sample sdf
    # constants
    depth, full_depth = 6, 4
    sample_num = 4 # number of samples in each octree node
    grid = np.array([
        [0, 0, 0], 
        [0, 0, 1], 
        [0, 1, 0], 
        [0, 1, 1],
        [1, 0, 0], 
        [1, 0, 1], 
        [1, 1, 0], 
        [1, 1, 1]
    ])

    loop.set_description_str("sampling sdf")
    pts = np.load(out_pnts_filepath)
    sdf = np.load(sdf_filepath)
    sdf = torch.from_numpy(sdf)
    points = pts['points'].astype(np.float32)
    normals = pts['normals'].astype(np.float32)
    points = points / state.output_scale

    # build octree
    loop.set_description_str("creating octree")
    points = ocnn.octree.Points(torch.from_numpy(points), torch.from_numpy(normals))
    octree = ocnn.octree.Octree(depth=depth, full_depth=full_depth)
    octree.build_octree(points)

    # sample points and grads according to vertex positions
    xyzs, grads, sdfs = [], [], []
    for d in range(full_depth, depth + 1):
        xyzb = octree.xyzb(d)
        x, y, z, b = xyzb
        xyz = torch.stack([x, y, z], dim=1).float()

        # sample k points in each octree node
        xyz = xyz.unsqueeze(1) + torch.rand(xyz.shape[0], sample_num, 3)
        xyz = xyz.view(-1, 3)
        xyz = xyz * (state.sdf_res / 2 ** d)
        xyz = xyz[(xyz < state.sdf_res - 1).all(dim=1)]
        xyzs.append(xyz)

        # interpolate sdf values
        xyzi = torch.floor(xyz)
        corners = xyzi.unsqueeze(1) + grid          # N x 8 x 3
        coordsf = xyz.unsqueeze(1) - corners        # N x 8 x 3 in [-1, 1]
        weights = (1 - coordsf.abs()).prod(dim=-1)  # N x 8 x 1
        corners = corners.long().view(-1, 3)
        x, y, z = corners[:, 0], corners[:, 1], corners[:, 2]
        s = sdf[x, y, z].view(-1, 8)
        sw = torch.sum(s * weights, dim=1)
        sdfs.append(sw)

        # calc gradients
        gx = s[:, 4] - s[:, 0] + s[:, 5] - s[:, 1] + \
                    s[:, 6] - s[:, 2] + s[:, 7] - s[:, 3]    # noqa
        gy = s[:, 2] - s[:, 0] + s[:, 3] - s[:, 1] + \
                    s[:, 6] - s[:, 4] + s[:, 7] - s[:, 5]    # noqa
        gz = s[:, 1] - s[:, 0] + s[:, 3] - s[:, 2] + \
                    s[:, 5] - s[:, 4] + s[:, 7] - s[:, 6]    # noqa
        
        grad = torch.stack([gx, gy, gz], dim=-1)
        norm = torch.sqrt(torch.sum(grad ** 2, dim=-1, keepdim=True))
        grad = grad / (norm + 1.0-8)
        grads.append(grad)
    
    # concat results
    xyz = torch.cat(xyzs, dim=0).numpy()
    points = (xyz / 64 - 1).astype(np.float16) * state.output_scale
    grads = torch.cat(grads, dim=0).numpy().astype(np.float16)
    sdfs = torch.cat(sdfs, dim=0).numpy().astype(np.float16)

    # save results
    np.savez(out_sampled_sdf_filepath, points=points, grad=grads, sdf=sdfs)


if __name__ == "__main__":
    state = State()
    state.parse_args()
    state.init_dirs()

    # get all shapes in the specified input directory
    row_dir = os.path.join(state.indir, state.row)
    shape_filepaths = glob.glob(os.path.join(row_dir, '*.obj'))

    loop = tqdm(shape_filepaths, total=len(shape_filepaths))
    for shape_filename in loop:
        loop.set_description_str(f"preprocessing")

        # create output directory for preprocessed sample
        shape_name = os.path.basename(shape_filename).replace('.obj', '')
        output_folder = os.path.join(state.outdir_processed, shape_name)
        if not os.path.exists(output_folder):
            os.mkdir(output_folder)
        sdf_filepath, mesh_filepath, bbox_filepath = process(shape_filename, output_folder, state, loop)

        # create output directory for dataset sample
        loop.set_description_str("generating dataset sample")
        generate_dataset_sample(output_folder, shape_filename, sdf_filepath, mesh_filepath, bbox_filepath, state, loop)
        loop.update(1)