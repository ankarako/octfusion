from typing import List, Tuple
import argparse
import os

import torch
import numpy as np
import mesh2sdf
import trimesh
import glob
import ocnn
from plyfile import PlyData, PlyElement

from tqdm import tqdm
import utils.log as log
from utils.viz_octree import plot_octree


def visualize_sdf(sdf: np.ndarray, sdf_res: int):
    import matplotlib.pyplot as plt
    x_res, y_res, z_res = sdf_res, sdf_res, sdf_res
    sdf_reshaped = sdf.reshape([x_res, y_res, z_res])

    # create 2d plot for a slice of the SDF
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # take the middle slice along the z axis
    slice_index = z_res // 2
    X, Y = np.meshgrid(np.arange(x_res), np.arange(y_res))
    Z = sdf_reshaped[:, :, slice_index]

    contour = ax.contourf(X, Y, Z, levels=100, cmap='viridis')
    fig.colorbar(contour)
    plt.show()

def parse_args():
    parser = argparse.ArgumentParser("Teeth3DS octree dataset preproc")
    parser.add_argument("--indir", type=str, help="The root Teeth3DS directory")
    parser.add_argument("--outdir", type=str, help="The output directory to save the preprocessed data.")
    parser.add_argument("--row", type=str, choices=['lower', 'upper'], help="The teeth row to load.")
    parser.add_argument("--sdf_res", type=int, default=128, help="")
    parser.add_argument("--mesh_scale", type=float, default=0.8, help="")
    parser.add_argument("--out_scale", type=float, default=0.5, help="")
    parser.add_argument("--n_pts_samples", type=int, default=200_000, help="")
    parser.add_argument("--debug", action='store_true', default=False)
    args = parser.parse_args()
    return args.indir, args.outdir, args.row, args.sdf_res, args.mesh_scale, args.out_scale, args.n_pts_samples, args.debug


def process_mesh2sdf(src_filepath: str, shape_id: str,  output_dir: str, mesh_scale: float, sdf_res: int) -> Tuple[str, str, str]:
    """
    Samples sdf values for the specified mesh.

    :param src_filepath The path to the .obj file to load.
    :param output_dir The directory to output the extracted data
    """
    out_filepath_obj = os.path.join(output_dir, shape_id + '_mesh.obj')
    out_filepath_bbox = os.path.join(output_dir, shape_id + '_bbox.npz')
    out_filepath_sdf = os.path.join(output_dir, shape_id + '_sdf.npy')
    
    # check if this sample is already processed so we don't
    # do any heavy calculations with no reasons
    shape_processed = os.path.exists(out_filepath_obj) and os.path.exists(out_filepath_bbox) and os.path.exists(out_filepath_sdf)
    if shape_processed:
        return out_filepath_bbox, out_filepath_obj, out_filepath_sdf

    mesh = trimesh.load(src_filepath, force='mesh')
    vertices = mesh.vertices
    
    bbmin, bbmax = vertices.min(0), vertices.max(0)
    center = (bbmin + bbmax) * 0.5
    
    # center mesh
    vertices = (vertices - center)

    # scale mesh
    max_extent = (bbmax - bbmin).max()
    scale = 2.0 * mesh_scale / max_extent
    vertices = vertices * scale
    
    # run mesh2sdf
    # make mesh watertight, calculate sdf, and return watertight mesh
    level = 2 / float(sdf_res)
    sdf,  mesh_new = mesh2sdf.compute(
        vertices, mesh.faces, sdf_res, fix=True, level=level, return_mesh=True
    )

    # save output
    np.savez(out_filepath_bbox, bbmax=bbmax, bbmin=bbmin, mul=mesh_scale)
    np.save(out_filepath_sdf, sdf)
    mesh_new.export(out_filepath_obj)
    return out_filepath_bbox, out_filepath_obj, out_filepath_sdf


def sample_pts_from_mesh(mesh_filepath: str, shape_id: str, output_dir: str, nsamples: int) -> Tuple[str, str]:
    """
    Sample points from the repaired mesh
    """
    out_filepath_pts = os.path.join(output_dir, shape_id + '_pcl.npz')
    out_filepath_ply = os.path.join(output_dir, shape_id + '_pcl.ply')
    shape_processed = os.path.exists(out_filepath_pts) and os.path.exists(out_filepath_ply)
    if shape_processed:
        return out_filepath_pts, out_filepath_ply
    
    mesh = trimesh.load(mesh_filepath, force='mesh')
    points, idx = trimesh.sample.sample_surface(mesh, nsamples)
    normals = mesh.face_normals[idx]

    # save output as npz
    np.savez(out_filepath_pts, points=points.astype(np.float16), normals=normals.astype(np.float16))

    # save output as ply (for debugging reasons mostly)
    vertices = np.array([
        (points[i, 0], points[i, 1], points[i, 2], normals[i, 0], normals[i, 1], normals[i, 2]) for i in range(nsamples)
        ], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4')]
    )
    vertex_element = PlyElement.describe(vertices, 'vertex')
    PlyData([vertex_element], text=True).write(out_filepath_ply)
    return out_filepath_pts, out_filepath_ply


def sample_gt_sdf(sdf_filepath: str, pts_filepath: str, shape_id: str, output_dir: str, sdf_res: int, debug: bool) -> Tuple[str]:
    """
    Sample SDF for ground truth training
    """
    out_filepath_sdf = os.path.join(output_dir, shape_id + '_sdf.npz')
    if os.path.exists(out_filepath_sdf):
        return out_filepath_sdf
    
    # dunno why they use this in octfusion dataset generation
    # First they scale the object in [-0.8, 0.8]
    # and here they rescale it in [-0.5, 0.5] (actually smaller than this)
    shape_scale = 0.5

    # The octree depth
    depth = 6

    # The octree layers with a depth smaller than full depth
    # are forced to be full
    full_depth = 4

    # number of samples for each octree node
    # Four points are sampled at each node, and the corresponding
    # SDF values are calculated
    nsamples = 4

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
    
    pts = np.load(pts_filepath)
    sdf = np.load(sdf_filepath)
    sdf = torch.from_numpy(sdf)
    points = torch.from_numpy(pts['points'].astype(np.float32))
    normals = torch.from_numpy(pts['normals'].astype(np.float32))
    # points = points / shape_scale

    points = ocnn.octree.Points(points, normals)
    octree = ocnn.octree.Octree(depth=depth, full_depth=full_depth)
    octree.build_octree(points)

    # sample points and grads according to the xyz
    xyzs, grads, sdfs = [], [], []
    for d in range(full_depth, depth + 1):
        xyzb = octree.xyzb(d)
        x,y,z,b = xyzb
        xyz = torch.stack((x,y,z),dim=1).float()

        # sample k points in each octree node
        xyz = xyz.unsqueeze(1) + torch.rand(xyz.shape[0], nsamples, 3)
        xyz = xyz.view(-1, 3)                       # (N, 3)
        # normalize to [0, 2^sdf_depth] which equivalent 
        # to scaling the coordinates to [0, sdf_res]
        xyz = xyz * (sdf_res / 2 ** d)              
        xyz = xyz[(xyz < sdf_res - 1).all(dim=1)]           # remove out-of-bound points
        xyzs.append(xyz)

        # interpolate the sdf values
        xyzi = torch.floor(xyz)                     # the integer part (N, 3)
        corners = xyzi.unsqueeze(1) + grid          # (N, 8, 3)
        coordsf = xyz.unsqueeze(1) - corners        # (N, 8, 3), in [-1.0, 1.0]
        weights = (1 - coordsf.abs()).prod(dim=-1)  # (N, 8, 1)
        corners = corners.long().view(-1, 3)
        x, y, z = corners[:, 0], corners[:, 1], corners[:, 2]
        s = sdf[x, y, z].view(-1, 8)
        sw = torch.sum(s * weights, dim=1)
        sdfs.append(sw)

        # calc the gradient
        gx = s[:, 4] - s[:, 0] + s[:, 5] - s[:, 1] + s[:, 6] - s[:, 2] + s[:, 7] - s[:, 3]    # noqa
        gy = s[:, 2] - s[:, 0] + s[:, 3] - s[:, 1] + s[:, 6] - s[:, 4] + s[:, 7] - s[:, 5]    # noqa
        gz = s[:, 1] - s[:, 0] + s[:, 3] - s[:, 2] + s[:, 5] - s[:, 4] + s[:, 7] - s[:, 6]    # noqa
        grad = torch.stack([gx, gy, gz], dim=-1)
        norm = torch.sqrt(torch.sum(grad ** 2, dim=-1, keepdims=True))
        grad = grad / (norm + 1.0e-8)
        grads.append(grad)
    xyzs = torch.cat(xyzs, dim=0).numpy()

    # The points here are the SDF sampled points, which are then further
    # scaled to [-0.5, 0.5]. I really don't understand why we need to add this 
    # 0.5 shape_scale and keep transforming it back and forth. 
    # What is the significance of this?
    points = (xyzs / (sdf_res // 2) - 1).astype(np.float16) #* shape_scale    
    grads = torch.cat(grads, dim=0).numpy().astype(np.float16)

    # The sdf here is still the same as before, in [-1, 1] (or [-0.8. 0.8])
    sdfs = torch.cat(sdfs, dim=0).numpy().astype(np.float16)   

    # save results
    # In other words, the values of the SDF are in [-1, 1] (or [-0.8, 0.8])
    # but the coordinates of the points are in [0.5, 0.5]
    np.savez(out_filepath_sdf, points=points, grad=grads, sdf=sdfs)

    if debug:
        plot_octree(octree)
    return out_filepath_sdf

if __name__ == "__main__":
    k_indir, k_outdir, k_row, k_sdf_res, k_mesh_scale, k_out_scale, k_n_pts_samples, k_debug = parse_args()
    
    # sanity check for directories
    if not os.path.exists(k_indir):
        log.ERROR(f"The specified directory is invalid: {k_indir}")
        exit(-1)

    k_input_row_dir = os.path.join(k_indir, k_row)
    if not os.path.exists(k_input_row_dir):
        log.ERROR(f"The specified directory is invalid: {k_input_row_dir}")
        exit(-1)

    # create output directories
    k_output_row_dir = os.path.join(k_outdir, k_row)
    if not os.path.exists(k_outdir):
        log.INFO(f"Output directory does not exist. Creating: {k_outdir}.")
        os.mkdir(k_outdir)
    
    if not os.path.exists(k_output_row_dir):
        log.INFO(f"Output directory does not exist. Creating: {k_output_row_dir}.")
        os.mkdir(k_output_row_dir)


    # gather input filepaths
    src_filepaths = glob.glob(os.path.join(k_input_row_dir, '*.obj'))
    loop = tqdm(src_filepaths, total=len(src_filepaths), ncols=100)

    # run dataset creation
    for src_filepath in loop:
        # create output folder the specified file
        src_id = os.path.basename(src_filepath).replace('.obj', '')
        output_shape_dir = os.path.join(k_output_row_dir, src_id)
        if not os.path.exists(output_shape_dir):
            os.mkdir(output_shape_dir)

        # run mesh2sdf
        loop.set_description_str(f"mesh2sdf")
        out_filepath_bbox, out_filepath_obj, out_filepath_sdf = process_mesh2sdf(
            src_filepath, src_id, output_shape_dir, k_mesh_scale, k_sdf_res
        )
        
        # run point sampling
        loop.set_description_str(f"sampling pcl pnts")
        out_filepath_pts, out_filepath_ply = sample_pts_from_mesh(
            out_filepath_obj, src_id, output_shape_dir, k_n_pts_samples
        )

        # run sdf sampling
        loop.set_description_str(f"sampling gt sdf")
        out_filepath_sdf_gt = sample_gt_sdf(
            out_filepath_sdf, out_filepath_pts, src_id, output_shape_dir, k_sdf_res, k_debug
        )
        loop.update(1)