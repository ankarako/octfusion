import torch
import torch.nn
from typing import Dict, TypedDict
import numpy as np

import ocnn
from ocnn.octree import Octree, Points
import open3d as o3d

def get_position_from_xyz_coords(octree: ocnn.octree.Octree, depth):
    x,y,z, _ = octree.xyzb(depth, nempty=True)
    size = 1/ (2**depth) * 2
    centers = (torch.stack([x, y, z], dim=1).cpu().numpy()+ 0.5) * size - 1.0  # [-1, 1]
    return centers, size


def get_lines_from_node(centers, size):
    """
        centers: (n_nodes, 3)
        size: scalar
    """
    corners_offsets = np.array([
        [[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1], [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]]
    ]) # (1, 8, 3) corners of the cube
    edges_offsets = np.array([
        [0, 1], [1, 2], [2, 3], [3, 0],
        [4, 5], [5, 6], [6, 7], [7, 4],
        [0, 4], [1, 5], [2, 6], [3, 7]
    ])  # (12, 2) edges of the cube, each row is a pair of indices of the corners that form an edge


    half_size = size / 2
    corners = centers[:, None, :] + half_size * corners_offsets

    edges = np.stack((corners[:, edges_offsets[:, 0], :],
                     corners[:, edges_offsets[:, 1], :]), axis=-2)  # (n_nodes, 12, 2, 3)

    return edges.reshape(-1, 2, 3)  # (n_lines, 2, 3)


def plot_octree_from_points(points):
    xyz, xyz_mean, scale = transform2origin(points)  # scale the points to [-1, 1]
    # Build octree
    points = Points(points=xyz, features=None, labels=None, batch_size=1)  # batch_size=1 means that we have one set of points at a time
    octree = ocnn.octree.Octree(depth=8, full_depth=3, batch_size=1, device=xyz.device) # batch_size=1 means that we have one octree at a time
    octree.build_octree(points)
    octree.construct_all_neigh()


    # plot with open3d
    all_lines = np.concatenate([get_lines_from_node(*get_position_from_xyz_coords(octree, d)) for d in range(0,9)])

    line_set = o3d.geometry.LineSet(points=o3d.utility.Vector3dVector(all_lines.reshape(-1, 3)),
                                    lines=o3d.utility.Vector2iVector(np.arange(all_lines.reshape(-1, 3).shape[0]).reshape(-1, 2)))

    o3d_pcd = o3d.geometry.PointCloud()
    o3d_pcd.points = o3d.utility.Vector3dVector(xyz.detach().cpu().numpy())
    o3d.visualization.draw_geometries([o3d_pcd, line_set])



def plot_octree(octree: ocnn.octree.Octree):
    all_lines = np.concatenate([
        get_lines_from_node(*get_position_from_xyz_coords(octree, d)) for d in range(octree.full_depth, octree.depth+1)
    ])
    points=o3d.utility.Vector3dVector(all_lines.reshape(-1, 3))
    ll = np.arange(all_lines.reshape(-1, 3).shape[0])
    ll = ll.reshape(-1, 2).astype(np.int32)
    lines=o3d.utility.Vector2iVector(ll)
    line_set = o3d.geometry.LineSet(points=points, lines=lines)

    o3d_pcd = o3d.geometry.PointCloud()
    feat = octree.get_input_feature(feature='P', nempty=False).detach().cpu().numpy().astype(np.float64)
    o3d_pcd.points = o3d.utility.Vector3dVector(feat)
    o3d.visualization.draw_geometries([o3d_pcd, line_set])



def transform2origin(xyz):
    # scale the points to [-1, 1]
    # works for both single and batched inputs
    min_pos = torch.min(xyz, -2, keepdim=True)[0]  # -2 so that it works for batched inputs
    max_pos = torch.max(xyz, -2, keepdim=True)[0]

    center = 0.5*(min_pos + max_pos)
    scale = torch.max(max_pos - min_pos, dim=-1, keepdim=True)[0] / 2.0
    new_position_tensor = (xyz - center) / scale

    return new_position_tensor, center, scale

def transformback(xyz, center, scale):
    return xyz * scale + center


def process_batch(xyz: torch.Tensor, features: torch.Tensor, depth: int, full_depth: int, feat: str, nempty: bool, normalize=False):
    """
    Process both single and batched inputs of xyz ([B], N , 3) and features ([B], N, F)
    into octree and query_pts.

    The purpose of octree: is to be able to encode the input points into a structure
    which keeps track of the spatial relationships between the points and allows for
    ussage of convolutions and other operations(attention/transformer) on the points.

    After we process  the octree node features through the network, we can get back
    a feature for each point by interpolating the octree node features at the query points.


    depth: int -- depth of the octree
    full_depth: int -- full depth of the octree (depth where the octree is full, i.e. all nodes have 8 children)


    query_pts: [B*N, 4] (normalized points [-1, 1] with batch index)
    """
    # scale the points to [-1, 1]
    if normalize:
        normalized_xyz, mean_xyz, scale_xyz = transform2origin(xyz)
    else:
        normalized_xyz = xyz

    if normalized_xyz.dim() == 2:
        # xyz: (N, 3) -- single
        points = Points(points=normalized_xyz, features=features, batch_size=1)
        query_pts = torch.cat([points.points, torch.zeros(normalized_xyz.shape[0], 1, device="cuda")], dim=1)
        B = 1
    else:
        # xyz: (B, N, 3) -- batched
        B, N, F = features.shape
        batch_ids = torch.arange(B, device=normalized_xyz.device).reshape(B, 1).repeat(1, N)  # (B, N)
        points = Points(points=normalized_xyz.reshape(B*N, 3), features=features.reshape(B*N, F), batch_id=batch_ids.reshape(B*N), batch_size=B)
        query_pts = torch.cat([points.points, points.batch_id.unsqueeze(-1)], dim=1)

    octree = ocnn.octree.Octree(depth=depth, full_depth=full_depth, batch_size=B, device=normalized_xyz.device)
    octree.build_octree(points)
    octree.construct_all_neigh()

    data = octree.get_input_feature(feat, nempty)  # get the feature on the octree nodes

    return octree, data, query_pts