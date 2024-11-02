import argparse
import os
import json
import numpy as np

from tqdm import tqdm
import trimesh

k_lower_labels = [31, 32, 33, 34, 35, 36, 37, 41, 42, 43, 44, 45, 46, 47]
k_upper_labels = [11, 12, 13, 14, 15, 16, 17, 21, 22, 23, 24, 25, 26, 27]


def parse_args():
    parser = argparse.ArgumentParser("Teeth3DS Teeth Extraction")
    parser.add_argument("--root_dir", type=str, help='The root directory with the original Teeth3DS data.')
    parser.add_argument("--row", type=str, choices=['lower', 'upper'], help='The teeth row to load.')
    parser.add_argument("--out_dir", type=str, help="The directory to output the extracted data.")
    args = parser.parse_args()
    
    root_dir = args.root_dir
    row = args.row
    out_dir = args.out_dir

    # sanity check
    if not os.path.exists(root_dir):
        print(f"The specified root directory is invalid: {root_dir}")
        exit()
    
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    out_dir = os.path.join(out_dir, row)
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    return root_dir, out_dir, row



def extract_teeth_mesh(mesh: trimesh.Trimesh, row: str, annotation: dict) -> trimesh.Trimesh:
    """
    Extract a mesh keeping only the teeth surfaces.

    :param mesh The original input mesh.
    :param row The teeth row the mesh represents
    :return a combined mesh with only the teeth
    """
    vertex_labels = np.array(annotation['labels'])
    labels = k_lower_labels if row == 'lower' else k_upper_labels
    submeshes = []
    for label in labels:
        verts_to_keep = np.where(vertex_labels == label)[0]
        faces = mesh.faces
        mask = np.isin(faces, verts_to_keep)
        faces_to_keep = np.any(mask, axis=1)
        submesh = mesh.submesh([np.where(faces_to_keep)[0]], append=True)
        submeshes.append(submesh)
    combined = trimesh.util.concatenate(submeshes)
    return combined



if __name__ == "__main__":
    root_dir, out_dir, row = parse_args()

    in_dir = os.path.join(root_dir, row)
    sample_folder_names = os.listdir(in_dir)
    loop = tqdm(sample_folder_names, total=len(sample_folder_names), desc='extracting teeth')
    for sample_folder_name in loop:
        # parse filenames
        sample_dir = os.path.join(in_dir, sample_folder_name)
        mesh_filename = sample_folder_name + f'_{row}.obj'
        annotation_filename = sample_folder_name + f'_{row}.json'
        mesh_filepath = os.path.join(sample_dir, mesh_filename)
        annotation_filepath = os.path.join(sample_dir, annotation_filename)

        # check if output already exists so we don't do any 
        # calculation with no reason
        output_filepath = os.path.join(out_dir, mesh_filename)
        if os.path.exists(output_filepath):
            continue

        # do the actual teeth extraction
        with open(annotation_filepath, 'r') as infd:
            annotation_data = json.load(infd)
        mesh = trimesh.load_mesh(mesh_filepath)
        teeth_mesh = extract_teeth_mesh(mesh, row, annotation_data)

        # output the result
        teeth_mesh.export(output_filepath)