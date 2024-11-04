from typing import Dict, Any
from datasets.registry import register_dataset

import torch
from torch.utils.data import Dataset
import numpy as np

import os
from tqdm import tqdm
import utils.log as log
from ocnn.octree.points import Points
from ocnn.dataset import CollateBatch

@register_dataset
class Teeth3DSDataset(Dataset):
    k_pcl_suffix = '_pcl.npz'
    k_sdf_suffix = '_sdf.npz'

    def __init__(
        self, 
        root_dir: str, 
        row: str, 
        point_scale: float=0.5, 
        point_nsamples: int=50_000,
        device: torch.device=torch.device('cpu')
    ):
        """
        Instantiate a Teeth3DSDataset object.

        :param root_dir The root directory of the data.
        :param row The row of the dataset.
        """
        # sanity checks
        if row not in ['upper', 'lower']:
            log.ERROR(f"Invalid row. Expected one of ['lower', 'upper'], got: {row}.")
            return
        
        if not os.path.exists(root_dir):
            log.ERROR(f"The specified Teeth3DS root directory does not exist.")
            return
        
        k_row_dir = os.path.join(root_dir, row)
        if not os.path.exists(k_row_dir):
            log.ERROR(f"The specified Teeth3DS directory does not contain a row <{row}> folder.")
            return
        
        self.point_scale = point_scale
        self.point_samples = point_nsamples
        self.device = device
        # load data filepaths
        self.samples = []
        samples_folders = os.listdir(k_row_dir)
        loop = tqdm(samples_folders, total=len(samples_folders), desc='loading data')
        # sample_folder is also shape id
        for sample_folder in loop:
            sample_dir = os.path.join(k_row_dir, sample_folder)
            pcl_filepath = os.path.join(sample_dir, sample_folder + self.k_pcl_suffix)
            sdf_filepath = os.path.join(sample_dir, sample_folder + self.k_sdf_suffix)
            if os.path.exists(pcl_filepath) and os.path.exists(sdf_filepath):
                self.samples += [{'pcl_filepath': pcl_filepath, 'sdf_filepath': sdf_filepath}]
        
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index) -> Dict[str, Any]:
        sample = self.samples[index]
        pcl_filepath = sample['pcl_filepath']
        sdf_filepath = sample['sdf_filepath']
        pcl_raw = np.load(pcl_filepath)
        sdf_raw = np.load(sdf_filepath)
        points, normals = pcl_raw['points'], pcl_raw['normals']
        # points = points / self.point_scale
        points = torch.from_numpy(points).float().to(self.device)
        normals = torch.from_numpy(normals).float().to(self.device)

        points_gt = Points(points=points, normals=normals).to(self.device)
        points_gt.clip(min=-1, max=1)

        sdf = sdf_raw['sdf']
        grad = sdf_raw['grad']
        pos = sdf_raw['points'] #/ self.point_scale
        
        rand_idx = np.random.choice(pos.shape[0], size=self.point_samples)
        pos = torch.from_numpy(pos[rand_idx]).float().to(self.device)
        sdf = torch.from_numpy(sdf[rand_idx]).float().to(self.device)
        grad = torch.from_numpy(grad[rand_idx]).float().to(self.device)

        output = {
            'points': points_gt,
            'pos': pos,
            'sdf': sdf,
            'grad': grad
        }
        return output