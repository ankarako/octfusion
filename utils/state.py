from typing import List

import torch
import torch.multiprocessing as mp
import numpy as np
import random

__all__ = ["seed", "set_state", "get_device"]

def seed(seed: int) -> None:
    """
    Utility function for seeding all the libraries with the same seed.

    :param seed The seed value to set
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def set_state(cudnn_enabled: bool=True, cudnn_benchmark: bool=True, allow_tf32: bool=True, cudnn_allow_tf32: bool=True, spawn: bool=True) -> None:
    """
    Utility function for setting torch backend state for improving training speed
    and numerical stability
    """
    torch.backends.cudnn.enabled = cudnn_enabled
    torch.backends.cudnn.benchmark = cudnn_benchmark
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    torch.backends.cudnn.allow_tf32 = cudnn_allow_tf32
    if spawn:
        mp.set_start_method('spawn', force=True)


def get_device(device_list: List[int]) -> torch.device:
    if torch.cuda.is_available():
        if len(device_list) > 1:
            raise NotImplementedError()
        return torch.device("cuda:0")
    else:
        return torch.device("cpu")
    