from typing import Tuple
import numpy as np
import skimage

def mcubes(sdf: np.ndarray, level: float=0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform marching cubes to get a triangle mesh
    for the specified SDF.
    """
    v_pos, t_pos_idx = np.zeros([0, 3]), np.zeros([0, 3])
    try:
        v_pos, t_pos_idx, _, _ = skimage.measure.marching_cubes(sdf, level)
    except:
        pass
    return v_pos, t_pos_idx
