import os
import os.path as osp
import numpy as np
from smpl_lib.ch_smpl import Smpl

def load_smpl_model(smpl_path: str) -> Smpl:
    if not osp.exists(smpl_path):
        raise FileNotFoundError(f"SMPL file not found at {smpl_path}")
    smpl_data = np.load(smpl_path, allow_pickle=True)
    smpl_dict = {key: smpl_data[key] for key in smpl_data.files}
    smpl_model = Smpl(smpl_dict)
    return smpl_model
