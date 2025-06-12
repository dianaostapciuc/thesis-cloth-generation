import os
import os.path as osp
import numpy as np
from typing import List, Tuple  # Use List and Tuple from typing
from config.config import config
from smpl_lib.ch_smpl import Smpl
from utils.smpl_utils import load_smpl_model


def save_obj(vertices: np.ndarray, faces: np.ndarray, filename: str) -> None:
    with open(filename, "w") as f:
        for v in vertices:
            f.write("v {:.4f} {:.4f} {:.4f}\n".format(v[0], v[1], v[2]))
        for face in faces:
            f.write("f {} {} {}\n".format(face[0] + 1, face[1] + 1, face[2] + 1))


def compute_body_obj(gender: str, betas: List[float], out_dir: str) -> Tuple[str, np.ndarray, np.ndarray]:
    betas_arr = np.zeros(10)
    if len(betas) < 2:
        raise Exception("At least 2 beta values are required!")
    betas_arr[:2] = betas[:2]

    smpl_base = config.get("paths.smpl_hres")
    gender_lower = gender.lower()
    smpl_path = osp.join(smpl_base, f"smpl_hres_{gender_lower}.npz")
    if not osp.exists(smpl_path):
        raise Exception(f"SMPL file not found for {gender} at {smpl_path}")
    smpl_model = load_smpl_model(smpl_path)

    smpl_model.betas[:] = betas_arr
    smpl_model._set_up()

    body_verts = np.array(smpl_model.r)
    body_faces = np.array(smpl_model.f, dtype=np.int32)

    if not osp.exists(out_dir):
        os.makedirs(out_dir)

    body_obj_path = osp.join(out_dir, f"body_{gender_lower}.obj")
    save_obj(body_verts, body_faces, body_obj_path)

    return body_obj_path, body_verts, body_faces
