import os
import os.path as osp
import numpy as np
from typing import List, Tuple  # Use List and Tuple from typing
from config.config import config
from smpl_lib.ch_smpl import Smpl
from utils.smpl_utils import load_smpl_model


def save_obj(vertices: np.ndarray, faces: np.ndarray, filename: str) -> None:
    """
    Saves a mesh as an OBJ file.
    vertices: numpy array of shape (N, 3)
    faces: numpy array of shape (F, 3) with 0-based indices
    """
    with open(filename, "w") as f:
        for v in vertices:
            f.write("v {:.4f} {:.4f} {:.4f}\n".format(v[0], v[1], v[2]))
        for face in faces:
            f.write("f {} {} {}\n".format(face[0] + 1, face[1] + 1, face[2] + 1))


def compute_body_obj(gender: str, betas: List[float], out_dir: str) -> Tuple[str, np.ndarray, np.ndarray]:
    """
    Loads the SMPL model for the given gender, uses the first 2 beta values (padded to 10),
    computes the body mesh (vertices and faces), saves it as an OBJ file, and returns the file path
    along with the computed vertices and faces.
    """
    # Create a full 10-element beta vector using only the first 2 values.
    betas_arr = np.zeros(10)
    if len(betas) < 2:
        raise Exception("At least 2 beta values are required!")
    betas_arr[:2] = betas[:2]

    # Load SMPL model.
    smpl_base = config.get("paths.smpl_hres")
    gender_lower = gender.lower()
    smpl_path = osp.join(smpl_base, f"smpl_hres_{gender_lower}.npz")
    if not osp.exists(smpl_path):
        raise Exception(f"SMPL file not found for {gender} at {smpl_path}")
    smpl_model = load_smpl_model(smpl_path)

    # Set parameters (do not force a T-pose, so current pose remains).
    smpl_model.betas[:] = betas_arr
    smpl_model._set_up()

    body_verts = np.array(smpl_model.r)
    body_faces = np.array(smpl_model.f, dtype=np.int32)

    # Ensure output directory exists.
    if not osp.exists(out_dir):
        os.makedirs(out_dir)

    body_obj_path = osp.join(out_dir, f"body_{gender_lower}.obj")
    save_obj(body_verts, body_faces, body_obj_path)

    return body_obj_path, body_verts, body_faces
