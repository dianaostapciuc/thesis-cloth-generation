import os.path as osp
import numpy as np
import torch
import pickle
from torch.utils.data import Dataset
from scipy.sparse import coo_matrix
from config.config import config

def faces_to_unique_edges(faces: np.ndarray) -> np.ndarray:
    edges = np.vstack([faces[:, [0, 1]],
                       faces[:, [1, 2]],
                       faces[:, [2, 0]]])
    edges = np.sort(edges, axis=1)
    edges = np.unique(edges, axis=0)
    return edges.astype(np.int32)


def laplacian_smooth(verts: np.ndarray,
                     faces: np.ndarray,
                     num_iter: int = 80,
                     lambda_val: float = 0.15) -> np.ndarray:
    if num_iter <= 0:
        return verts.copy()

    V = verts.shape[0]

    ii = np.hstack([faces[:, 0], faces[:, 1], faces[:, 2]])
    jj = np.hstack([faces[:, 1], faces[:, 2], faces[:, 0]])
    idx = np.hstack([ii, jj])          # add both directions (undirected)
    jdx = np.hstack([jj, ii])
    ones = np.ones_like(idx, dtype=np.float32)

    A = coo_matrix((ones, (idx, jdx)), shape=(V, V)).tocsr()
    deg = np.array(A.sum(axis=1)).flatten()
    deg[deg == 0] = 1.0
    inv_deg = 1.0 / deg

    v = verts.astype(np.float32).copy()
    for _ in range(num_iter):
        neighbor_avg = (A @ v) * inv_deg[:, None]
        v += lambda_val * (neighbor_avg - v)
    return v

class MultiGarmentDataset(Dataset):
    def __init__(self,
                 dataset_root: str,
                 garment_type: str,
                 gender: str,
                 avail_txt: str = "avail.txt",
                 smooth: bool = False):
        super().__init__()
        self.samples = []
        self.smooth  = smooth

        meta_path = config.get("paths.garment_info")
        if not osp.isfile(meta_path):
            raise FileNotFoundError(f"garment_class_info.pkl not found at {meta_path}")

        with open(meta_path, "rb") as f:
            garment_meta_all = pickle.load(f, encoding="latin-1")

        if garment_type not in garment_meta_all:
            raise KeyError(f"{garment_type} not present in garment_class_info.pkl")

        g_meta = garment_meta_all[garment_type]

        self.faces = g_meta["f"]
        self.edges = g_meta.get("edges",
                                faces_to_unique_edges(self.faces))

        base      = osp.join(dataset_root, f"{garment_type}_{gender}")
        pose_dir  = osp.join(base, "pose")
        shape_dir = osp.join(base, "shape")
        style_dir = osp.join(base, "style")

        txt_path = osp.join(base, avail_txt)
        if not osp.isfile(txt_path):
            raise FileNotFoundError(f"avail.txt not found at {txt_path}")

        with open(txt_path, "r") as f:
            pivots = f.read().strip().split()

        for pivot in pivots:
            shape_id, style_id = pivot.split("_")
            subdir = osp.join(pose_dir, pivot)
            if not osp.isdir(subdir):
                continue

            unposed_path = osp.join(subdir, f"unposed_{shape_id}.npy")
            beta_path    = osp.join(shape_dir, f"beta_{shape_id}.npy")
            gamma_path   = osp.join(style_dir, f"gamma_{style_id}.npy")
            if not all(map(osp.isfile, [unposed_path, beta_path, gamma_path])):
                continue

            unposed_arr = np.load(unposed_path)
            betas_full  = np.load(beta_path)
            gammas      = np.load(gamma_path)

            betas_trunc = betas_full[:10]

            for f_idx in range(unposed_arr.shape[0]):
                disp = unposed_arr[f_idx]
                if self.smooth:
                    disp = laplacian_smooth(disp, self.faces,
                                            num_iter=80, lambda_val=0.15)

                self.samples.append({
                    "betas": betas_trunc,
                    "gammas": gammas,
                    "lf_disp": disp
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return {
            "betas":   torch.tensor(s["betas"],  dtype=torch.float32),
            "gammas":  torch.tensor(s["gammas"], dtype=torch.float32),
            "lf_disp": torch.tensor(s["lf_disp"], dtype=torch.float32),
        }
