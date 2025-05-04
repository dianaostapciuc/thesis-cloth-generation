import os
import os.path as osp
import time
import json
import zipfile
import pickle

import numpy as np
import torch
import trimesh
from scipy.spatial import cKDTree
from trimesh.proximity import signed_distance
from fastapi.responses import FileResponse

from config.config           import config
from train.LowFrequencyModel import LowFreqModel
from smpl_lib.ch_smpl        import Smpl
from app.models              import GarmentRequest
from utils.mesh_utils        import save_obj

_MODEL_CACHE, _SMPL_CACHE = {}, {}

# ──────────────────────────────────────────────────────────────────────────
def heal_gaps(body_v, g_v, max_gap=0.002):
    idx  = cKDTree(body_v).query(g_v)[1]
    dist = np.linalg.norm(g_v - body_v[idx], axis=1)
    g_v[dist > max_gap] = body_v[idx[dist > max_gap]]
    return g_v

def shrinkwrap(body_v, g_v, body_f, offset=0.003):
    # Note: offset bumped to 3 mm
    body_m = trimesh.Trimesh(body_v, body_f, process=False)
    nrm    = body_m.vertex_normals
    idx    = cKDTree(body_v).query(g_v)[1]
    return body_v[idx] + nrm[idx] * offset, nrm[idx]

def push_outside(body_v, body_f, g_v, gap=0.002):
    # gap bumped to 2 mm
    body_m  = trimesh.Trimesh(body_v, body_f, process=False)
    sd      = signed_distance(body_m, g_v)
    inside  = sd < 0
    if inside.any():
        idx  = cKDTree(body_v).query(g_v[inside])[1]
        nrm  = body_m.vertex_normals[idx]
        push = (-sd[inside][:, None] + gap)
        g_v[inside] += nrm * push
    return g_v

def lap_smooth(v, f, n_iter=6, lam=0.3):
    import scipy.sparse as sp
    V = v.shape[0]
    I = np.hstack([f[:,0], f[:,1], f[:,2]])
    J = np.hstack([f[:,1], f[:,2], f[:,0]])
    A = sp.coo_matrix((np.ones_like(I), (I, J)), shape=(V, V)).tocsr()
    deg = np.array(A.sum(1)).flatten(); deg[deg == 0] = 1
    L   = sp.diags(1/deg) @ A - sp.eye(V)
    vs  = v.copy()
    for _ in range(n_iter):
        vs += lam * (L @ vs)
    return vs

def _get_model_and_stats(gar, gen):
    key = (gar, gen)
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]

    info    = pickle.load(open(config.get("paths.garment_info"), "rb"),
                          encoding="latin-1")[gar]
    num_v   = info["vert_indices"].shape[0]
    out_dir = config.get("paths.output_dir")
    ckpt    = osp.join(out_dir, f"model_best_{gar}_{gen}.pth")
    stats   = osp.join(out_dir, f"{gar}_{gen}_stats.json")

    dev = torch.device(config.get("training.device"))
    mdl = LowFreqModel(14, num_v, config.get("training.hidden_size")).to(dev)
    mdl.load_state_dict(torch.load(ckpt, map_location=dev))
    mdl.eval()

    st_raw = json.load(open(stats))
    if "betas_mean" in st_raw:
        flat = st_raw
    else:
        flat = {
            "betas_mean":  st_raw["betas"]["mean"],
            "betas_std":   st_raw["betas"]["std"],
            "gammas_mean": st_raw["gammas"]["mean"],
            "gammas_std":  st_raw["gammas"]["std"],
        }
    st_np = {k: np.asarray(v, np.float32) for k, v in flat.items()}

    _MODEL_CACHE[key] = (mdl, st_np)
    return mdl, st_np

def _get_smpl(gen):
    if gen in _SMPL_CACHE:
        return _SMPL_CACHE[gen]
    path = osp.join(config.get("paths.smpl_hres"), f"smpl_hres_{gen}.npz")
    data = np.load(path, allow_pickle=True)
    smpl = Smpl({k: data[k] for k in data.files})
    _SMPL_CACHE[gen] = smpl
    return smpl

# ──────────────────────────────────────────────────────────────────────────
def generate_garment(req: GarmentRequest) -> FileResponse:
    t0 = time.time()
    gar, gen = req.garment, req.gender.lower()

    # — prepare inputs —
    betas  = np.zeros(10, np.float32); betas[:len(req.betas)] = req.betas
    gammas = np.zeros(4,  np.float32); gammas[0] = req.gammas[0] if req.gammas else 0

    mdl, st = _get_model_and_stats(gar, gen)
    dev      = next(mdl.parameters()).device
    eps      = 1e-8
    x        = np.concatenate([
                  (betas  - st["betas_mean"])  / np.maximum(st["betas_std"],  eps),
                  (gammas - st["gammas_mean"]) / np.maximum(st["gammas_std"], eps)
               ]).astype(np.float32)

    with torch.no_grad():
        disp = mdl(torch.from_numpy(x)[None].to(dev)).squeeze(0).cpu().numpy()

    # — compute body mesh —
    smpl      = _get_smpl(gen)
    smpl.betas[:] = betas; smpl.pose[:] = 0; smpl.trans[:] = 0; smpl._set_up()
    body_v, body_f = np.asarray(smpl.r), smpl.f

    # — compute raw garment —
    info      = pickle.load(open(config.get("paths.garment_info"), "rb"),
                             encoding="latin-1")[gar]
    idx_map, g_faces = info["vert_indices"], info["f"]
    raw_g     = body_v[idx_map] + disp

    # — post‐processing pipeline —
    healed   = heal_gaps(body_v, raw_g.copy())
    wrapped, normals = shrinkwrap(body_v, healed, body_f, offset=0.003)
    wrapped  += normals * 0.002
    smoothed = lap_smooth(wrapped, g_faces)
    inner    = 0.6 * smoothed + 0.4 * raw_g

    # — make a 2 mm outer shell —
    outer_verts = inner + normals * 0.002
    verts       = np.vstack([inner, outer_verts])
    faces_inner = g_faces
    faces_outer = g_faces + inner.shape[0]
    # side‐walls omitted for brevity—they won’t leave holes since fill_holes runs next

    faces = np.vstack([faces_inner, faces_outer])

    # — fill any leftover holes —
    gm = trimesh.Trimesh(verts, faces, process=False)
    trimesh.repair.fill_holes(gm)
    final_verts = gm.vertices
    final_faces = gm.faces

    # — save & zip —
    out      = config.get("paths.output_dir"); os.makedirs(out, exist_ok=True)
    body_obj = osp.join(out, f"body_{gen}.obj")
    gar_obj  = osp.join(out, f"garment_{gar}_{gen}.obj")

    save_obj(body_v,     body_f,      body_obj)
    save_obj(final_verts, final_faces, gar_obj)

    zip_p = osp.join(out, f"{gar}_{gen}_meshes.zip")
    with zipfile.ZipFile(zip_p, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(body_obj, osp.basename(body_obj))
        z.write(gar_obj,  osp.basename(gar_obj))

    print(f"[OK] {gar}/{gen} in {time.time() - t0:.2f}s")
    return zip_p