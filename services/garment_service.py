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

from config.config import config
from train.LowFrequencyModel import LowFreqModel
from smpl_lib.ch_smpl import Smpl
from app.models import GarmentRequest
from utils.mesh_utils import save_obj
from collections import Counter

_MODEL_CACHE, _SMPL_CACHE = {}, {}

def heal_gaps(body_v, g_v, max_gap=0.002):
    idx  = cKDTree(body_v).query(g_v)[1]
    dist = np.linalg.norm(g_v - body_v[idx], axis=1)
    g_v[dist > max_gap] = body_v[idx[dist > max_gap]]
    return g_v

def shrinkwrap(body_v, g_v, body_f, offset=0.003):
    body_m = trimesh.Trimesh(body_v, body_f, process=False)
    nrm    = body_m.vertex_normals
    idx    = cKDTree(body_v).query(g_v)[1]
    return body_v[idx] + nrm[idx] * offset, nrm[idx]

def lap_smooth(v, f, n_iter=6, lam=0.3):
    import scipy.sparse as sp
    V = v.shape[0]
    I = np.hstack([f[:,0], f[:,1], f[:,2]])
    J = np.hstack([f[:,1], f[:,2], f[:,0]])
    A = sp.coo_matrix((np.ones_like(I),(I,J)),shape=(V,V)).tocsr()
    deg = np.array(A.sum(1)).flatten(); deg[deg==0]=1
    L   = sp.diags(1/deg)@A - sp.eye(V)
    vs  = v.copy()
    for _ in range(n_iter):
        vs += lam*(L@vs)
    return vs

def _get_model_and_stats(gar, gen):
    key = (gar, gen)
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]
    info  = pickle.load(open(config.get("paths.garment_info"),"rb"),encoding="latin-1")[gar]
    num_v = info["vert_indices"].shape[0]
    out   = config.get("paths.output_dir")
    ckpt  = osp.join(out, f"model_best_{gar}_{gen}.pth")
    stats = osp.join(out, f"{gar}_{gen}_stats.json")

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
    st_np = {k: np.asarray(v, np.float32) for k,v in flat.items()}
    _MODEL_CACHE[key] = (mdl, st_np)
    return mdl, st_np

def _get_smpl(gen):
    if gen in _SMPL_CACHE:
        return _SMPL_CACHE[gen]
    path = osp.join(config.get("paths.smpl_hres"), f"smpl_hres_{gen}.npz")
    data = np.load(path, allow_pickle=True)
    smpl = Smpl({k:data[k] for k in data.files})
    _SMPL_CACHE[gen] = smpl
    return smpl

def export_body_skin_data(smpl:Smpl, out_dir:str, gen:str)->str:
    smpl._set_up()
    J = len(smpl.J)
    joint_array = np.stack([np.array(p, dtype=float) for p in smpl.J], axis=0)
    jointPos_flat = joint_array.ravel().tolist()
    A = np.asarray(smpl.A, dtype=float)
    if A.ndim==3 and A.shape[1:]==(4,4):
        mats = A
    else:
        mats = A.transpose(2,0,1)
    bindposes_flat = []
    for M in mats:
        bindposes_flat.extend(np.linalg.inv(M).ravel().tolist())
    W = np.asarray(smpl.weights, dtype=float)
    boneIndices_flat  = []
    boneWeights_flat  = []
    for w in W:
        top4 = np.argsort(w)[-4:][::-1]
        vals = w[top4]
        vals = (vals/ (vals.sum()+1e-8)).tolist()
        boneIndices_flat.extend(top4.tolist())
        boneWeights_flat.extend(vals)
    parents = [-1,0,0,0,1,2,3,4,5,6,7,8,
                9,9,3,3,14,15,16,17,18,19,20,21]

    vertices_flat = smpl.r.astype(float).reshape(-1).tolist()
    triangles = smpl.f.astype(int).reshape(-1).tolist()

    skin = {
        "jointCount":       J,
        "vertexCount":      W.shape[0],
        "jointPos_flat":    jointPos_flat,
        "bindposes_flat":   bindposes_flat,
        "parents":          parents,
        "boneIndices_flat": boneIndices_flat,
        "boneWeights_flat": boneWeights_flat,
        "vertices_flat":    vertices_flat,
        "triangles":        triangles
    }

    os.makedirs(out_dir, exist_ok=True)
    fp = osp.join(out_dir, f"body_{gen}_skin.json")
    with open(fp,"w") as f:
        json.dump(skin,f,separators=(",",":"),ensure_ascii=False)
    return fp

def export_garment_skin_data(smpl:Smpl, out_dir:str, gar:str, gen:str)->str:
    smpl._set_up()
    A = np.asarray(smpl.A, dtype=float)
    if A.ndim==3 and A.shape[0]==4 and A.shape[1]==4:
        J = A.shape[2]; A_all = A.transpose(2,0,1)
    elif A.ndim==3 and A.shape[1:]==(4,4):
        J = A.shape[0]; A_all = A
    else:
        raise ValueError(f"Unexpected A shape {A.shape!r}")
    joint_array = np.stack([np.array(p, dtype=float) for p in smpl.J], axis=0)
    jointPos_flat = joint_array.ravel().tolist()
    bindposes_flat = []
    for i in range(J):
        inv = np.linalg.inv(A_all[i])
        bindposes_flat.extend(inv.ravel().tolist())

    body_v, body_f = np.asarray(smpl.r), smpl.f
    info = pickle.load(open(config.get("paths.garment_info"),"rb"),encoding="latin-1")[gar]
    idx_map, gf = info["vert_indices"], info["f"]

    raw = body_v[idx_map]
    healed = heal_gaps(body_v, raw.copy())
    wrapped, nm = shrinkwrap(body_v, healed, body_f, offset=0.003)
    wrapped += nm * 0.002
    smooth = lap_smooth(wrapped, gf)
    inner = 0.6*smooth + 0.4*raw
    outer_raw = inner + nm * 0.002
    outer = lap_smooth(outer_raw, gf, n_iter=3, lam=0.25)

    verts = outer
    faces = gf
    gm = trimesh.Trimesh(verts, faces, process=False)
    trimesh.repair.fill_holes(gm)
    final_vs = gm.vertices
    final_fs = gm.faces

    tree = cKDTree(verts)
    W_body = np.asarray(smpl.weights, dtype=float)
    W_gar = W_body[idx_map]
    boneIndices_flat = []
    boneWeights_flat = []
    for v in final_vs:
        nearest = tree.query(v)[1]
        w = W_gar[nearest % len(idx_map)]
        top4 = np.argsort(w)[-4:][::-1]
        vals = (w[top4]/(w[top4].sum()+1e-8)).tolist()
        boneIndices_flat.extend(top4.tolist())
        boneWeights_flat.extend(vals)

    counts = Counter(boneIndices_flat)
    missing = [j for j in range(J) if counts[j] == 0]
    if missing:
        print(f"WARNING: injecting missing joints {missing}")
    V_final = final_vs.shape[0]
    for j in missing:
        vidx = j % V_final
        off = vidx * 4
        w_slot = boneWeights_flat[off:off+4]
        i_min = int(np.argmin(w_slot))
        w_slot[i_min] = 0.0
        total = sum(w_slot)
        if total > 0:
            w_slot[:] = [w_i/total for w_i in w_slot]
        boneWeights_flat[off:off+4] = w_slot
        boneIndices_flat[off+i_min] = j

    parents = [-1,0,0,0,1,2,3,4,5,6,7,8,
               9,9,3,3,14,15,16,17,18,19,20,21]

    skin = {
        "jointCount":       J,
        "vertexCount":      final_vs.shape[0],
        "jointPos_flat":    jointPos_flat,
        "bindposes_flat":   bindposes_flat,
        "parents":          parents,
        "boneIndices_flat": boneIndices_flat,
        "boneWeights_flat": boneWeights_flat,
        "vertices_flat":    final_vs.astype(float).reshape(-1).tolist(),
        "triangles":        final_fs.astype(int).reshape(-1).tolist()
    }

    os.makedirs(out_dir, exist_ok=True)
    fp = osp.join(out_dir, f"{gar}_{gen}_skin.json")
    with open(fp,"w") as f:
        json.dump(skin,f,separators=(",",":"),ensure_ascii=False)
    return fp

def generate_garment(req: GarmentRequest) -> str:
    t0 = time.time()
    gar, gen = req.garment, req.gender.lower()

    betas  = np.zeros(10, np.float32)
    betas[:len(req.betas)] = req.betas

    gammas = np.zeros(4, np.float32)
    if req.gammas:
        gammas[:len(req.gammas)] = req.gammas

    # Model inference
    mdl, st = _get_model_and_stats(gar, gen)
    dev      = next(mdl.parameters()).device
    eps      = 1e-8
    x        = np.concatenate([
                   (betas  - st["betas_mean"])  / np.maximum(st["betas_std"],  eps),
                   (gammas - st["gammas_mean"]) / np.maximum(st["gammas_std"], eps)
               ]).astype(np.float32)

    with torch.no_grad():
        disp = mdl(torch.from_numpy(x)[None].to(dev)).squeeze(0).cpu().numpy()

    smpl      = _get_smpl(gen)
    smpl.betas[:] = betas
    smpl.pose[:]  = 0
    smpl.trans[:] = 0
    smpl._set_up()
    body_v, body_f = np.asarray(smpl.r), smpl.f

    info      = pickle.load(open(config.get("paths.garment_info"), "rb"),
                              encoding="latin-1")[gar]
    idx_map, g_faces = info["vert_indices"], info["f"]
    raw_g     = body_v[idx_map] + disp

    healed    = heal_gaps(body_v, raw_g.copy())
    wrapped, normals = shrinkwrap(body_v, healed, body_f, offset=0.003)
    wrapped  += normals * 0.002

    smoothed  = lap_smooth(wrapped, g_faces)
    inner     = 0.6 * smoothed + 0.4 * raw_g

    outer_raw = inner + normals * 0.002
    outer     = lap_smooth(outer_raw, g_faces, n_iter=3, lam=0.25)

    verts       = np.vstack([inner, outer])
    faces_inner = g_faces
    faces_outer = g_faces + inner.shape[0]
    faces       = np.vstack([faces_inner, faces_outer])

    gm = trimesh.Trimesh(verts, faces, process=False)
    trimesh.repair.fill_holes(gm)
    final_verts = gm.vertices
    final_faces = gm.faces

    out      = config.get("paths.output_dir"); os.makedirs(out, exist_ok=True)
    body_obj = osp.join(out, f"body_{gen}.obj")
    gar_obj  = osp.join(out, f"garment_{gar}_{gen}.obj")

    save_obj(body_v,      body_f,       body_obj)
    save_obj(final_verts, final_faces,  gar_obj)

    zip_p = osp.join(out, f"{gar}_{gen}_meshes.zip")
    with zipfile.ZipFile(zip_p, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(body_obj, osp.basename(body_obj))
        z.write(gar_obj,  osp.basename(gar_obj))

    print(f"[OK] {gar}/{gen} in {time.time() - t0:.2f}s")
    return zip_p

def export_skinning_json(req:GarmentRequest) -> str:
    gar, gen = req.garment, req.gender.lower()
    smpl = _get_smpl(gen)
    smpl.betas[:] = 0; smpl.pose[:] = 0; smpl.trans[:] = 0
    smpl._set_up()
    out = config.get("paths.output_dir")
    os.makedirs(out, exist_ok=True)
    body_skin = export_body_skin_data(smpl, out, gen)
    info = pickle.load(open(config.get("paths.garment_info"),"rb"),encoding="latin-1")[gar]
    idx_map = info["vert_indices"]
    gar_skin = export_garment_skin_data(smpl, idx_map, out, gar, gen)
    zip_p = osp.join(out, f"{gar}_{gen}_skins.zip")
    with zipfile.ZipFile(zip_p, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(body_skin, osp.basename(body_skin))
        z.write(gar_skin,  osp.basename(gar_skin))
    return zip_p