import os, os.path as osp, random, pickle
import numpy as np
import torch

from config.config           import config
from dataset.DatasetProcessing import MultiGarmentDataset
from train.LowFrequencyModel   import LowFreqModel
from train.Trainer             import Trainer
from smpl_lib.ch_smpl          import Smpl


# --------------------------------------------------------------------------- #
def save_obj(vertices, faces, filename):
    """Write a mesh to an OBJ file (1‑based face indices)."""
    with open(filename, "w") as f:
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for tri in faces:
            f.write(f"f {tri[0]+1} {tri[1]+1} {tri[2]+1}\n")


# --------------------------------------------------------------------------- #
def main():
    BASE       = config.get("paths.data_root")
    SMPL_BASE  = config.get("paths.smpl_hres")          # high‑res SMPL npz
    INFO       = config.get("paths.garment_info")       # central pickle
    OUT        = config.get("paths.output_dir")
    os.makedirs(OUT, exist_ok=True)

    # ------------ load central garment metadata (faces & vert_indices) ------
    with open(INFO, "rb") as f:
        class_info = pickle.load(f, encoding="latin-1")

    device = torch.device(config.get("training.device"))

    for gender, garments in config.get("garments").items():
        # ------------------------------------------------------------------
        # 1) load SMPL per gender
        # ------------------------------------------------------------------
        smpl_npz = osp.join(SMPL_BASE, f"smpl_hres_{gender.lower()}.npz")
        if not osp.isfile(smpl_npz):
            print(f"[WARN] SMPL not found: {smpl_npz}")
            continue

        smpl_data  = np.load(smpl_npz, allow_pickle=True)
        smpl_model = Smpl({k: smpl_data[k] for k in smpl_data.files})
        body_faces = smpl_model.f
        print(f"[INFO] SMPL {gender}: V={smpl_model.v_template.r.shape[0]}")

        for garment in garments:
            # ----------------------------------------------------------------
            # 2) dataset for this garment+gender
            # ----------------------------------------------------------------
            ds = MultiGarmentDataset(BASE, garment, gender, smooth=False)
            if len(ds) == 0:
                print(f"[WARN] no samples for {garment}_{gender}")
                continue

            # meta info (faces, vert_indices)
            if garment not in class_info:
                print(f"[WARN] missing meta for {garment}; skipping.")
                continue
            g_meta = class_info[garment]        # dict with 'f' and 'vert_indices'

            # ----------------------------------------------------------------
            # 3) build model + trainer
            # ----------------------------------------------------------------
            sample       = ds[0]
            in_dim       = sample["betas"].numel() + sample["gammas"].numel()
            num_verts    = sample["lf_disp"].shape[0]

            model = LowFreqModel(input_dim=in_dim,
                                 num_verts=num_verts,
                                 hidden_size=2048).to(device)

            ckpt_path = osp.join(OUT, f"model_best_{garment}_{gender}.pth")
            trainer   = Trainer(model, ds, config, model_save_path=ckpt_path)
            trainer.train()                         # uses epochs & LR from config

            # ----------------------------------------------------------------
            # 4) quick inference / OBJ export
            # ----------------------------------------------------------------
            model.eval()
            n_export = min(2, len(ds))
            for i, idx in enumerate(random.sample(range(len(ds)), n_export)):
                samp = ds[idx]

                betas  = samp["betas"].unsqueeze(0).to(device)
                gammas = samp["gammas"].unsqueeze(0).to(device)
                x      = torch.cat([betas, gammas], dim=1)

                pred_disp = model(x).squeeze(0).detach().cpu().numpy()  # (V_g,3)

                # ---- update SMPL pose/shape --------------------------------
                smpl_model.betas[:] = betas.cpu().numpy()[0]
                smpl_model.pose[:]  = 0
                smpl_model.trans[:] = 0
                smpl_model._set_up()
                body_verts = np.asarray(smpl_model.r)                   # (V_b,3)

                # ---- correct vertex mapping --------------------------------
                g_faces = g_meta["f"].astype(np.int32)
                g_idx   = g_meta["vert_indices"].astype(np.int32)
                g_verts = body_verts[g_idx] + pred_disp

                # ---- save OBJs ---------------------------------------------
                save_obj(g_verts, g_faces,
                         osp.join(OUT, f"garment_{garment}_{gender}_{i}.obj"))
                save_obj(body_verts, body_faces,
                         osp.join(OUT, f"body_{garment}_{gender}_{i}.obj"))
                print(f"[OK] exported sample {i} for {garment}_{gender}")


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    main()
