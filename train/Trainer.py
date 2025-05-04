import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np


class Trainer:
    """
    Trains LowFreqModel with geometric regularisation:

        loss =  L1
              + 0.1 * MSE
              + 0.5 * edge‑length loss
              + 0.01 * Laplacian smoothness
    """

    # ------------------------------------------------------------------ #
    def __init__(self, model, dataset, config, model_save_path=None):
        self.device = torch.device(config.get("training.device"))
        self.model  = model.to(self.device)
        self.loader = DataLoader(dataset,
                                 batch_size=config.get("training.batch_size"),
                                 shuffle=True,
                                 num_workers=4,
                                 drop_last=True)

        self.epochs          = config.get("training.epochs")
        self.model_save_path = model_save_path

        # ------------ stats for input normalisation ------------
        betas_all  = torch.stack([s["betas"]  for s in dataset])  # (N,10)
        gammas_all = torch.stack([s["gammas"] for s in dataset])  # (N,S)

        self.betas_mean  = betas_all.mean(0, keepdim=True)
        self.betas_std   = betas_all.std(0,  keepdim=True) + 1e-8
        self.gammas_mean = gammas_all.mean(0, keepdim=True)
        self.gammas_std  = gammas_all.std(0,  keepdim=True) + 1e-8

        # ------------ optimizer & LR scheduler ------------
        self.optimizer = optim.AdamW(self.model.parameters(),
                                     lr=config.get("training.learning_rate"),
                                     weight_decay=1e-5)

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.epochs, eta_min=1e-7
        )

        # ------------ edge list tensor (E,2) ------------
        self.edges = torch.tensor(dataset.edges,
                                  dtype=torch.long,
                                  device=self.device)     # (E,2)

    # ------------------------------------------------------------------ #
    def train_one_epoch(self):
        self.model.train()
        total_loss = 0.0

        for batch in self.loader:
            betas   = batch["betas"].to(self.device)    # (B,10)
            gammas  = batch["gammas"].to(self.device)   # (B,S)
            lf_disp = batch["lf_disp"].to(self.device)  # (B,V,3)

            # ---------- normalise inputs ----------
            bn = (betas  - self.betas_mean.to(self.device))  / self.betas_std.to(self.device)
            gn = (gammas - self.gammas_mean.to(self.device)) / self.gammas_std.to(self.device)
            x  = torch.cat([bn, gn], dim=1)             # (B, 10+S)

            # ---------- forward ----------
            pred = self.model(x)                        # (B,V,3)

            # ---------- base losses ----------
            diff = pred - lf_disp
            l1  = diff.abs().mean()
            mse = diff.pow(2).mean()

            # ---------- edge‑length loss ----------
            v1_p = pred[:, self.edges[:, 0], :]         # (B,E,3)
            v2_p = pred[:, self.edges[:, 1], :]
            v1_g = lf_disp[:, self.edges[:, 0], :]
            v2_g = lf_disp[:, self.edges[:, 1], :]

            len_p = (v1_p - v2_p).norm(dim=-1)          # (B,E)
            len_g = (v1_g - v2_g).norm(dim=-1)
            edge_loss = F.l1_loss(len_p, len_g)

            # ---------- Laplacian smoothness ----------
            lap = pred - pred.mean(dim=1, keepdim=True)
            lap_loss = lap.norm(dim=-1).mean()

            loss = l1 + 0.1 * mse + 1.0 * edge_loss + 0.005 * lap_loss

            # ---------- optimise ----------
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(self.loader)

    # ------------------------------------------------------------------ #
    def train(self):
        best = float("inf")

        for epoch in range(1, self.epochs + 1):
            loss_val = self.train_one_epoch()
            print(f"Epoch {epoch}/{self.epochs}  TrainLoss={loss_val:.6f}")
            self.scheduler.step()

            # save best checkpoint
            if loss_val < best:
                best = loss_val
                path = self.model_save_path or "output/model_best.pth"
                torch.save(self.model.state_dict(), path)
                print(f"Improved → saved to {path}")
