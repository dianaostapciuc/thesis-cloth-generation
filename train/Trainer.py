import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F

class Trainer:
    def __init__(self, model, dataset, config, model_save_path=None):
        self.device = torch.device(config.get("training.device"))
        self.model  = model.to(self.device)
        self.loader = DataLoader(dataset,
                                 batch_size=config.get("training.batch_size"),
                                 shuffle=True,
                                 num_workers=4,
                                 drop_last=True)

        self.epochs  = config.get("training.epochs")
        self.model_save_path = model_save_path

        betas_all  = torch.stack([s["betas"]  for s in dataset])
        gammas_all = torch.stack([s["gammas"] for s in dataset])

        self.betas_mean  = betas_all.mean(0, keepdim=True)
        self.betas_std   = betas_all.std(0,  keepdim=True) + 1e-8
        self.gammas_mean = gammas_all.mean(0, keepdim=True)
        self.gammas_std  = gammas_all.std(0,  keepdim=True) + 1e-8

        self.optimizer = optim.AdamW(self.model.parameters(),
                                     lr=config.get("training.learning_rate"),
                                     weight_decay=1e-5)

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.epochs, eta_min=1e-7
        )

        self.edges = torch.tensor(dataset.edges,
                                  dtype=torch.long,
                                  device=self.device)

    def train_one_epoch(self):
        self.model.train()
        total_loss = 0.0

        for batch in self.loader:
            betas   = batch["betas"].to(self.device)
            gammas  = batch["gammas"].to(self.device)
            lf_disp = batch["lf_disp"].to(self.device)

            bn = (betas  - self.betas_mean.to(self.device))  / self.betas_std.to(self.device)
            gn = (gammas - self.gammas_mean.to(self.device)) / self.gammas_std.to(self.device)
            x  = torch.cat([bn, gn], dim=1)

            pred = self.model(x)

            diff = pred - lf_disp
            l1  = diff.abs().mean()
            mse = diff.pow(2).mean()

            v1_p = pred[:, self.edges[:, 0], :]
            v2_p = pred[:, self.edges[:, 1], :]
            v1_g = lf_disp[:, self.edges[:, 0], :]
            v2_g = lf_disp[:, self.edges[:, 1], :]

            len_p = (v1_p - v2_p).norm(dim=-1)
            len_g = (v1_g - v2_g).norm(dim=-1)
            edge_loss = F.l1_loss(len_p, len_g)

            lap = pred - pred.mean(dim=1, keepdim=True)
            lap_loss = lap.norm(dim=-1).mean()

            loss = l1 + 0.1 * mse + 1.0 * edge_loss + 0.005 * lap_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(self.loader)

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
