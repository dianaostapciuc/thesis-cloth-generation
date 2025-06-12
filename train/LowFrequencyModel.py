import torch.nn as nn

class LowFreqModel(nn.Module):
    def __init__(self, input_dim, num_verts, hidden_size=1024):
        super().__init__()
        output_dim = 3 * num_verts
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_dim)
        )

    def forward(self, x):
        out = self.mlp(x)
        return out.view(-1, out.shape[1] // 3, 3)
