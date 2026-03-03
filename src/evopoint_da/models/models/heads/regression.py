import torch.nn as nn


class DisplacementHead(nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 3),
        )

    def forward(self, x):
        return self.net(x)
