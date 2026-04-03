import torch
import torch.nn as nn

class EGNNLayer(nn.Module):
    def __init__(
        self,
        feat_dim: int,
        edge_dim: int = 2,
        hidden_dim: int = 128,
        coord_init_gain: float = 0.001,
    ):
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(feat_dim * 2 + edge_dim + 1, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(feat_dim + hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, feat_dim),
        )
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), 
            nn.SiLU(), 
            nn.Linear(hidden_dim, 1)
        )
        
        # === 核心修改: 必须给一个非零的初始扰动 ===
        # gain=0.001 既保证了初始位移很小（不破坏结构），又保证了梯度存在
        nn.init.xavier_uniform_(self.coord_mlp[-1].weight, gain=coord_init_gain)
        nn.init.zeros_(self.coord_mlp[-1].bias)
        # =========================================

    def forward(self, x, pos, edge_index, edge_attr):
        src, dst = edge_index
        rel = pos[dst] - pos[src]
        r2 = (rel * rel).sum(dim=-1, keepdim=True)

        m_ij = self.edge_mlp(torch.cat([x[src], x[dst], edge_attr, r2], dim=-1))

        agg_m = torch.zeros(x.size(0), m_ij.size(1), device=x.device)
        agg_m.index_add_(0, dst, m_ij)
        x = x + self.node_mlp(torch.cat([x, agg_m], dim=-1))

        trans = rel * self.coord_mlp(m_ij)
        agg_trans = torch.zeros_like(pos)
        agg_trans.index_add_(0, dst, trans)
        pos = pos + agg_trans
        return x, pos

class EGNNBackbone(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_dim: int = 128,
        num_layers: int = 4,
        edge_dim: int = 2,
        coord_init_gain: float = 0.001,
    ):
        super().__init__()
        self.input_proj = nn.Linear(in_channels, hidden_dim)
        self.layers = nn.ModuleList(
            [
                EGNNLayer(
                    hidden_dim,
                    edge_dim=edge_dim,
                    hidden_dim=hidden_dim,
                    coord_init_gain=coord_init_gain,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, x, pos, edge_index, edge_attr):
        x = self.input_proj(x)
        for layer in self.layers:
            x, pos = layer(x, pos, edge_index, edge_attr)
        return x, pos
