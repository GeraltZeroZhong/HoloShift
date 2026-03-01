import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from .backbones.egnn import EGNNBackbone


class EvoPointLitModule(pl.LightningModule):
    def __init__(
        self,
        in_channels: int = 130,
        hidden_dim: int = 128,
        num_layers: int = 4,
        edge_dim: int = 2,
        lr: float = 3e-4,
        weight_decay: float = 1e-4,
        lambda_clash: float = 0.1,
        clash_cutoff: float = 2.0,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.backbone = EGNNBackbone(in_channels=in_channels, hidden_dim=hidden_dim, num_layers=num_layers, edge_dim=edge_dim)

    def forward(self, batch):
        _, pos_updated = self.backbone(batch.x, batch.pos, batch.edge_index, batch.edge_attr)
        return pos_updated - batch.pos

    def _clash_penalty(self, pos_pred: torch.Tensor, edge_index: torch.Tensor):
        if edge_index.numel() == 0:
            return torch.zeros((), device=pos_pred.device, dtype=pos_pred.dtype)

        src, dst = edge_index
        dist = torch.norm(pos_pred[src] - pos_pred[dst], dim=-1)
        return F.relu(self.hparams.clash_cutoff - dist).mean()

    def _shared_step(self, batch, stage: str):
        delta_pred = self.forward(batch)
        loss_mse = F.mse_loss(delta_pred, batch.y)
        pos_pred = batch.pos + delta_pred
        loss_clash = self._clash_penalty(pos_pred, batch.edge_index)
        loss = loss_mse + self.hparams.lambda_clash * loss_clash

        self.log(f"{stage}/loss", loss, prog_bar=(stage != "train"))
        self.log(f"{stage}/loss_mse", loss_mse)
        self.log(f"{stage}/loss_clash", loss_clash)
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        self._shared_step(batch, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
        return [optimizer], [scheduler]
