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
        lr: float = 1e-3,           # 大 LR 帮助跳出局部最优
        weight_decay: float = 1e-5,
        lambda_clash: float = 0.1,
        clash_cutoff: float = 2.0,
        coord_scale: float = 10.0,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.backbone = EGNNBackbone(in_channels=in_channels, hidden_dim=hidden_dim, num_layers=num_layers, edge_dim=edge_dim)
        self.coord_scale = coord_scale

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
        
        # 1. Scaling
        target_norm = batch.y / self.coord_scale
        
        # 2. MSE Loss
        loss_mse = F.mse_loss(delta_pred, target_norm)
        
        # 3. 辅助 Loss
        pred_mag = torch.norm(delta_pred, dim=-1)
        target_mag = torch.norm(target_norm, dim=-1)
        loss_mag = F.mse_loss(pred_mag, target_mag)
        
        cos_sim = F.cosine_similarity(delta_pred, target_norm, dim=-1, eps=1e-6)
        loss_cos = 1.0 - cos_sim.mean()

        # === 核心修正: 先计算 loss_clash，再计算总 loss ===
        
        # 4. 计算 Clash Loss (需还原尺度)
        delta_pred_real = delta_pred * self.coord_scale
        pos_pred = batch.pos + delta_pred_real
        loss_clash = self._clash_penalty(pos_pred, batch.edge_index)

        # 5. 最后计算总 Loss (此时 loss_clash 已定义)
        loss = loss_mse + 0.1 * loss_mag + 0.1 * loss_cos + self.hparams.lambda_clash * loss_clash
        # =================================================

        batch_size = getattr(batch, "num_graphs", None)
        if batch_size is None and hasattr(batch, "ptr"):
            batch_size = batch.ptr.numel() - 1

        self.log(f"{stage}/loss", loss, prog_bar=(stage != "train"), batch_size=batch_size)
        
        # 记录真实指标
        mse_real = F.mse_loss(delta_pred_real, batch.y)
        self.log(f"{stage}/loss_mse", mse_real, batch_size=batch_size)
        self.log(f"{stage}/pred_magnitude", torch.norm(delta_pred_real, dim=-1).mean(), batch_size=batch_size)
        
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        delta_pred = self.forward(batch)
        delta_pred_real = delta_pred * self.coord_scale

        loss_mse_real = F.mse_loss(delta_pred_real, batch.y)
        pos_pred = batch.pos + delta_pred_real
        loss_clash = self._clash_penalty(pos_pred, batch.edge_index)

        loss = loss_mse_real + self.hparams.lambda_clash * loss_clash

        self.log("test/loss", loss)
        self.log("test/loss_mse", loss_mse_real)
        self.log("test/loss_clash", loss_clash)

        # Baseline (预测0的 Loss)
        baseline_delta = torch.zeros_like(batch.y)
        self.log("test/baseline_mse", F.mse_loss(baseline_delta, batch.y))

        # 监控幅度 (如果这个值 > 0.1，说明模型活过来了)
        self.log("test/pred_magnitude", torch.norm(delta_pred_real, dim=-1).mean())

        # Flexible regions: only residues with ground-truth displacement magnitude > 1.0 Å
        gt_disp_mag = torch.norm(batch.y, dim=-1)
        flexible_mask = gt_disp_mag > 1.0
        if flexible_mask.any():
            flex_mse = F.mse_loss(delta_pred_real[flexible_mask], batch.y[flexible_mask])
            baseline_flex_mse = F.mse_loss(baseline_delta[flexible_mask], batch.y[flexible_mask])
            self.log("test/flexible_mse", flex_mse)
            self.log("test/flexible_rmsd", torch.sqrt(flex_mse))
            self.log("test/baseline_flexible_mse", baseline_flex_mse)
            self.log("test/baseline_flexible_rmsd", torch.sqrt(baseline_flex_mse))

        # pLDDT-binned metrics (raw pLDDT scale: 0~100)
        if hasattr(batch, "plddt") and batch.plddt is not None:
            plddt = batch.plddt
            if plddt.dim() > 1:
                plddt = plddt.squeeze(-1)

            plddt_bins = {
                "le50": plddt <= 50.0,
                "50to70": (plddt > 50.0) & (plddt <= 70.0),
                "gt70": plddt > 70.0,
            }
            for suffix, mask in plddt_bins.items():
                if mask.any():
                    bin_mse = F.mse_loss(delta_pred_real[mask], batch.y[mask])
                    baseline_bin_mse = F.mse_loss(baseline_delta[mask], batch.y[mask])
                    self.log(f"test/plddt_{suffix}_mse", bin_mse)
                    self.log(f"test/plddt_{suffix}_rmsd", torch.sqrt(bin_mse))
                    self.log(f"test/baseline_plddt_{suffix}_mse", baseline_bin_mse)
                    self.log(f"test/baseline_plddt_{suffix}_rmsd", torch.sqrt(baseline_bin_mse))

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
        return [optimizer], [scheduler]
