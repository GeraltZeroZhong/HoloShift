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
        lr: float = 1e-3, 
        weight_decay: float = 1e-5,
        lambda_clash: float = 0.1,
        clash_cutoff: float = 2.0,
        coord_scale: float = 10.0,
        mse_weight_min: float = 1.0,
        mse_weight_peak: float = 50.0,
        mse_weight_steepness: float = 3.0,
        mse_weight_rise_center: float = 1.0,
        mse_weight_fall_center: float = 5.0,
        direction_mask_threshold: float = 0.5,
        lambda_cos: float = 1.0,
        lambda_mag: float = 1.0,
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
        target_norm = batch.y / self.coord_scale
        target_mag_real = torch.norm(batch.y, dim=-1)
        rise = torch.sigmoid(
            self.hparams.mse_weight_steepness * (target_mag_real - self.hparams.mse_weight_rise_center)
        )
        fall = 1.0 - torch.sigmoid(
            self.hparams.mse_weight_steepness * (target_mag_real - self.hparams.mse_weight_fall_center)
        )
        w_min = self.hparams.mse_weight_min
        w_peak = self.hparams.mse_weight_peak
        mse_weights = w_min + w_peak * rise * fall
        loss_node_mse = F.mse_loss(delta_pred, target_norm, reduction='none').mean(dim=-1)
        loss_mse = (loss_node_mse * mse_weights).mean()
        direction_mask = target_mag_real > self.hparams.direction_mask_threshold
        
        if direction_mask.sum() > 0:
            cos_sim = F.cosine_similarity(delta_pred[direction_mask], target_norm[direction_mask], dim=-1, eps=1e-6)
            loss_cos = (1.0 - cos_sim).mean()
        else:
            loss_cos = torch.tensor(0.0, device=self.device, dtype=delta_pred.dtype)

        pred_mag = torch.norm(delta_pred, dim=-1)
        target_mag = torch.norm(target_norm, dim=-1)
        loss_mag = F.mse_loss(pred_mag, target_mag)

        delta_pred_real = delta_pred * self.coord_scale
        pos_pred = batch.pos + delta_pred_real
        loss_clash = self._clash_penalty(pos_pred, batch.edge_index)
        
        loss = (
            loss_mse
            + self.hparams.lambda_cos * loss_cos
            + self.hparams.lambda_mag * loss_mag
            + self.hparams.lambda_clash * loss_clash
        )
        
        batch_size = getattr(batch, "num_graphs", None)
        if batch_size is None and hasattr(batch, "ptr"):
            batch_size = batch.ptr.numel() - 1

        self.log(f"{stage}/loss", loss, prog_bar=(stage != "train"), batch_size=batch_size)
        self.log(f"{stage}/loss_cos", loss_cos, batch_size=batch_size) 
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

        baseline_delta = torch.zeros_like(batch.y)
        self.log("test/baseline_mse", F.mse_loss(baseline_delta, batch.y))
        self.log("test/pred_magnitude", torch.norm(delta_pred_real, dim=-1).mean())

        gt_disp_mag = torch.norm(batch.y, dim=-1)
        flexible_mask = gt_disp_mag > 1.0
        if flexible_mask.any():
            flex_mse = F.mse_loss(delta_pred_real[flexible_mask], batch.y[flexible_mask])
            baseline_flex_mse = F.mse_loss(baseline_delta[flexible_mask], batch.y[flexible_mask])
            self.log("test/flexible_mse", flex_mse)
            self.log("test/flexible_rmsd", torch.sqrt(flex_mse))
            self.log("test/baseline_flexible_mse", baseline_flex_mse)
            self.log("test/baseline_flexible_rmsd", torch.sqrt(baseline_flex_mse))

        # Fine-grained displacement bins: [0,1), [1,2), ..., [9,10), [10,+inf)
        for lower in range(10):
            upper = lower + 1
            disp_mask = (gt_disp_mag >= float(lower)) & (gt_disp_mag < float(upper))
            count = int(disp_mask.sum().item())
            self.log(
                f"test/disp_{lower}to{upper}_count",
                torch.tensor(float(count), device=self.device),
                on_step=False,
                on_epoch=True,
                reduce_fx=torch.sum,
                batch_size=1,
            )
            if count > 0:
                bin_mse = F.mse_loss(delta_pred_real[disp_mask], batch.y[disp_mask])
                self.log(f"test/disp_{lower}to{upper}_mse", bin_mse)
                self.log(f"test/disp_{lower}to{upper}_rmsd", torch.sqrt(bin_mse))

        disp_mask_gt10 = gt_disp_mag >= 10.0
        count_gt10 = int(disp_mask_gt10.sum().item())
        self.log(
            "test/disp_gt10_count",
            torch.tensor(float(count_gt10), device=self.device),
            on_step=False,
            on_epoch=True,
            reduce_fx=torch.sum,
            batch_size=1,
        )
        if count_gt10 > 0:
            disp_gt10_mse = F.mse_loss(delta_pred_real[disp_mask_gt10], batch.y[disp_mask_gt10])
            self.log("test/disp_gt10_mse", disp_gt10_mse)
            self.log("test/disp_gt10_rmsd", torch.sqrt(disp_gt10_mse))

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

            # Fine-grained pLDDT bins: [0,10), [10,20), ..., [90,100]
            for lower in range(0, 100, 10):
                upper = lower + 10
                if upper < 100:
                    plddt_mask = (plddt >= float(lower)) & (plddt < float(upper))
                else:
                    plddt_mask = (plddt >= float(lower)) & (plddt <= float(upper))

                count = int(plddt_mask.sum().item())
                self.log(
                    f"test/plddt_{lower}to{upper}_count",
                    torch.tensor(float(count), device=self.device),
                    on_step=False,
                    on_epoch=True,
                    reduce_fx=torch.sum,
                    batch_size=1,
                )
                if count > 0:
                    bin_mse = F.mse_loss(delta_pred_real[plddt_mask], batch.y[plddt_mask])
                    self.log(f"test/plddt_{lower}to{upper}_mse", bin_mse)
                    self.log(f"test/plddt_{lower}to{upper}_rmsd", torch.sqrt(bin_mse))

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
        return [optimizer], [scheduler]
