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
        mse_hard_gamma: float = 1.5,
        mse_hard_beta: float = 3.0,
        mse_hard_p: float = 1.5,
        node_mse_cap: float = 2.0,
        direction_mask_threshold: float = 0.5,
        direction_mask_upper: float = 5.0,
        mse_hard_range_min: float = 1.0,
        mse_hard_range_max: float = 5.0,
        flexible_threshold: float = 1.0,
        plddt_low_cutoff: float = 50.0,
        plddt_mid_cutoff: float = 70.0,
        plddt_bin_size: int = 10,
        plddt_max: float = 100.0,
        lambda_cos: float = 1.0,
        lambda_mag: float = 1.0,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.backbone = EGNNBackbone(in_channels=in_channels, hidden_dim=hidden_dim, num_layers=num_layers, edge_dim=edge_dim)
        self.coord_scale = coord_scale
        self._test_disp_agg = {}

    def on_test_epoch_start(self):
        self._test_disp_agg = {}

    def _accumulate_disp_bin(self, suffix: str, sq_error: torch.Tensor, mask: torch.Tensor):
        n_elem = int(mask.sum().item()) * sq_error.size(-1)
        sse_sum = sq_error[mask].sum() if n_elem > 0 else torch.zeros((), device=self.device, dtype=sq_error.dtype)

        self.log(
            f"test/disp_{suffix}_sse_sum",
            sse_sum,
            on_step=False,
            on_epoch=True,
            reduce_fx=torch.sum,
            batch_size=1,
        )
        self.log(
            f"test/disp_{suffix}_n_elem",
            torch.tensor(float(n_elem), device=self.device),
            on_step=False,
            on_epoch=True,
            reduce_fx=torch.sum,
            batch_size=1,
        )

        if suffix not in self._test_disp_agg:
            self._test_disp_agg[suffix] = {
                "sse_sum": torch.zeros((), device=self.device, dtype=sq_error.dtype),
                "n_elem": 0,
            }

        self._test_disp_agg[suffix]["sse_sum"] = self._test_disp_agg[suffix]["sse_sum"] + sse_sum.detach()
        self._test_disp_agg[suffix]["n_elem"] += n_elem

    def on_test_epoch_end(self):
        for suffix, agg in self._test_disp_agg.items():
            if agg["n_elem"] <= 0:
                continue
            mse = agg["sse_sum"] / agg["n_elem"]
            self.log(f"test/disp_{suffix}_mse", mse)
            self.log(f"test/disp_{suffix}_rmsd", torch.sqrt(mse))

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

        hard_min = self.hparams.mse_hard_range_min
        hard_max = self.hparams.mse_hard_range_max
        in_hard_range = (target_mag_real >= hard_min) & (target_mag_real <= hard_max)
        hard_span = max(hard_max - hard_min, 1e-8)
        t = (target_mag_real.clamp(hard_min, hard_max) - hard_min) / hard_span
        w_hard = torch.ones_like(target_mag_real)
        w_hard[in_hard_range] = 1.0 + self.hparams.mse_hard_beta * (t[in_hard_range] ** self.hparams.mse_hard_p)
        mse_weights = mse_weights * w_hard

        eps = 1e-8
        mse_weights = mse_weights / (mse_weights.mean().detach() + eps)

        loss_node_mse = F.smooth_l1_loss(delta_pred, target_norm, reduction='none').mean(dim=-1)
        loss_mse = (loss_node_mse * mse_weights).mean()
        mask_focus = (
            (target_mag_real >= self.hparams.direction_mask_threshold)
            & (target_mag_real <= self.hparams.direction_mask_upper)
        )
        direction_mask = mask_focus
        
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

        gt_disp_mag = torch.norm(batch.y, dim=-1)
        flexible_mask = gt_disp_mag > self.hparams.flexible_threshold
        if flexible_mask.any():
            flex_mse = F.mse_loss(delta_pred_real[flexible_mask], batch.y[flexible_mask])
        else:
            flex_mse = torch.full((), float("inf"), device=self.device, dtype=delta_pred_real.dtype)
        self.log(f"{stage}/flexible_mse", flex_mse, batch_size=batch_size)
    
        return loss
    
    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        delta_pred = self.forward(batch)
        delta_pred_real = delta_pred * self.coord_scale
        sq_error = (delta_pred_real - batch.y) ** 2

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
        flexible_mask = gt_disp_mag > self.hparams.flexible_threshold
        if flexible_mask.any():
            flex_mse = F.mse_loss(delta_pred_real[flexible_mask], batch.y[flexible_mask])
            baseline_flex_mse = F.mse_loss(baseline_delta[flexible_mask], batch.y[flexible_mask])
            self.log("test/flexible_mse", flex_mse)
            self.log("test/flexible_rmsd", torch.sqrt(flex_mse))
            self.log("test/baseline_flexible_mse", baseline_flex_mse)
            self.log("test/baseline_flexible_rmsd", torch.sqrt(baseline_flex_mse))

        # Fine-grained displacement bins: [0,0.5), [0.5,1), [1,2), [2,3), [3,4), [4,5), [5,+inf)
        disp_bins = [
            (0.0, 0.5, "0to0p5"),
            (0.5, 1.0, "0p5to1"),
            (1.0, 2.0, "1to2"),
            (2.0, 3.0, "2to3"),
            (3.0, 4.0, "3to4"),
            (4.0, 5.0, "4to5"),
        ]
        for lower, upper, suffix in disp_bins:
            disp_mask = (gt_disp_mag >= lower) & (gt_disp_mag < upper)
            count = int(disp_mask.sum().item())
            self.log(
                f"test/disp_{suffix}_count",
                torch.tensor(float(count), device=self.device),
                on_step=False,
                on_epoch=True,
                reduce_fx=torch.sum,
                batch_size=1,
            )
            self._accumulate_disp_bin(suffix, sq_error, disp_mask)

        disp_mask_gt5 = gt_disp_mag >= 5.0
        count_gt5 = int(disp_mask_gt5.sum().item())
        self.log(
            "test/disp_gt5_count",
            torch.tensor(float(count_gt5), device=self.device),
            on_step=False,
            on_epoch=True,
            reduce_fx=torch.sum,
            batch_size=1,
        )
        self._accumulate_disp_bin("gt5", sq_error, disp_mask_gt5)

        # pLDDT-binned metrics (raw pLDDT scale: 0~100)
        if hasattr(batch, "plddt") and batch.plddt is not None:
            plddt = batch.plddt
            if plddt.dim() > 1:
                plddt = plddt.squeeze(-1)

            plddt_low_cutoff = self.hparams.plddt_low_cutoff
            plddt_mid_cutoff = self.hparams.plddt_mid_cutoff
            plddt_max = self.hparams.plddt_max
            plddt_bin_size = int(self.hparams.plddt_bin_size)

            plddt_bins = {
                "le50": plddt <= plddt_low_cutoff,
                "50to70": (plddt > plddt_low_cutoff) & (plddt <= plddt_mid_cutoff),
                "gt70": plddt > plddt_mid_cutoff,
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
            for lower in range(0, int(plddt_max), plddt_bin_size):
                upper = lower + plddt_bin_size
                if upper < plddt_max:
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
