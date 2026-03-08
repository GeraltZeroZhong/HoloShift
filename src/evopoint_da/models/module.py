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
        lr: float = 1e-4, 
        weight_decay: float = 1e-5,
        lambda_clash: float = 0.1,
        clash_cutoff: float = 2.0,
        coord_scale: float = 10.0,
        direction_mask_threshold: float = 0.5,
        direction_mask_upper: float = 5.0,
        disp_focus_min: float = 1.0,
        disp_focus_max: float = 5.0,
        disp_focus_weight: float = 1.5,
        flexible_threshold: float = 1.0,
        lambda_cos: float = 0.5,
        lambda_mag: float = 1.0,
        cos_warmup_epochs: int = 0,
        mag_warmup_epochs: int = 0,
        focus_warmup_epochs: int = 0,
        plddt_gate_start: float = 90.0,
        plddt_gate_end: float = 100.0,
        lambda_high_plddt_l2: float = 0.1,
        lr_warmup_epochs: int = 10,
        inference_disp_multiplier: float = 2.0,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.backbone = EGNNBackbone(in_channels=in_channels, hidden_dim=hidden_dim, num_layers=num_layers, edge_dim=edge_dim)
        self.coord_scale = coord_scale
        self._test_disp_agg = {}

    def on_test_epoch_start(self):
        self._test_disp_agg = {}

    def _accumulate_disp_bin(
        self,
        suffix: str,
        sq_error: torch.Tensor,
        baseline_sq_error: torch.Tensor,
        mask: torch.Tensor,
    ):
        n_elem = int(mask.sum().item()) * sq_error.size(-1)
        sse_sum = sq_error[mask].sum() if n_elem > 0 else torch.zeros((), device=self.device, dtype=sq_error.dtype)
        baseline_sse_sum = (
            baseline_sq_error[mask].sum() if n_elem > 0 else torch.zeros((), device=self.device, dtype=sq_error.dtype)
        )

        self.log(
            f"test/disp_{suffix}_sse_sum",
            sse_sum,
            on_step=False,
            on_epoch=True,
            reduce_fx=torch.sum,
            batch_size=1,
        )

        if suffix not in self._test_disp_agg:
            self._test_disp_agg[suffix] = {
                "sse_sum": torch.zeros((), device=self.device, dtype=sq_error.dtype),
                "baseline_sse_sum": torch.zeros((), device=self.device, dtype=sq_error.dtype),
                "n_elem": 0,
            }

        self._test_disp_agg[suffix]["sse_sum"] = self._test_disp_agg[suffix]["sse_sum"] + sse_sum.detach()
        self._test_disp_agg[suffix]["baseline_sse_sum"] = (
            self._test_disp_agg[suffix]["baseline_sse_sum"] + baseline_sse_sum.detach()
        )
        self._test_disp_agg[suffix]["n_elem"] += n_elem

    def on_test_epoch_end(self):
        for suffix, agg in self._test_disp_agg.items():
            if agg["n_elem"] <= 0:
                continue
            mse = agg["sse_sum"] / agg["n_elem"]
            baseline_mse = agg["baseline_sse_sum"] / agg["n_elem"]
            self.log(f"test/disp_{suffix}_mse", mse)
            self.log(f"test/disp_{suffix}_rmsd", torch.sqrt(mse))
            self.log(f"test/baseline_disp_{suffix}_mse", baseline_mse)
            self.log(f"test/baseline_disp_{suffix}_rmsd", torch.sqrt(baseline_mse))

        # Aggregated displacement bin: [1.0, 5.0)
        agg_suffixes = ["1to2", "2to3", "3to4", "4to5"]
        agg_sse_sum = None
        agg_baseline_sse_sum = None
        agg_n_elem = 0
        for suffix in agg_suffixes:
            if suffix not in self._test_disp_agg:
                continue
            if agg_sse_sum is None:
                agg_sse_sum = self._test_disp_agg[suffix]["sse_sum"]
                agg_baseline_sse_sum = self._test_disp_agg[suffix]["baseline_sse_sum"]
            else:
                agg_sse_sum = agg_sse_sum + self._test_disp_agg[suffix]["sse_sum"]
                agg_baseline_sse_sum = agg_baseline_sse_sum + self._test_disp_agg[suffix]["baseline_sse_sum"]
            agg_n_elem += self._test_disp_agg[suffix]["n_elem"]

        if agg_n_elem > 0:
            disp_1to5_mse = agg_sse_sum / agg_n_elem
            baseline_disp_1to5_mse = agg_baseline_sse_sum / agg_n_elem
            self.log("test/disp_1to5_mse", disp_1to5_mse)
            self.log("test/baseline_disp_1to5_mse", baseline_disp_1to5_mse)
            rel_improve = (baseline_disp_1to5_mse - disp_1to5_mse) / baseline_disp_1to5_mse.clamp_min(1e-8)
            self.log("test/summary/disp_1to5_rel_improve_vs_baseline", rel_improve)

    def forward(self, batch):
        _, pos_updated = self.backbone(batch.x, batch.pos, batch.edge_index, batch.edge_attr)
        return pos_updated - batch.pos

    def predict_displacement(self, batch):
        """Return final displacement in real coordinate space for inference/export usage."""
        delta_pred = self.forward(batch)
        return delta_pred * self.coord_scale * self.hparams.inference_disp_multiplier

    def _clash_penalty(self, pos_pred: torch.Tensor, edge_index: torch.Tensor):
        if edge_index.numel() == 0:
            return torch.zeros((), device=pos_pred.device, dtype=pos_pred.dtype)
        src, dst = edge_index
        dist = torch.norm(pos_pred[src] - pos_pred[dst], dim=-1)
        return F.relu(self.hparams.clash_cutoff - dist).mean()

    def _log_disp_group_metrics(
        self,
        stage: str,
        delta_pred_real: torch.Tensor,
        y_true: torch.Tensor,
        gt_disp_mag: torch.Tensor,
        batch_size: int | None,
    ):
        groups = [
            (0.0, 1.0, "0to1"),
            (1.0, 5.0, "1to5"),
            (5.0, None, "gt5"),
        ]
        for low, high, suffix in groups:
            if high is None:
                mask = gt_disp_mag >= low
            else:
                mask = (gt_disp_mag >= low) & (gt_disp_mag < high)

            count = int(mask.sum().item())
            self.log(f"{stage}/disp_group/{suffix}_count", float(count), batch_size=batch_size)
            if count > 0:
                mse = F.mse_loss(delta_pred_real[mask], y_true[mask])
                mae = F.l1_loss(delta_pred_real[mask], y_true[mask])
                self.log(f"{stage}/disp_group/{suffix}_mse", mse, batch_size=batch_size)
                self.log(f"{stage}/disp_group/{suffix}_rmsd", torch.sqrt(mse), batch_size=batch_size)
                self.log(f"{stage}/disp_group/{suffix}_mae", mae, batch_size=batch_size)

    def _shared_step(self, batch, stage: str):
        delta_pred = self.forward(batch)
        high_plddt_l2 = torch.zeros((), device=self.device, dtype=delta_pred.dtype)

        def _warmup_factor(warmup_epochs: int) -> float:
            if warmup_epochs <= 0:
                return 1.0
            return min(1.0, float(self.current_epoch + 1) / float(warmup_epochs))

        cos_warmup = _warmup_factor(int(self.hparams.cos_warmup_epochs))
        mag_warmup = _warmup_factor(int(self.hparams.mag_warmup_epochs))
        focus_warmup = _warmup_factor(int(self.hparams.focus_warmup_epochs))

        if hasattr(batch, "plddt") and batch.plddt is not None:
            plddt = batch.plddt
            if plddt.dim() > 1:
                plddt = plddt.squeeze(-1)

            gate_start = self.hparams.plddt_gate_start
            gate_end = max(self.hparams.plddt_gate_end, gate_start + 1e-6)
            uncertainty_gate = ((gate_end - plddt) / (gate_end - gate_start)).clamp(0.0, 1.0)
            delta_pred = delta_pred * uncertainty_gate.unsqueeze(-1)

            high_plddt_mask = plddt > gate_start
            if high_plddt_mask.any():
                high_plddt_l2 = (delta_pred[high_plddt_mask] ** 2).mean()

        target_norm = batch.y / self.coord_scale
        target_mag_real = torch.norm(batch.y, dim=-1)
        mse_weights = torch.ones_like(target_mag_real)

        focus_min = self.hparams.disp_focus_min
        focus_max = self.hparams.disp_focus_max
        in_focus = (target_mag_real >= focus_min) & (target_mag_real < focus_max)
        focus_weights = torch.ones_like(target_mag_real)
        focus_weights[in_focus] = self.hparams.disp_focus_weight
        mse_weights = mse_weights * (1.0 + focus_warmup * (focus_weights - 1.0))

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
        
        lambda_cos_eff = self.hparams.lambda_cos * cos_warmup
        lambda_mag_eff = self.hparams.lambda_mag * mag_warmup

        loss = (
            loss_mse
            + lambda_cos_eff * loss_cos
            + lambda_mag_eff * loss_mag
            + self.hparams.lambda_clash * loss_clash
            + self.hparams.lambda_high_plddt_l2 * high_plddt_l2
        )
        
        batch_size = getattr(batch, "num_graphs", None)
        if batch_size is None and hasattr(batch, "ptr"):
            batch_size = batch.ptr.numel() - 1

        self.log(f"{stage}/loss", loss, prog_bar=(stage != "train"), batch_size=batch_size)
        self.log(f"{stage}/loss_cos", loss_cos, batch_size=batch_size) 
        self.log(f"{stage}/loss_high_plddt_l2", high_plddt_l2, batch_size=batch_size)
        self.log(f"{stage}/loss_components/weighted_node", loss_mse, batch_size=batch_size)
        self.log(f"{stage}/loss_components/cos", loss_cos, batch_size=batch_size)
        self.log(f"{stage}/loss_components/magnitude", loss_mag, batch_size=batch_size)
        self.log(f"{stage}/weights/lambda_cos_eff", lambda_cos_eff, batch_size=batch_size)
        self.log(f"{stage}/weights/lambda_mag_eff", lambda_mag_eff, batch_size=batch_size)
        self.log(f"{stage}/weights/cos_warmup", cos_warmup, batch_size=batch_size)
        self.log(f"{stage}/weights/mag_warmup", mag_warmup, batch_size=batch_size)
        self.log(f"{stage}/weights/focus_warmup", focus_warmup, batch_size=batch_size)
        self.log(f"{stage}/loss_components/clash", loss_clash, batch_size=batch_size)
        self.log(f"{stage}/loss_components/high_plddt_l2", high_plddt_l2, batch_size=batch_size)
        mse_real = F.mse_loss(delta_pred_real, batch.y)
        self.log(f"{stage}/loss_mse", mse_real, batch_size=batch_size)
        self.log(f"{stage}/pred_magnitude", torch.norm(delta_pred_real, dim=-1).mean(), batch_size=batch_size)
        self.log(f"{stage}/weights/mean", mse_weights.mean(), batch_size=batch_size)
        self.log(f"{stage}/weights/std", mse_weights.std(unbiased=False), batch_size=batch_size)
        self.log(f"{stage}/weights/focus_frac", in_focus.float().mean(), batch_size=batch_size)

        gt_disp_mag = torch.norm(batch.y, dim=-1)
        self._log_disp_group_metrics(stage, delta_pred_real, batch.y, gt_disp_mag, batch_size)
        if stage == "val":
            disp_1to5_mask = (gt_disp_mag >= 1.0) & (gt_disp_mag < 5.0)
            if disp_1to5_mask.any():
                disp_1to5_mse = F.mse_loss(delta_pred_real[disp_1to5_mask], batch.y[disp_1to5_mask])
                self.log(
                    "val/disp_1to5_mse",
                    disp_1to5_mse,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=True,
                    batch_size=int(disp_1to5_mask.sum().item()),
                )

        flexible_mask = gt_disp_mag > self.hparams.flexible_threshold
        flex_count = int(flexible_mask.sum().item())
        if flexible_mask.any():
            flex_mse = F.mse_loss(delta_pred_real[flexible_mask], batch.y[flexible_mask])
        else:
            flex_mse = torch.zeros((), device=self.device, dtype=delta_pred_real.dtype)
        self.log(f"{stage}/flexible_mse", flex_mse, batch_size=batch_size)
        self.log(
            f"{stage}/flexible_count",
            torch.tensor(float(flex_count), device=self.device),
            batch_size=batch_size,
        )
    
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
        self.log("test/loss_mae", F.l1_loss(delta_pred_real, batch.y))
        self.log("test/loss_clash", loss_clash)
        self.log("test/loss_components/weighted_node", loss_mse_real)
        self.log("test/loss_components/clash", loss_clash)

        baseline_delta = torch.zeros_like(batch.y)
        baseline_sq_error = baseline_delta - batch.y
        baseline_sq_error = baseline_sq_error ** 2
        baseline_mse = F.mse_loss(baseline_delta, batch.y)
        self.log("test/baseline_mse", baseline_mse)
        self.log("test/baseline_mae", F.l1_loss(baseline_delta, batch.y))
        overall_rel_improve = (baseline_mse - loss_mse_real) / baseline_mse.clamp_min(1e-8)
        self.log("test/summary/overall_rel_improve_vs_baseline", overall_rel_improve)
        self.log("test/pred_magnitude", torch.norm(delta_pred_real, dim=-1).mean())

        gt_disp_mag = torch.norm(batch.y, dim=-1)
        self._log_disp_group_metrics("test", delta_pred_real, batch.y, gt_disp_mag, batch_size=1)
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
            self._accumulate_disp_bin(suffix, sq_error, baseline_sq_error, disp_mask)

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
        self._accumulate_disp_bin("gt5", sq_error, baseline_sq_error, disp_mask_gt5)

        # pLDDT-binned metrics (raw pLDDT scale: 0~100)
        if hasattr(batch, "plddt") and batch.plddt is not None:
            plddt = batch.plddt
            if plddt.dim() > 1:
                plddt = plddt.squeeze(-1)

            # pLDDT bins: [0,60), [60,70), [70,80), [80,90), [90,100]
            plddt_ranges = [(0, 60), (60, 70), (70, 80), (80, 90), (90, 100)]
            for lower, upper in plddt_ranges:
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
                    baseline_bin_mse = F.mse_loss(baseline_delta[plddt_mask], batch.y[plddt_mask])
                    self.log(f"test/plddt_{lower}to{upper}_mse", bin_mse)
                    self.log(f"test/plddt_{lower}to{upper}_rmsd", torch.sqrt(bin_mse))
                    self.log(f"test/baseline_plddt_{lower}to{upper}_mse", baseline_bin_mse)
                    self.log(f"test/baseline_plddt_{lower}to{upper}_rmsd", torch.sqrt(baseline_bin_mse))

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)

        warmup_epochs = max(0, int(self.hparams.lr_warmup_epochs))
        total_epochs = int(getattr(self.trainer, "max_epochs", 100))
        if total_epochs <= 0:
            total_epochs = 100

        if warmup_epochs > 0:
            warmup = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=1e-8,
                end_factor=1.0,
                total_iters=warmup_epochs,
            )
            cosine_epochs = max(1, total_epochs - warmup_epochs)
            cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cosine_epochs)
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup, cosine],
                milestones=[warmup_epochs],
            )
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, total_epochs))

        return [optimizer], [scheduler]
