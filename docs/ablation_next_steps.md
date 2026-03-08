# Ablation code check and next-step plan

## Quick code check findings

1. **A0 relative-improvement should be exactly 0 in post-analysis tables**
   - In model test logging, relative improvement is computed against a zero-motion baseline as `(baseline - model) / baseline`.
   - So for the baseline run itself, any non-zero "Rel Improve (vs Baseline)" is likely a table export / rounding / row-alignment issue rather than model behavior.

2. **Current default config enables many loss groups simultaneously**
   - `configs/model/egnn_module.yaml` turns on shaping, hard mining, focus, cosine, magnitude, clash, and high-pLDDT regularization.
   - This makes interactions likely and can hide which component causes instability.

3. **E1 degradation is consistent with auxiliary term scale dominance**
   - In training, `loss = weighted_node + lambda_cos * loss_cos + lambda_mag * loss_mag + ...`.
   - With `lambda_cos=1.0` and `lambda_mag=1.0`, aux terms can dominate if their numerical scale is larger than the weighted node loss.

4. **Additive ablation runner is conceptually correct**
   - `scripts/run_additive_ablation_all.py` correctly defines A0 as all-off and restores one group per run.
   - This script is good for isolating single-factor effects.

## Recommended next-step experiments (priority ordered)

### Phase 1 — Stabilize auxiliary losses first (highest priority)

Run a small sweep with A0 baseline + aux-only, lowering aux weights by 10x–100x:
- `lambda_cos`: `[0.05, 0.1, 0.2]`
- `lambda_mag`: `[0.01, 0.05, 0.1]`

Goal: recover non-catastrophic `test/disp_1to5_mse` and normalize `test/pred_magnitude`.

### Phase 2 — Introduce warmup instead of full-strength from step 0

Use curriculum activation for components that previously hurt:
- hard mining (`mse_hard_beta`) warmup after base convergence,
- focus weights (`disp_focus_weight`, `disp_outside_focus_weight`) enabled mid-training,
- aux weights linearly ramped to target.

Goal: reduce optimization shock and avoid early gradient domination.

### Phase 3 — Re-test hard/focus with milder settings

Based on current defaults, try:
- hard mining: `mse_hard_beta` in `[0.5, 1.0, 1.5]` (instead of 3.0),
- focus: `disp_focus_weight` in `[1.1, 1.25]`,
- outside focus: `disp_outside_focus_weight` in `[0.95, 1.0]`.

Goal: keep the intended bias but avoid over-emphasizing 1–5 Å nodes.

### Phase 4 — Keep clash as a safe regularizer

Given current results, keep `lambda_clash` as a low-risk term while iterating other losses.

## Decision gates for accepting a variant

Compare against A0 with at least 3 seeds:
- `test/disp_1to5_mse`: mean improvement >= 3–5%,
- no major regression on short-displacement bins (`0–0.5`, `0.5–1.0`),
- `test/pred_magnitude` remains in the same scale band as ground-truth displacement.

If not passing gates, reject and continue tuning scales/warmup before adding more complexity.
