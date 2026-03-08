# disp_1to5 ablation workflow

This workflow targets `test/disp_1to5_mse` with the smallest staged matrix that still checks parameter contribution.

## 1) Print the planned runs

```bash
python scripts/disp1to5_ablation.py plan
```

This prints 11 phase-1 runs that vary:
- `disp_focus_weight`
- `disp_outside_focus_weight`
- `mse_hard_beta`
- `lambda_cos`
- `lambda_mag`

and adds a focused interaction stress-test (`D1`) to check whether gains persist when hard-range/aux boosts are dialed down.

It also removes a redundant run that duplicated the current default settings.

## 2) Execute runs

Option A (manual): run each printed `python train.py ...` command.

Option B (automatic phase-1 runner):

```bash
python scripts/run_disp1to5_ablation_all.py
```

Use `--dry-run` to only print commands, or `--seed` to change the seed used for all phase-1 runs.

Each ablation run sets `study_name=disp1to5_ablation/<RUN_ID>_seed<SEED>`, so checkpoints are grouped under:

`checkpoints/disp1to5_ablation/<RUN_ID>_seed<SEED>/<timestamp>/`

## 3) Build a results CSV

Use the extraction script to parse each run's logs and write the CSV:

The extractor scans each run's checkpoint timestamp directory and, when needed, the matching Hydra output directory (`outputs/YYYY-MM-DD/HH-MM-SS`) for that same run. It does not fall back to unrelated global logger versions, so runs cannot accidentally reuse another run's metrics.

```bash
python scripts/disp1to5_ablation.py extract \
  --checkpoints-root checkpoints/disp1to5_ablation \
  --output artifacts/disp1to5_ablation_results.csv
```

The output CSV has one row per run and these columns:

- `run_id`
- `test_disp_1to5_mse`
- `test_loss_mse`
- `test_disp_0to0p5_mse`
- `test_disp_0p5to1_mse`
- `disp_focus_weight`
- `disp_outside_focus_weight`
- `mse_hard_beta`
- `lambda_cos`
- `lambda_mag`

## 4) Analyze contribution

```bash
python scripts/disp1to5_ablation.py analyze --results artifacts/disp1to5_ablation_results.csv
```

The script prints:
- Top runs by `test_disp_1to5_mse`
- Marginal mean deltas per parameter level (negative delta means better for `disp_1to5`)

## Acceptance gates

1. `test/disp_1to5_mse` improves >= 5% vs A0 control.
2. `test/loss_mse` degradation <= 10% vs A0.
3. `test/disp_0to0p5_mse` and `test/disp_0p5to1_mse` each degrade <= 15% vs A0.

---

## Additive/Constructive ablation runner (A0/B1/C1/D1/E1/F1)

For the one-factor-at-a-time additive design (all-off baseline A0, then restore one logical loss group at a time), use:

```bash
python scripts/run_additive_ablation_all.py
```

Use `--dry-run` to print all `train.py` commands without running, and `--extra-override` for shared Hydra overrides (for example, limiting epochs for quick sweeps).


## Aux stabilization sweep (Phase-1 recommended follow-up)

To run the aux-only stabilization grid on top of A0 (as suggested in `docs/ablation_next_steps.md`):

```bash
python scripts/run_aux_stabilization_sweep.py --seed 42
```

This sweeps:
- `lambda_cos` in `[0.05, 0.1, 0.2]`
- `lambda_mag` in `[0.01, 0.05, 0.1]`

while keeping shaping/hard/focus/clash disabled, and uses aux warmup epochs by default (`--warmup-epochs 10`).
