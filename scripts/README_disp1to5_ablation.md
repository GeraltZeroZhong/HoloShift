# disp_1to5 ablation workflow

This workflow targets `test/disp_1to5_mse` with the smallest staged matrix that still checks parameter contribution.

## 1) Print the planned runs

```bash
python scripts/disp1to5_ablation.py plan
```

This prints 9 phase-1 runs that vary:
- `disp_focus_weight`
- `disp_outside_focus_weight`
- `mse_hard_beta`
- `lambda_cos`
- `lambda_mag`

and optional phase-2 replication guidance for top runs across multiple seeds.

## 2) Execute runs

Run each printed `python train.py ...` command.

Each ablation run sets `study_name=disp1to5_ablation/<RUN_ID>_seed<SEED>`, so checkpoints are grouped under:

`checkpoints/disp1to5_ablation/<RUN_ID>_seed<SEED>/<timestamp>/`

## 3) Build a results CSV

Create a CSV with one row per run and these columns:

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
