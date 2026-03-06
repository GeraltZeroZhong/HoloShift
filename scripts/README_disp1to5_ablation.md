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
