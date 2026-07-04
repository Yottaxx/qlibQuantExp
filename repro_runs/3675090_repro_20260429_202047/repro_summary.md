# Reproduction Summary

## Run Identity

- Commit: `3675090480ea25c84fc073716460ec8277eafd5b`
- Commit note: `best ic_raw:0.070609 | rank_ic:0.078597`
- Run directory: `C:\Users\60585\PycharmProjects\qibMacV2\repro_runs\3675090_repro_20260429_202047`
- Command: `"C:\Users\60585\miniconda3\envs\quantEnv\python.exe" -u work_flow.py`
- Started: `2026-04-29T20:21:37.069+08:00`
- Ended: `2026-04-30T01:05:41.433+08:00`
- Duration: `04:44:04.364`
- MLflow experiment ID: `591706357338261427`
- MLflow run/recorder ID: `f602b2ca0a9142f8b589d11979a23e35`
- MLflow status: `3` (finished)

## Runtime Config Notes

- Python: `C:\Users\60585\miniconda3\envs\quantEnv\python.exe`
- Qlib data: `%USERPROFILE%\.qlib\qlib_data\cn_data`
- GPU detected before run: `NVIDIA GeForce RTX 4070 SUPER`
- `market_state_path` was changed for this run to `data/market_state_csi300.pkl`.
- The working tree already had the evaluation window set to `2020-07-01` through `2022-12-31` for test/backtest; that is the window used by this run.

## Key Training Result

- Best validation epoch observed: `17`
- Validation `ic_raw`: `0.070942`
- Validation `rank_ic`: `0.078940`
- Validation `loss_main`: `0.041822`
- This matches/slightly exceeds the target commit note: `ic_raw=0.070609`, `rank_ic=0.078597`.
- Training stopped at epoch `32`: `loss_main=1.329779 <= 1.33` for `k=2`.
- Restored best training checkpoint: `loss_main=1.329056`.

## Final Test / Report Metrics

- Test IC mean: `0.061809`
- Test ICIR: `0.420690`
- Test IC HAC t-stat: `6.9` with `lags=5`
- Test RankIC mean: `0.068077`
- Test RankIC IR: `0.469358`
- Test RankIC HAC t-stat: `7.8` with `lags=5`
- Backtest with cost annualized return: `17.4099%`
- Backtest with cost information ratio: `1.4033`
- Backtest with cost max drawdown: `-11.3949%`
- Backtest turnover: `32.64%`
- Gate time_ratio stats on test: `mean=0.664, std=0.068, p10=0.606, p90=0.708`

## Artifacts

- Console stdout: `stdout.log`
- Console stderr: `stderr.log`
- Run manifest: `run_manifest.txt`
- Source snapshot: `work_flow.snapshot.py`
- Source diff: `work_flow.diff`
- Full report: `C:\Users\60585\PycharmProjects\qibMacV2\mlruns\591706357338261427\f602b2ca0a9142f8b589d11979a23e35\kdd_report.md`
- MLflow run directory: `C:\Users\60585\PycharmProjects\qibMacV2\mlruns\591706357338261427\f602b2ca0a9142f8b589d11979a23e35`
- Main Qlib graph example: `qlib_analysis_position_report_graph.html`
- Training curve: `train_curves_mse_rankic.png`

## Reproducibility Notes

- `work_flow.snapshot.py`, `run_matrix.snapshot.yaml`, `experiments.snapshot.md`, `git_head.txt`, `git_status_short.txt`, `work_flow.diff`, environment text, GPU text, and raw stdout/stderr were captured before or during the run.
- The only source line changed by the assistant before the run was `market_state_path` in `work_flow.py`.
