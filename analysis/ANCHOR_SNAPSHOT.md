# anchorTimeTau05 — frozen τ scale=0.5 baseline anchor

**Branch purpose:** preserve the designated baseline anchor (τ `time_tau_mlp_out_scale=0.5` + d1pma g≈0.12) at commit `b1df36d`, plus durable parameter/log snapshots for reproduction and comparison.

## Code anchor (reproduce training)

```bash
source scripts/baseline_g012_scale05.env.sh
export SEED=42   # or 43, 44
python work_flow.py
```

Canonical env: `scripts/baseline_g012_scale05.env.sh`  
Model override: `"time_tau_mlp_out_scale": 0.5` (+ temporal_readout=d1pma, gate_init=-2.0)

## Archived artifacts on this branch

| Path | Contents |
|---|---|
| `analysis/readout_redesign/archive_g012_baseline/` | n=3 training logs, resolved run configs, BASELINE_ANCHOR.md |
| `logs/init_ablation_scale05/` | g012 d1pma anchor run logs (seeds 42–44) |
| `logs/readout_matrix_scale05/` | readout design matrix logs at scale=0.5 |
| `diagnostic_runs/tau_phase2_25e_20260530/` | 3-seed × 6-scale sweep (stdout/stderr/manifests) |
| `diagnostic_runs/tau_phase2_seedext_20260604/` | seed extension 45–47 vs baseline |
| `diagnostic_runs/tau_forensics_refine_20260527/` | mid-range scale fill-in {0.2,0.5,0.8,1.5} |
| `analysis/tau_phase2_25e_3seed_20260604.csv` | aggregated 3-seed Phase-2 table |
| `analysis/tau_0p5_vs_baseline_n6_20260606.csv` | n=6 paired kill-check (scale=0.5 IR edge null) |
| `analysis/tau_scale05_diagnosis/` | post-anchor diagnosis memos, ceiling probes, L-6/L-4 results |
| `experiments_ledger.jsonl` | machine-readable experiment ledger through anchor era |

**Not included (gitignored):** `mlruns/` model weights — see MLflow run IDs in `archive_g012_baseline/BASELINE_ANCHOR.md`.

## Relation to qlibMac202607ckptCode

`qlibMac202607ckptCode` continues from this anchor with stock-expert, readout-redesign, and weekend experiment tracks. This branch stays frozen at the anchor decision point.
