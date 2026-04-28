# Repro Archive: hsf_legacy_noinner_simplepool_mhc_v7_ampfp16_factorid_zloss_wd001_b0998_mscsi300_20260428_144940

Archived at: 2026-04-28T20:59:40

## Best Observed Performance

- Best Valid RankIC epoch: 16
- Best Valid RankIC: 0.075459
- Best Valid IC: 0.062379
- Best Valid loss_main/MSE: 0.051479
- Last archived Valid epoch: 27
- Last archived Valid RankIC: 0.067686
- Last archived Valid IC: 0.057691

## Command

```powershell
"C:\Users\60585\miniconda3\envs\quantEnv\python.exe" "-u" "C:\Users\60585\PycharmProjects\qlibQuantExp\scripts\run_workflow_market_state_variant.py" "--qlib_kernels" "1" "--workflow_module" "next_step.hierarchical_state_field.experiment.configs.ref_all_hsf_csi300_tri_scope_stocktime_simplepool_mhc_final" "--market_state_path" "artifacts/market_state/market_state_csi300.pkl" "--market_day_summary_path" "artifacts/market_state/daily_market_observation_tri_scope_v1.pkl" "--label" "csi300_tri_scope_legacy_noinner_simplepool_mhc_v7" "--experiment_suffix" "hsf_legacy_noinner_simplepool_mhc_v7_ampfp16_factorid_zloss_wd001_b0998_mscsi300_20260428_144940_stocktime_simplepool_mhc_v7_final" "--seed" "42" "--enable_local_counterfactual_diag" "1" "--enable_expert_advantage_diag" "1" "--skip_visuals"
```

## Key Settings

```json
{
  "feature_tokenizer_add_factor_id": true,
  "router_aux_loss_type": "z_loss",
  "router_z_loss_coef": 0.01,
  "use_hierarchical_state_field": false,
  "factor_gate_scale": 1.0,
  "factor_gate_shift_scale": 0.2,
  "router_summary_source": "batch",
  "router_summary_fusion_mode": "default",
  "use_inner_cross_stock_attention": false,
  "inner_cross_stock_mode": "day_token",
  "use_cross_stock_attention": false,
  "pooling_mode": "simple_static",
  "pooling_alpha": 0.7,
  "pooling_summary_source": "none",
  "temporal_pooling_mode": "gru_mhc_lite_v1",
  "temporal_mhc_mix_init": 0.05,
  "temporal_mhc_mix_max": 0.25,
  "dropout": 0.3,
  "regime_macro_dropout": 0.2,
  "lr": 1e-05,
  "optimizer": "adamw",
  "weight_decay": 0.01,
  "adam_betas": [
    0.9,
    0.998
  ],
  "adam_eps": 1e-08,
  "adam_amsgrad": false,
  "adam_foreach": true,
  "adam_fused": null,
  "adamw_decay_matrix_only": true,
  "n_epochs": 40,
  "early_stop": 0,
  "train_stop_threshold": null,
  "seed": 42
}
```

## Files

- `status_snapshot.json`: runner status and original command.
- `commands_snapshot.txt`: command emitted by the runner.
- `best_metrics_and_settings.json`: parsed metrics and settings at archive time.
- `config_snapshot.py`: exact config snapshot.
- `runner_snapshot.py`: exact runner snapshot.
- `git_diff_before_commit.patch`: tracked working-tree code diff before commit.
- `git_diff_cached_before_commit.patch`: staged diff before commit.
- `stdout_tail_snapshot.txt` / `stderr_tail_snapshot.txt`: log tails at archive time.
