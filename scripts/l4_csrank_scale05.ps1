# L-4 CS-rank INPUT channels on the g012 scale=0.5 anchor (PowerShell / Windows).
# Reproduces analysis/readout_redesign/archive_g012_baseline/BASELINE_ANCHOR.md config, then
# APPENDS per-day cross-sectional rank channels (module.utils.processors.CSRankAppend) to the
# handler infer_processors so train (DK_L) and eval (DK_I) share the identical 158+158=316 schema,
# and turns on strict_feature_schema so a schema mismatch raises instead of silently truncating.
#
# Usage (from repo root, quantEnv):
#   $env:SEED=42; $env:N_EPOCHS=25; .\scripts\l4_csrank_scale05.ps1
# Proxy smoke (cheap):
#   $env:SEED=42; $env:N_EPOCHS=1; $env:PROXY=1; .\scripts\l4_csrank_scale05.ps1
param()

$py = "C:\Users\60585\miniconda3\envs\quantEnv\python.exe"
$repo = "C:\Users\60585\PycharmProjects\qibMacV2"
$env:PYTHONPATH = $repo

# L-4 doubles the feature axis (158->316) => time expert batch B*N=94800 exceeds the fused SDPA grid
# limit. chunk=16384 is the measured speed/memory sweet spot on a 12GB card (~44min/epoch, peak 7.9GB;
# smaller only trims memory at a slight speed cost, larger tips VRAM). See tmp_run/bench_step_time.py.
if (-not $env:QIB_SDPA_CHUNK) { $env:QIB_SDPA_CHUNK = "16384" }

if (-not $env:SEED) { $env:SEED = "42" }
if (-not $env:N_EPOCHS) { $env:N_EPOCHS = "25" }
$seed = $env:SEED
$nep = $env:N_EPOCHS
$proxy = $env:PROXY

$env:QIB_RUN_SETTING = "l4_csrank_scale05_seed$seed"

# Model: identical to the g012 scale=0.5 anchor, plus use_grad_checkpoint=true. Checkpointing is
# numerically exact (verified: plain-vs-ckpt grad diff == plain-vs-plain backend noise floor) and is
# REQUIRED here because L-4 doubles the feature axis (158->316) => the full activation graph for the
# B*N=94800 time expert exceeds the 12GB card (~13GB reserved) without it; with it, peak ~5GB.
$env:QIB_MODEL_OVERRIDES_JSON = '{"time_tau_mlp_out_scale":0.5,"temporal_readout":"d1pma","temporal_readout_gate_init":-2.0,"use_regime_time_embedding":true,"use_regime_factor_gate":true,"router_use_layer_summary":true,"router_mode":"learned","use_grad_checkpoint":true}'

# Trainer: anchor training discipline + strict feature-schema guard (L-4).
if ($proxy -eq "1") {
  $env:QIB_TRAINER_OVERRIDES_JSON = '{"seed":' + $seed + ',"n_epochs":' + $nep + ',"min_epochs":1,"consecutive_k":999999,"train_stop_threshold":null,"checkpoint_metric":"valid_rank_ic","checkpoint_mode":"max","use_tqdm":false,"strict_feature_schema":true}'
} else {
  $env:QIB_TRAINER_OVERRIDES_JSON = '{"seed":' + $seed + ',"n_epochs":' + $nep + ',"min_epochs":' + $nep + ',"consecutive_k":999999,"train_stop_threshold":null,"checkpoint_metric":"valid_rank_ic","checkpoint_mode":"max","use_tqdm":false,"strict_feature_schema":true}'
}

# Data: append CSRankAppend AFTER RobustZScoreNorm+Fillna (rank is monotone => invariant to the
# global z-score for non-clipped values; placed last so the rank channels are NOT re-normalized).
$infer = '[{"class":"RobustZScoreNorm","kwargs":{"fields_group":"feature","clip_outlier":true}},{"class":"Fillna","kwargs":{"fields_group":"feature"}},{"class":"CSRankAppend","module_path":"module.utils.processors","kwargs":{"fields_group":"feature","suffix":"__csr","center":true}}]'

if ($proxy -eq "1") {
  # Trim date range for a cheap end-to-end pipeline smoke.
  $env:QIB_DATA_OVERRIDES_JSON = '{"kwargs":{"handler":{"kwargs":{"start_time":"2018-01-01","end_time":"2020-12-31","fit_end_time":"2019-12-31","infer_processors":' + $infer + '}},"segments":{"train":["2018-01-01","2019-12-31"],"valid":["2020-07-01","2020-12-31"],"test":["2020-07-01","2020-12-31"]}}}'
} else {
  $env:QIB_DATA_OVERRIDES_JSON = '{"kwargs":{"handler":{"kwargs":{"infer_processors":' + $infer + '}}}}'
}

Write-Host ">>> [L-4] RUN_SETTING=$($env:QIB_RUN_SETTING) seed=$seed n_epochs=$nep proxy=$proxy"
& $py "$repo\work_flow.py"
