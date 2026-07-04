# C1 CONTRAST readout cross-stock block on the g012 scale=0.5 anchor (PowerShell / Windows).
# Reproduces the g012 anchor, then turns ON the readout cross-stock block in CONTRAST mode:
#   c_i = Wv·u_i − Σ_j a_ij·Wv·u_j   (each stock minus its attention-weighted peer set).
# Rationale (ledger 82/88): the standard pooling readout (R1) was ~anchor-parity (peak valid RankIC
# ~0.0804 vs anchor 0.0816); the contrast operator is the structurally-distinct untried variant —
# uniform attention ⇒ c_i = Wv·(u_i−mean(u)) = the pure cross-sectional-demeaned coordinate, which the
# CS-blind backbone (ledger 84) cannot produce per-stock; a flat cross-section ⇒ c_i≈0 (repeller, no
# loss-neutral parking spot), and no de-mean/out-norm ⇒ V-escape-immune. N stays 158 (~2h/seed, no
# SDPA-grid blowup — unlike the killed L-4).
#
# Usage (from repo root, quantEnv):
#   $env:SEED=42; $env:N_EPOCHS=25; .\scripts\readout_contrast_scale05.ps1
# Proxy smoke (cheap, trimmed window, 1 epoch):
#   $env:SEED=42; $env:N_EPOCHS=1; $env:PROXY=1; .\scripts\readout_contrast_scale05.ps1
param()

$py = "C:\Users\60585\miniconda3\envs\quantEnv\python.exe"
$repo = "C:\Users\60585\PycharmProjects\qibMacV2"
$env:PYTHONPATH = $repo

if (-not $env:SEED) { $env:SEED = "42" }
if (-not $env:N_EPOCHS) { $env:N_EPOCHS = "25" }
$seed = $env:SEED
$nep = $env:N_EPOCHS
$proxy = $env:PROXY

$env:QIB_RUN_SETTING = "readout_contrast_scale05_seed$seed"

# Model: g012 scale=0.5 anchor + readout cross-stock block in CONTRAST mode (qknorm on = R1 defense).
$env:QIB_MODEL_OVERRIDES_JSON = '{"time_tau_mlp_out_scale":0.5,"temporal_readout":"d1pma","temporal_readout_gate_init":-2.0,"use_regime_time_embedding":true,"use_regime_factor_gate":true,"router_use_layer_summary":true,"router_mode":"learned","use_readout_stock_attn":true,"readout_stock_attn_qknorm":true,"readout_stock_attn_gated":false,"readout_stock_attn_contrast":true}'

if ($proxy -eq "1") {
  $env:QIB_TRAINER_OVERRIDES_JSON = '{"seed":' + $seed + ',"n_epochs":' + $nep + ',"min_epochs":1,"consecutive_k":999999,"train_stop_threshold":null,"checkpoint_metric":"valid_rank_ic","checkpoint_mode":"max","use_tqdm":false}'
  # Trim date range for a cheap end-to-end pipeline smoke.
  $env:QIB_DATA_OVERRIDES_JSON = '{"kwargs":{"handler":{"kwargs":{"start_time":"2018-01-01","end_time":"2020-12-31","fit_end_time":"2019-12-31"}},"segments":{"train":["2018-01-01","2019-12-31"],"valid":["2020-07-01","2020-12-31"],"test":["2020-07-01","2020-12-31"]}}}'
} else {
  $env:QIB_TRAINER_OVERRIDES_JSON = '{"seed":' + $seed + ',"n_epochs":' + $nep + ',"min_epochs":' + $nep + ',"consecutive_k":999999,"train_stop_threshold":null,"checkpoint_metric":"valid_rank_ic","checkpoint_mode":"max","use_tqdm":false}'
  Remove-Item Env:\QIB_DATA_OVERRIDES_JSON -ErrorAction SilentlyContinue
}

Write-Host ">>> [C1-contrast] RUN_SETTING=$($env:QIB_RUN_SETTING) seed=$seed n_epochs=$nep proxy=$proxy"
& $py "$repo\work_flow.py"
