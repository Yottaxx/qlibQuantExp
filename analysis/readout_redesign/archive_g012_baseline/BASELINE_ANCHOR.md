# BASELINE ANCHOR — τ scale=0.5 + d1pma g=0.12 (designated 2026-06-12)

**Owner decision:** the project's baseline anchor moves from `tau_scale_05` control (last-step readout) to
**τ=0.5 + temporal_readout=d1pma + gate_init=−2 (g≈0.12)**. Future experiments (DropExtremeLabel,
stock-expert, gen-gap track, …) compare against THIS anchor's three seed runs.

**Honest framing (binding for future readers):** this is an **anchor CHOICE at parity, NOT an evidence-backed
promotion.** The g0.12 n=3 kill-check (`g012_n3_RESULT.md`, card `time-readout-bonus-20260607` →
settled-CLOSED-KILLED) found paired ΔRankIC vs the old control = **+0.00099 ± 0.00503** (sign-flips
+0.0006/−0.0038/+0.0062), ΔIR incoherent (−0.71/+0.32/−0.10) — RankIC-parity, no portfolio edge. Defensible
rationales for anchoring here anyway: (a) keeps a live-but-dormant attention-over-T path in the architecture as
a platform for future levers (gate is FROZEN at init — mechanism n=3-confirmed — so it costs nothing and stays
inert until deliberately revisited); (b) suggestively lower seed-variance on RankIC (std 0.0033 vs 0.0044, n=3,
NOT significant); (c) flag-controlled and fully reversible (unset `temporal_readout` ⇒ exact old control).

## Canonical settings (the anchor, verbatim)

```bash
export QIB_MODEL_OVERRIDES_JSON='{"time_tau_mlp_out_scale":0.5,"temporal_readout":"d1pma","temporal_readout_gate_init":-2.0,"use_regime_time_embedding":true,"use_regime_factor_gate":true,"router_use_layer_summary":true,"router_mode":"learned"}'
export QIB_TRAINER_OVERRIDES_JSON='{"seed":<S>,"n_epochs":25,"min_epochs":25,"consecutive_k":999999,"train_stop_threshold":null,"checkpoint_metric":"valid_rank_ic","checkpoint_mode":"max","use_tqdm":false}'
```
Sourceable: `scripts/baseline_g012_scale05.env.sh`. Data window: work_flow.py defaults (train 2008–2020-03,
valid=test 2020-07–2022-12, csi300, t+5). Code state: branch `pool-readout-forensics`, base SHA `9b3a2ca` +
the temporal-readout commits (see git log); flags `temporal_readout`/`temporal_readout_init`/
`temporal_readout_gate_init` in `module/quant_moe_model.py` + `module/utils/model_configuration.py`
(defaults unchanged ⇒ no flag = old control, byte-identical).

## Code flow (the anchor's readout path; full per-design flows in `../READOUT_FLOWS.md`)

```
x[B,T,N] → embed → MoE blocks (time⊕factor expert, router) → h = final_norm(h)   [B,T,N,D]  (T=8,N=158,D=64)
 h_time = h.mean(dim=2)                                  [B,T,D]    (mean over N)
 logits = (h_time@tr_q)/√D + tr_b_t                      [B,T] fp32  (tr_q~N(0,0.02), tr_b_t=0 ⇒ ~uniform attn)
 a      = softmax_T(logits)                              [B,T]
 attn   = einsum("bt,btnd->bnd", a, h)                   [B,N,D]
 g      = sigmoid(tr_gate)   tr_gate=−2.0 ⇒ g≈0.119      scalar — FROZEN at init (n=3-confirmed: final 0.124/0.125/0.124)
 z      = (1−g)·h[:,-1] + g·attn                         [B,N,D]    ≈ 88% last-step + 12% learned-attn mixture
 h_pooled, faw = factor_pooling(z)                       [B,D],[B,N]
 score  = head(h_pooled).squeeze                         [B]
diag: tr_gate_g, tr_attn_last_frac, tr_attn_entropy (per-epoch, in train logs)
```

## Anchor data (n=3, the new CTRL set for future compares)

| seed | MLflow run dir | rank_ic | IR_with_cost | MaxDD_with_cost | ppd | final g |
|---|---|---:|---:|---:|---:|---:|
| 42 | `mlruns/716849652326531066/2e8e24fe541a4a31949275b387da2ef2` | 0.08163 | 1.53606 | −0.06634 | 0.00210 | 0.124 |
| 43 | `mlruns/686003645691947084/500a105a10b043068d704fa2e93d60da` | 0.07500 | 2.08932 | −0.08020 | 0.00194 | 0.125 |
| 44 | `mlruns/520724743245902018/79af86fa4f444f218ae3cfc4425e65da` | 0.07857 | 1.62313 | −0.08045 | 0.00348 | 0.124 |

mean rank_ic **0.07840** (seed-std 0.0033) vs old control 0.07741 (std 0.0044). run_setting pattern:
`readout_full_g012_d1pma_seed<S>_scale05`. New CTRL map for compare scripts:

```python
CTRL_RUN = {42: "mlruns/716849652326531066/2e8e24fe541a4a31949275b387da2ef2",
            43: "mlruns/686003645691947084/500a105a10b043068d704fa2e93d60da",
            44: "mlruns/520724743245902018/79af86fa4f444f218ae3cfc4425e65da"}
```

Each run dir durably holds: `artifacts/model` (pickled trained model — loadable, no retrain),
`pred.pkl`/`label.pkl`, `run_conf` + `run_conf_resolved` (post-provenance-fix: records
temporal_readout/init/gate_init correctly), full diagnostic_matrix + regime buckets + portfolio_analysis.
This folder additionally archives: the 3 full training logs, the n=3 master log, and the 3 resolved configs.

## Provenance chain
Old anchor: `tau_scale_05` control `mlruns/867178867749867261/{917808d5,659cb7dc,21a05e19}` (seeds 42/43/44) —
KEPT, still referenced by all settled cards. Decision trail: readout matrix (`readout_matrix_RESULT.md`) →
init-ablation (`init_ablation_RESULT.md`) → g0.12 n=3 kill-check (`g012_n3_RESULT.md`) → this anchor designation.
Ledger rows 76–79; card `shared/falsifiers/time-readout-bonus-20260607.md` (settled-CLOSED-KILLED — the anchor
choice does NOT reopen it).
