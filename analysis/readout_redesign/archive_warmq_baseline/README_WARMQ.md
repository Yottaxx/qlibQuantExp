# warm-q baseline — complete archive (h-20260610-002 follow-up #3)

Archived 2026-06-19. This preserves the **warm-q** cross-stock variant as a complete, reproducible
reference: exact settings, per-seed checkpoints, per-run resolved config + code diff, run logs, and
launchers. warm-q is the **reference bar** for the readout cross-stock series (it is the best
cross-stock RankIC obtained so far — but a pathological win; see §3).

## 1. Exact settings (verified from each run's `run_conf_resolved`)

Model = g012 anchor + block-internal stock_expert with de-mean, optimizer = stock_expert params
excluded from weight decay. NO output-norm (true warm-q), NO readout stock block.

```
MODEL  : temporal_readout=d1pma, temporal_readout_gate_init=-2.0, time_tau_mlp_out_scale=0.5,
         use_regime_time_embedding=true, use_regime_factor_gate=true,
         router_use_layer_summary=true, router_mode=learned,
         use_stock_expert=true, stock_expert_demean=true,
         stock_expert_xs_center=false, stock_expert_out_norm=none
TRAINER: stock_expert_no_wd=true, n_epochs=25, min_epochs=25, seed in {42,43},
         checkpoint_metric=valid_rank_ic (max), lr=5e-5, wd=0.01, batch=300, amp_fp16
```

## 2. Results (the bar to beat)

| seed | best valid_rank_ic | test RankIC | portfolio IR (w/cost) | MaxDD (w/cost) | ckpt |
|---|---|---|---|---|---|
| 42 | 0.083165 @ ep16 | 0.0831 | 1.4118 | −0.08328 | `seed42/model.pth` |
| 43 | 0.079006 @ ep14 | 0.0790 (recomputed from pred.pkl) | *backtest crashed* | *recoverable* | `seed43/model.pth` |

vs anchor (g012): seed42 RankIC 0.0816 / IR 1.5361 / MaxDD −0.0663; seed43 RankIC 0.0750 / IR 2.0893.
→ warm-q test RankIC beats anchor BOTH seeds (+0.0015, +0.0040; mean **+0.0028**, same sign).

seed43's portfolio is recoverable WITHOUT retrain: `seed43/pred.pkl` + `label.pkl` → run qlib backtest
(kernels=1 makes it crash-proof now).

## 3. Why it is a REFERENCE, not a solution (the 3 pathologies)

- **P1 RankIC≠IR**: RankIC up but portfolio IR/MaxDD worse (seed42 1.41<1.54, −0.083 vs −0.066). The
  gain is a high-variance side channel, not real ranking edge. Cause: loss-neutral router-gated/parallel
  placement (the optimizer has a zero-cost exit `w_stock→0`).
- **P2 entropy frozen 1.0000**: cross-stock attention never sharpened — it stayed uniform (= cross-
  sectional mean = rank-invariant). Cause: cold query, `‖Wq‖→0` under weight decay pins softmax uniform.
- **P3 V-escape (`stock_expert_norm`→88)**: the de-mean gradient was satisfied by inflating `‖Wv‖‖Wo‖`
  instead of sharpening. (Later confirmed: out-norm-over-D does NOT close it — the escape moves to the
  B/stock axis; `stock_expert_norm` climbed 7→125. That run was killed.)

So warm-q's RankIC gain rides P3, NOT cross-stock attention. The **readout QK-norm series** is the clean
migration that removes all three (no de-mean → no P3; QK-norm+temp → no P2; ungated mainpath → no P1).
See `analysis/stock_expert/warmq_to_readout_qknorm_PLAN.md`.

## 4. Reproduce

```
# from repo root; needs the warm-q code state (see seed*/code_diff.txt for the exact tree at run time)
SEED=42 bash analysis/readout_redesign/archive_warmq_baseline/scripts/smoke_warmq_stockexp.sh
# or both seeds:
bash analysis/readout_redesign/archive_warmq_baseline/scripts/run_warmq_43_44.sh
```

NOTE on code state: the launchers reproduce the *config*. The exact *source* at run time is captured per
run in `seed*/code_diff.txt` + `seed*/code_status.txt` (mlflow snapshot). The repo has since evolved
(added `stock_expert_out_norm`, then the readout `CrossStockBlock`); all are default-off, so re-running
the launchers on the current tree reproduces warm-q faithfully (the warm-q flags don't touch the new
code paths). `code_state/git_HEAD_at_archive.txt` records HEAD at archive time.

## 5. Files
- `seed{42,43}/model.pth` — restored best checkpoint (loadable, no retrain).
- `seed{42,43}/run_conf_resolved` (pickle), `run_conf`, `best_checkpoint_info`, `code_diff.txt`,
  `code_status.txt`, `run.log`.
- `seed43/pred.pkl`, `label.pkl` — for recovering the crashed portfolio backtest.
- `scripts/smoke_warmq_stockexp.sh`, `run_warmq_43_44.sh` — launchers.
