# Temporal Readout Redesign — Overnight Execution Plan (LOCKED 2026-06-06)

Base = `tau_scale_05` backbones (exp 867178867749867261), seeds 42–47 (n=6), loaded from MLflow (no retrain). csi300, t+5, main_loss=MSE on CSZScoreNorm.

## Invariants (all designs)
1. Collapse factors by **mean over N** (proven IC-optimal), keep D; readout ends at `[B,D]` → `head(D→1)`.
2. **Identity-start by function-class:**
   - softmax-attention readouts (D1-PMA, dual z_T, dual-b z_N) → **gated-residual**, NOT saturated one-hot
     (one-hot `b_t[last]=+20` saturates softmax → off-last gradient ≈ e^-20 ≈ 2e-9 → attention gradient-LOCKED at last-step = mirror of the six-nines uniform-lock → manufactures a FALSE null). Use:
     `pooled = (1-g)·h_time[:,-1] + g·attn_over_T(h_time; q, b_t=0)`, `g=sigmoid(gate)`, `gate init=0 (g=0.5, responsive)`, attention `b_t=0` (uniform, can concentrate OR spread), `q` small-random. Nests last-step (g→0 = baseline safety net); learned `g` reads "how much the temporal pool helps".
   - linear readouts (D3-CI) → **one-hot-last linear init is FINE** (linear has full gradient `∂z/∂W[:,t]=h_time[:,t]` regardless of W; no saturation). Keep `W[:,last]=1, else 0`.
   - dual branch gate (head2 zeroing z_F half) → **fine** (linear head; `∂L/∂head2_zF=z_F·∂L/∂score` grows from 0).
3. **Exact-step0-baseline is NOT required.** Baseline = the loaded ckpt's valid rank_ic (free, exact, = last-step). bonus = trained_design − baseline. Wiring sanity = one-off `gate→−∞` ⇒ pooled==last-step.

## Designs (Step-A screen set)
| id | readout (h:[B,T,N,D] → score[B]) | new params | identity-start |
|---|---|---|---|
| **D6** | `mean_N(h[:,-1])`→LN→head (replace AdaptivePooling) | −16.6k | ≈baseline (pool≈mean) |
| **D1-PMA** | gated: `(1-g)·h_time[:,-1]+g·attn_over_T(h_time)` →LN→head | ~q64+b8+gate1+LN128 | gate=0, b_t=0, q small |
| **D1-dual-a** | `concat[z_T(=D1-PMA), global_mean]`→head2(2D→1) | ~+head 64 | head2 z_F half=0 |
| **D1-dual-b** | `concat[z_T, attn_over_N(mean_T(h)=h_fac; q_N)]`→head2 | ~+q_N64+b_n158 | head2 z_F half=0, q_N small,b_n=0 |
| **D3-CI/D** | per-D linear over T: `einsum('btd,dt->bd', h_time, W_D)` | W_D 64×8=512 | W_D[:,last]=1 |
| **D3-CI/N** | per-N linear over T then mean_N: `einsum('btnd,nt->bnd')→mean_N` | W_N 158×8=1264 | W_N[:,last]=1 |

`attn_over_T(X[B,T,D]; q,b)= softmax_T((X@q)/√D + b) · X → [B,D]`.

## Two-step methodology
- **Step-A (cheap lower-bound SCREEN, frozen backbone):** freeze entire backbone, cache `h_time[B,T,D]` + `h_fac[B,N,D]` over train(strided)+valid via final_norm hook; train ONLY the readout(+head) 8 epochs (Adam) on cached reps; baseline-arm = same-capacity trained head on `h_time[:,-1]`; bonus = design − baseline-arm; n=6. ESCALATION gate (never kill): any design ≥ baseline +0.003 on ≥4/6 seeds → escalate to Step-B; all-tie → escalate best-by-point-estimate. ~1–2 GPU-hr total.
- **Step-B (un-confounded VERDICT, readout + top MoE block):** for the escalated design, build model with the readout (config-flagged), load scale=0.5 weights, freeze all but {last MoE block, final_norm, readout, head}, continue-finetune 8 epochs (identity-start, warm), n=6. PROMOTION gate = HC-6 four-tuple (RankIC +0.003 AND IR_with_cost / MaxDD_with_cost not worse). ~4 GPU-hr.

## Decision rule (pre-registered, falsifier `time-readout-bonus-20260607`)
- Step-A bonus < +0.003 on ALL designs at n=6 → temporal readout shows no frozen-rep headroom (lower bound); still escalate best-by-point-estimate to Step-B.
- Step-B winner bonus < +0.003 at n=6 (HC-6) → **last-step readout is ~optimal** (confound resolved in the falsification direction); temporal readout is NOT a lever; headroom is upstream (blocks/data). 
- Step-B winner ≥ +0.003 with IR/MaxDD not worse → temporal readout IS a lever; promote, then /quant-walk-forward + csi800.
- n≥6 kill-check (τ n=3→n=6 reversal lesson). Dropped: D2 (no new time hyperparam). Deferred: D5 (use_alibi=False → ALiBi confound; revisit via causal-mask).

## Autonomous execution (tonight)
1. Implement `scripts/time_readout_finetune.py` (Step-A) + smoke (D1-PMA on 1 backbone: wiring gate→−∞==baseline; 8ep trains).
2. Driver runs Step-A: 6 designs × seeds 42–47 → aggregate → report.
3. Implement + smoke Step-B; if clean, escalate winner (+D1-PMA) × n=6 → HC-6.
4. Final report `analysis/readout_redesign/RESULT_overnight.md` by morning.
