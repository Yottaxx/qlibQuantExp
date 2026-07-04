# Falsifier Card — Temporal Readout Bonus (last-step vs learned-over-T)

**claim_id:** `time-readout-bonus-20260607`
**filed_at:** 2026-06-06 (pre-registered before execution)
**parent_base:** `tau_scale_05` backbones (exp 867178867749867261), seeds 42–47
**status:** open — executing (overnight autonomous)
**follows:** `pool-readout-gate-20260606` (factor pool settled-dead) → headroom redirected to the TIME-axis readout.

## The claim (stripped)
Replacing the last-step temporal readout `h[:,-1]` with a learned-over-T readout (D1-PMA gated-attention / dual-path / D3-CI), continue-finetuned with identity-start, raises CSI300 valid `daily_rank_ic` by **≥ +0.003** over the last-step baseline on **≥4/6 seeds**, on the scale=0.5 base.

## The prohibition (what it FORBIDS)
| | Failure meaning |
|---|---|
| **P1.** If NO design (Step-A frozen screen, n=6) beats the capacity-matched last-step baseline-arm by +0.003, the temporal readout shows no frozen-rep headroom (a confound-aware LOWER BOUND — not a verdict; still escalate best-by-point-estimate to Step-B). |
| **P2.** If the escalated design under Step-B (readout + top-MoE-block continue-finetune, n=6, HC-6) does NOT beat the last-step baseline by +0.003 RankIC (with IR_with_cost / MaxDD_with_cost not worse), then **last-step readout is ~optimal** — the bidirectional in-block attention already integrates the window, the t+5 label is closest to the last step, and temporal readout is NOT a RankIC lever. Headroom is upstream (blocks/data/regime), not readout. Confound resolved in the falsification direction. |
| **P3.** If a design beats baseline ONLY at Step-A (frozen) but NOT Step-B (finetune), the Step-A gain was reading frozen-rep noise; trust Step-B. |
| **P4.** If the learned gate `g` (D1-PMA) converges to ~0 (or attention mass stays >0.9 on the last step) AND bonus<+0.003, that is direct mechanistic confirmation last-step is optimal. |

## Arms / identity-start (binding)
softmax readouts use **gated-residual** init (NOT saturated one-hot: `b_t[last]=+20` saturates softmax → off-last grad ≈2e-9 → gradient-lock → false null). `pooled=(1-g)·h_time[:,-1]+g·attn_over_T`, gate=0 (g=0.5, responsive), attention b_t=0, q small. Linear (D3) keeps one-hot-last. Designs: D6 / D1-PMA / D1-dual-a / D1-dual-b / D3-CI-D / D3-CI-N (see `analysis/readout_redesign/PLAN_overnight.md`). Baseline = loaded ckpt last-step rank_ic (exact, free). bonus = trained − baseline.

## Ex-ante metric (binding)
universe csi300 (csi800 corroboration only on a Step-B promotion); horizon t+5; metric daily_rank_ic; base scale=0.5 seeds 42-47 (n=6); Step-A gate +0.003 vs capacity-matched last-step arm; Step-B promotion HC-6 four-tuple; identity-start; 8 epochs; n≥6 kill-check (τ n=3→n=6 lesson). Forensic/exploration, not a deflated-Sharpe promotion until Step-B+walk-forward.

## Anti-rescue
No metric swap (RankIC, not IC/IR for the screen gate); no seed drop (all 6); no epoch surgery (8 fixed); no init change to saturated one-hot to force a null; Step-A is a screen NOT a verdict (cannot claim "settled" on the frozen lower bound alone).

## Settlement
On completion append `## Settlement` + ledger result/decision; route P1/P2/P3/P4. Impl `scripts/time_readout_finetune.py`; report `analysis/readout_redesign/RESULT_overnight.md`.

---

## Settlement — `settled-P1+P2-fire (scoped)` (2026-06-07)

Overnight autonomous run complete (master serial, scale=0.5). Artifacts: `analysis/readout_redesign/RESULT_overnight.md` (+ `stepA/`, `stepB/`).

**P1 FIRES (Step-A frozen screen, n=6, clean):** NO design beats the capacity-matched frozen last-step baseline on a seed-majority. Per-seed bonus means: d1pma −0.0018, duala −0.0034, dualb −0.0056, **d3cid −0.0006 (best, 1/6 pass)**. d3cid's seed-42 +0.0054 did NOT replicate (−0.006 @45, −0.005 @47) — a textbook n=1→n=6 reversal (τ-episode trap). Gated-residual identity-start validated (d1pma gate g=0.49, non-saturated; the +20 one-hot would have gradient-locked).

**P2 FIRES, but SCOPED (Step-B continue-finetune, n=3):** d3cid ≡ control to ~1e-4 (diff +0.00000), so the temporal readout adds nothing. TWO mechanisms: (a) continue-finetuning the converged model HURTS valid every epoch (0.0811→0.078, plateaus) for BOTH arms → overfits train = the gen-gap (durable finding #2 re-confirmed); best=step0. (b) tr_A stayed at one-hot-last (W_last_frac 0.95) → NO gradient pull off last-step (the last-step-trained converged model is a local optimum). **CAVEAT (binding):** the finetune OVERFITS rather than EXPLORES → does NOT fully exclude a FROM-SCRATCH d3cid (backbone co-adapting from init). That gold test was not run (cost).

**Decision.** No temporal-readout RankIC headroom under frozen-probe(n=6) + converged-finetune. With the settled-dead factor pool, **the READOUT LAYER (factor + temporal axes) is ~optimal as-is and is NOT the RankIC lever.** Step-B re-demonstrated the binding wall = the gen-gap (finetune hurts valid). **Route OFF readout architecture → onto the gen-gap track** (sampler-dedup / EMA-rampup — the honest untested levers per finding #2). The one residual to definitively close the temporal axis: a from-scratch d3cid (Tier-B-gold), only if the gen-gap track stalls. Identity-start mechanics (temporal-before-pool, gated-residual) are reusable. csi300/2020-22 scope; seed-robust not regime-robust.

*Status: `settled-P1+P2-fire (scoped; from-scratch d3cid untested)`. Do not edit pre-registered content above.*

---

## Settlement addendum — from-scratch + init-ablation: **CLOSED, unscoped** (2026-06-12)

The two residuals are now tested. Scale=0.5, 25ep, ckpt=valid_rank_ic, seed 42 (n=1 screen), control RankIC 0.08107.

**A. From-scratch matrix (the Tier-B-gold residual): NULL.** Six designs trained from scratch with identity-start
(one-hot-last linears; gated-residual attentions): d3cid −0.0003, d3cin −0.0017, d3mix +0.0002 (best, noise),
d1pma −0.0034, duala −0.0060, dualb −0.0093 ΔRankIC vs control; ALL worse on IR_with_cost AND MaxDD. d3mix
*activated* its mixing capacity (collapse_last_frac→0.878, timemix/chanmix gains 0.13/0.19) for zero RankIC —
mechanism≠benefit replayed. The P2 caveat ("finetune overfits-not-explores") is retired: from-scratch DOES explore
(tr_W_last_frac drifts 1.0→0.906) and still lands null.

**B. Init-ablation (owner's basin-escape hypothesis): FALSIFIED — the trap runs the OTHER way.** Paired A/B with
byte-identical backbones (separate-RNG init), uniform_mean (`1/T+0.02·randn`, sum-preserving — NOT Kaiming, which
is zero-mean/DC-destroying on a T-aggregation) vs onehot_last: **uniform CLEARLY worse on all 3 linears**
(d3cid −0.0137, d3cin −0.0131, d3mix −0.0088), and the temporal profile **never travels back** (last_frac starts
0.125, ends 0.134–0.147; best_ep=16, not budget-censored). One-hot-last does NOT trap; the spread/uniform basin
is itself the trap. Last-step is the attractor.

**C. d1pma gate-init sweep: the gate is FROZEN at init (mechanistic finding).** Final tr_gate_g ≈ init in all
three runs (0.12→0.124, 0.5→0.503, 0.88→0.874) — the scalar sigmoid gate receives ~no gradient. RankIC: g0.12
0.08163 (≈control, +0.0006), g0.5 0.07765, g0.88 0.07848. Pre-registered clause (b) fired (+0.0040 ≥ +0.003 over
g0.5): the g=0.5 start FORCED a permanent 50% attention mixture, explaining the attentions' matrix deficit. Honest
re-read: with a last-step-leaning gate, attention ≈ control — it stops hurting but adds NOTHING (and the control
is free). Engineering rule: a scalar sigmoid gate keeps whatever you init ⇒ never init such a gate at 0.5 expecting
training to choose; the attentions' original deficit was an init artifact, their best case is parity.

**Decision (final).** The temporal-readout axis is closed in every regime tested: frozen screen (n=6),
continue-finetune (n=3), **from-scratch (6 designs)**, **init-ablation (paired A/B ×3)**, **gate sweep (3 points)**
— all converge on **last-step `h[:,-1]` ~optimal; the readout layer is NOT a RankIC lever**. No promotion candidate
exists (d1pma-g0.12's +0.0006 is n=1 noise with mechanism "approximate the control"). Route stays OFF readout →
gen-gap track (sampler-dedup / EMA-rampup) + the 2026-06-10 cards (DropExtremeLabel, stock-expert).
Scope: csi300/2020-22 single-split, seed42 for A/B (paired design mitigates), n=6 only on the frozen screen.

Artifacts: `analysis/readout_redesign/readout_matrix_RESULT.md`, `init_ablation_RESULT.md`, `READOUT_FLOWS.md`;
impl `scripts/run_readout_matrix_scale05.sh`, `run_init_ablation_scale05.sh`, `compare_init_ablation.py`,
`verify_temporal_readout.py`; flags `temporal_readout`, `temporal_readout_init`, `temporal_readout_gate_init`
(defaults unchanged).

*Status: `settled-CLOSED (P2 unscoped; init-trap falsified; gate-frozen mechanism logged)`.*

---

## Final addendum — g0.12 n=3 kill-check: **KILL** (2026-06-12 evening)

Owner-requested fresh-seed check of the one above-control point (g0.12 d1pma seed42 +0.0006). Seeds 43/44 added,
paired vs control: ΔRankIC **+0.0006 / −0.0038 / +0.0062 → mean +0.00099 ± 0.00503** (n=3, 1/3 ≥ +0.003);
portfolio incoherent across seeds (ΔIR −0.71 / +0.32 / −0.10). Sign flips seed-to-seed = sampling noise around
parity — the exact n=1→n=3 mirage pattern of the τ episode, caught by the same kill-check discipline.
Gate frozen at init in all runs (final g 0.124/0.125/0.124 from 0.119) — mechanism now n=3-confirmed.
**Pre-registered kill line fires (mean < +0.003): attention readout is parity-at-best under its best init;
no init flavor rescues it. Last-step control stands. The readout axis is closed with no open residuals.**
Report `analysis/readout_redesign/g012_n3_RESULT.md`; impl `scripts/run_g012_n3.sh`+`compare_g012_n3.py`.

*Final status: `settled-CLOSED-KILLED (all residuals exhausted: frozen n=6 / finetune n=3 / from-scratch ×6 /
init A/B ×3 / gate sweep ×3 / g0.12 fresh-seed n=3)`.*
