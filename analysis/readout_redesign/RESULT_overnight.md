# Time-axis Readout Screen (Step-A, frozen backbone, scale=0.5) — n=6

**Lower-bound SCREEN** (backbone frozen, reps NOT reshaped). Per the confound, a NULL here only escalates the best to Step-B (readout+top-block finetune); it is not a verdict. Gate = bonus ≥ +0.003 vs capacity-matched last-step baseline.

- model valid rank_ic (ref): 0.07745 ± 0.00330
- baseline (trained last-step head on frozen reps): 0.07481 ± 0.00400

| design | valid rank_ic (mean±std) | **bonus vs baseline** | seeds bonus≥+0.003 | per-seed bonus |
|---|---:|---:|---:|---|
| d1pma | 0.07306 ± 0.00327 | **-0.00175 ± 0.00456** | 1/6 | 42:-0.0002, 43:+0.0005, 44:+0.0041, 45:-0.0006, 46:-0.0061, 47:-0.0082 |
| duala | 0.07142 ± 0.00625 | **-0.00339 ± 0.00656** | 1/6 | 42:-0.0041, 43:+0.0068, 44:-0.0016, 45:-0.0008, 46:-0.0087, 47:-0.0119 |
| dualb | 0.06924 ± 0.00641 | **-0.00556 ± 0.00370** | 0/6 | 42:-0.0018, 43:-0.0012, 44:-0.0105, 45:-0.0054, 46:-0.0057, 47:-0.0088 |
| d3cid | 0.07416 ± 0.00563 | **-0.00064 ± 0.00430** | 1/6 | 42:+0.0054, 43:+0.0016, 44:+0.0012, 45:-0.0060, 46:-0.0009, 47:-0.0050 |

- **d1pma learned gate g** (≈0 ⇒ readout falls back to last-step): 0.491 ± 0.007

## Screen verdict
- **No design clears +0.003 on seed-majority (frozen lower bound).** Per the never-kill rule, escalate the best-by-point-estimate (**d3cid**, bonus -0.00064) to Step-B before any 'last-step optimal' claim. A Step-B null would then settle P2.
- Best-by-point-estimate: **d3cid** (-0.00064)

_Card: time-readout-bonus-20260607. Next: Step-B continue-finetune of the escalated design(s), n=6, HC-6 four-tuple._

---

## Step-B — readout + top-MoE-block continue-finetune (UN-confounded, n=3, scale=0.5)

Paired arms per seed: **control** = last-step readout + top-block finetuned; **d3cid** = per-channel-D linear-over-T readout + top-block finetuned. The readout's true contribution = **d3cid − control** (isolates the readout from the finetuning-the-top effect). baseline = loaded model step0.

| seed | model(step0) | control best | d3cid best | **d3cid − control** | d3cid W_last_frac |
|---|---:|---:|---:|---:|---:|
| 42 | 0.08106 | 0.08106 | 0.08106 | **-0.00000** | 0.957 |
| 43 | 0.07881 | 0.07881 | 0.07881 | **+0.00000** | 0.953 |
| 44 | 0.07235 | 0.07235 | 0.07235 | **+0.00000** | 0.936 |

- **d3cid − control: +0.00000 ± 0.00000** (n=3); seeds with diff≥+0.003: 0/3
- control gain vs step0 (top-block finetune alone): +0.00000; d3cid gain vs step0: +0.00000
- d3cid learned W last-step mass-fraction: 0.949 (≈1 ⇒ stays at last-step = no temporal use; <<1 ⇒ uses earlier steps)

### Verdict (Step-B, un-confounded)
- **d3cid does NOT beat the finetuned last-step control by +0.003 (diff +0.00000, n=3).** Even with the top block reshaping under an all-T readout gradient, the temporal readout is ~at parity → **last-step readout is ~optimal** (P2 of time-readout-bonus-20260607); headroom is upstream (blocks/data/regime), not the temporal readout. CAVEAT n=3.

---

## Lead analysis (honest reading — supersedes the auto-verdicts above)

**Step-A (frozen screen, n=6): NULL.** No design beats the capacity-matched frozen last-step baseline on a seed-majority. d3cid's seed-42 +0.0054 did NOT replicate (−0.006 @45, −0.005 @47; mean −0.0006) — a textbook n=1→n=6 reversal (same trap as the τ episode). The gated-residual fix worked (d1pma gate g=0.49, non-saturated) — the softmax designs simply found no gain.

**Step-B (continue-finetune, n=3): two clean facts, one caveat.**
1. Continue-finetuning the converged model on train HURTS valid every epoch (0.0811→0.078, plateaus) for BOTH control and d3cid — i.e. it overfits (the gen-gap). best=step0. **This re-confirms durable finding #2: the gen-gap is the binding wall; more training on train does not help valid.**
2. The temporal weighting tr_A stayed at one-hot-last (W_last_frac 0.95; W_per_t last≈1.0, others≈0.005) → d3cid ≡ control to ~1e-4. The readout had NO gradient pull to use earlier steps — the last-step-trained, converged model sits at a local optimum where moving tr_A off last-step only raises the loss.
   CAVEAT: because the finetune OVERFITS rather than EXPLORES, this is not a clean exploration of the temporal readout. It does NOT fully exclude that a FROM-SCRATCH d3cid model (backbone co-adapting temporal reps with an all-T readout from init) would find value. That gold test (~full training × n) was not run.

**Verdict (scoped):** No temporal-readout RankIC headroom under (a) frozen probe at n=6 and (b) continue-finetune of the converged model. Together with the settled-dead factor pool (pool-readout-gate-20260606), **the readout layer — both the factor axis AND the temporal axis — is ~optimal as-is and is NOT the RankIC lever.** The binding wall is the gen-gap (finding #2), which Step-B just re-demonstrated. Headroom is upstream: sampler/EMA/data-side (the only honest untested gen-gap levers), or a from-scratch d3cid if temporal readout is to be definitively excluded.

**Routing:** settle time-readout-bonus-20260607 as P1+P2-fire (scoped: from-scratch untested); redirect to the gen-gap track (sampler-dedup / EMA-rampup — the levers finding #2 left open). Do NOT spend more on readout architecture.
