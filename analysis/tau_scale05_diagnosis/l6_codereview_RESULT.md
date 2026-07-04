# L-6 code review — synth fix-list

## Verdict tally
- 8  false_positive
- 16  confirmed_nice_to_have
- 2  confirmed_must_fix_before_gpu

## MUST-FIX before n=3

### [blocker] fp16 0/0 NaN in book weights permanently poisons the detached EMA buffer -> silent, run-killing training death (NOT caught by the isfinite skip-guard)
- loc: module/quant_moe_model.py:724-740 (denom :726, EMA update :735-740); guard at module/model_adapter.py:1733-1735; autocast wrap at module/model_adapter.py:1727
- fix: Apply BOTH fixes. IMPORTANT refinement verified empirically: `denom = pc.abs().sum().clamp_min(1e-12)` in fp16 STILL returns 0.0 (the 1e-12 floor underflows in the clamp comparison too) — so the clamp_min is only effective AFTER the fp32 cast. The fp32 cast is the load-bearing fix; the isfinite guard is the second, source-agnostic safety layer. Replace quant_moe_model.py:724-740 with:\n\n                if self.ir_aux_lambda > 0.0:\n                    # FP32 book/vol math: under amp_fp16 the denom floor underflows to 0 (fp16 min subnormal\n                    # ~6e-8 >> 1e-12), so an all-equal cross-section -> 0/0 -> NaN that permanently poisons the\n                    # EMA buffers (cf. regime_encoder.py:138-145 fp32-under-AMP precedent).\n                    with torch.autocast(device_type=p.device.type, enabled=False):\n                        p32 = p.float(); y32 = y.float()\n                        pc = p32 - p32.mean()\n                        denom = pc.abs().sum().clamp_min(1e-12)   # clamp_min in fp32 (cannot underflow)\n                        w_book = pc / denom\n                        r_book = (w_book * y32).sum()\n                        var = (self.ir_ema_r2.float() - self.ir_ema_r.float() ** 2).clamp_min(0.0)\n                        vol = torch.sqrt(var.detach() + self.ir_aux_var_eps).reshape(())\n                        l_ir = -(r_book / vol)\n                    # Finiteness guard: a bad day can never poison persistent EMA state nor the loss.\n                    if self.training and torch.isfinite(r_book):\n                        ramp = min(1.0, float(self.ir_step.item()) / max(1.0, float(self.ir_aux_ramp_steps)))\n                        lam_eff = self.ir_aux_lambda * ramp\n                        total_loss = total_loss + lam_eff * l_ir\n                        with torch.no_grad():\n                            rd = r_book.detach(); dec = self.ir_aux_ema_decay\n                            self.ir_ema_r.mul_(dec).add_(rd * (1.0 - dec))\n                            self.ir_ema_r2.mul_(dec).add_(rd * rd * (1.0 - dec))\n                            self.ir_step += 1\n\nNotes: (a) The metrics block (lines 756-760) reading float(l_ir...) is harmless if it logs NaN on a skipped day, but for cleanliness you may also gate those metric writes on torch.isfinite(r_book). (b) Default-off behavior (ir_aux_lambda=0) is preserved since the whole block stays guarded by `if self.ir_aux_lambda > 0.0`. (c) After the fix, re-run scripts/test_ir_aux.py and add a regression case: an all-equal fp16 cross-section must yield finite r_book=0 and must NOT mutate ir_ema_r/ir_ema_r2 to NaN.

### [high] Denominator eps (1e-12) is dtype-fragile and asymmetric vs the well-guarded vol eps; only 'works' in the exactly-degenerate fp32 case
- loc: module/quant_moe_model.py:726 (denom eps 1e-12) vs :728-729 (vol eps 1e-6, safe)
- fix: Compute the entire book/Sharpe in fp32 with autocast locally disabled, floor the denom with clamp_min (not +eps), and guard the EMA update against non-finite values. Verified working empirically (degenerate -> r_book=0.0 finite; varied -> correct value). Replace the block at quant_moe_model.py:724-740 with:

    if self.ir_aux_lambda > 0.0:
        with torch.autocast(device_type=p.device.type, enabled=False):
            pf = p.float()
            pc = pf - pf.mean()
            w_book = pc / pc.abs().sum().clamp_min(self.ir_aux_var_eps)   # fp32; clamp can't underflow, no 0/0
            r_book = (w_book * y.float()).sum()                           # fp32, differentiable in p
            var = (self.ir_ema_r2.float() - self.ir_ema_r.float() ** 2).clamp_min(0.0)
            vol = torch.sqrt(var.detach() + self.ir_aux_var_eps)
            l_ir = -(r_book / vol)
        if self.training:
            ramp = min(1.0, float(self.ir_step.item()) / max(1.0, float(self.ir_aux_ramp_steps)))
            lam_eff = self.ir_aux_lambda * ramp
            total_loss = total_loss + lam_eff * l_ir
            with torch.no_grad():
                rd = r_book.detach()
                if torch.isfinite(rd):                                   # belt-and-suspenders: never poison EMA
                    dec = self.ir_aux_ema_decay
                    self.ir_ema_r.mul_(dec).add_(rd * (1.0 - dec))
                    self.ir_ema_r2.mul_(dec).add_(rd * rd * (1.0 - dec))
                    self.ir_step += 1

The autocast(enabled=False) wrapper is the load-bearing change: it stops autocast from re-casting the pc/denom division (and r_book/vol division) back to fp16 where 1e-12 underflows. clamp_min(self.ir_aux_var_eps=1e-6) is rock-solid in fp32. The isfinite guard on the EMA makes a single bad batch non-catastrophic instead of permanently dead. The finding's own proposed fix (clamp_min(eps) computed in fp32 + the fp32 cast) is essentially equivalent and also correct; the explicit autocast-disable just makes the fp32-ness guaranteed rather than dependent on autocast op-policy quirks. Cost is negligible (a few-hundred-element cross-section).

## NICE-TO-HAVE

- [med] Detached vol + eps=1e-6 variance floor => effective gradient gain on the aux silently amplifies as book-return variance shrinks (positive-feedback / drifting effective LR) (module/quant_moe_model.py:728-730)
- [med] w-denominator eps=1e-12 guards the forward VALUE but not the GRADIENT under near-collapsed predictions (module/quant_moe_model.py:726)
- [low] EMA variance: clamp_min(0) present, init warmup sane, causal ordering prevents same-step self-normalization (module/quant_moe_model.py:728-740)
- [nit] Duplicate (upsampled-with-replacement) names: not worse than stated, but they inflate gross and add variance to r_book and the EMA (module/quant_moe_model.py:725-727)
- [nit] Monitoring nit: loss_ir is negative for a good model and is logged even in eval (module/quant_moe_model.py:757-759)
- [nit] _sanity_check_batch seeds the IR EMA once in train mode before the main loop (off-by-one in ramp; NOT leakage) (module/model_adapter.py:642,650,2063; module/quant_moe_model.py:731-740)
- [nit] EMA variance is initialized at scale ~1 (ir_ema_r2=1.0) vs true tiny book-return scale — benign transient, watch the early ramp (module/quant_moe_model.py:128-129,728-734)
- [high] PROVENANCE BLOCKER: ir_aux_* (and time_tau_mlp_out_scale) absent from MODEL_CONFIG_KEYS_FULL -> dropped from run_conf_resolved + MLflow note + paper report (work_flow.py:264-325 (list); 408-415 (filter); 2744-2757, 2790-2797 (consumers))
- [nit] l_ir composes additively without disturbing l_aux (router z-loss) or l_reg (PASS) (quant_moe_model.py:684-689,706-714,722-740,748-749)
- [med] IR metrics ARE logged to MLflow but are NOT captured into the saved train_curve object (quant_moe_model.py:756-760; model_adapter.py:1801-1803,1391-1399,2247,2253-2261)
- [low] Both sweep arms produce an identical MLflow experiment name; arm separation relies entirely on QIB_RUN_SETTING (work_flow.py:460-495 (esp. 471-494))
- [low] valid.sum()<2 path is correctly skipped (no bug) — but the >=2 guard does NOT cover the degenerate-but->=2 collapse (module/quant_moe_model.py:652 (guard), :632 (total_loss=None default); module/model_adapter.py:1730-1732)
- [low] Dedup approximation: uniform-with-replacement upsampling injects variance (not systematic bias) into r_book; marginally worse than MSE but acceptable (module/quant_moe_model.py:725-727; module/dataloader/sampler.py:68-78)
- [med] ramp_steps unit is per-day FORWARDS (microbatches), not optimizer steps — 5x the 'TRAIN steps' intuition; comment is ambiguous (module/quant_moe_model.py:732,740; module/utils/model_configuration.py:120; module/model_adapter.py:144)
- [nit] EMA vol window (~100 days) is sane; cold-start bias is benign/conservative — informational (module/quant_moe_model.py:128-129 (buffer init), :728-729,737-739 (EMA))
- [med] Unit test gives false confidence: covers none of the numerics-risk dimensions (fp16/autocast, degenerate day, EMA poison) (scripts/test_ir_aux.py:13-19 (CPU/fp32 cfg), :22-24 (random inputs))

## SYNTH MEMO

Verified against the live code. Both "blocker"/"high" findings are the same confirmed root cause, the fp32-under-AMP precedent exists (regime_encoder.py:138-145), and the provenance gap is real (ir_aux_* / time_tau_mlp_out_scale absent from MODEL_CONFIG_KEYS_FULL). Here is the fix-list.

---

# Fix-List — L-6 Portfolio-IR aux (l_ir, h-20260627-001) — GATES n=3 GPU launch

Reviewed against live code: `module/quant_moe_model.py:120-130,716-760`, `module/model_adapter.py:1726-1768`, `module/architecture/regime_encoder.py:138-145`, `work_flow.py:198,324`. Verdict: **ONE must-fix**, then launch. The control arm (`ir_aux_lambda=0`) is safe regardless (block is gated at `quant_moe_model.py:724`); the exposure is entirely on the **treatment arm**, which is the whole point of the ablation.

## 1. MUST-FIX-BEFORE-N3 (confirmed blocker — would silently corrupt the treatment arm)

### MF-1 — fp16 0/0 NaN in the book denom permanently poisons the detached EMA buffers → silent run-death
**File:** `module/quant_moe_model.py:724-740` (denom `:726`; unconditional EMA update `:735-740`). Interacts with autocast wrap `module/model_adapter.py:1727` and NaN-loss guard `:1733-1735`.

**Verified mechanism (all confirmed against code, not just the unit test):**
- The entire l_ir block runs inside `with self._autocast_ctx()` (`model_adapter.py:1727`); training default is `precision="amp_fp16"` (`work_flow.py:198` → fp16 autocast on CUDA, `model_adapter.py:202-206`). The intended override JSON does NOT change precision.
- With readout OFF, `p = stock_score` is the fp16 `nn.Linear` head output; `pc = p - p.mean()` stays fp16. At `:726` the division `pc / (pc.abs().sum() + 1e-12)` is re-cast to fp16 by autocast, where `1e-12` underflows to `0` (fp16 min subnormal ≈ 6e-8). On a degenerate day (cross-section collapses to one fp16 value, `sum|pc|=0`) this is `0/0 = NaN`. Identical fp32 inputs give `r_book=0.0` (safe). This regime is realistic in early from-scratch bare-backbone training (the config that already produced the observed 4/1033 e1 grad-skips).
- The EMA update (`:735-740`, `ir_ema_r.mul_().add_()`, `ir_ema_r2...`) runs **unconditionally** (train-only, but no finiteness gate) and executes **inside `forward()` BEFORE** the adapter's NaN guard (`model_adapter.py:1733-1735`), which only does `continue` (skips backward/step) and never rolls back the buffer mutation. So one NaN `r_book` sets `ir_ema_r=NaN, ir_ema_r2=NaN` → `var=NaN` → `vol=NaN` → every subsequent `l_ir=NaN` → every subsequent `total_loss=NaN` → every step silently skipped. The run finishes all 25 epochs frozen at the poison point with only an MLflow `loss=nan` signal — exactly the silent-waste failure this gate exists to catch.
- The ramp gives **zero** protection: `lam_eff=0` at step 0 but `0.0 * NaN = NaN`. The EMA poison is itself λ-independent.

**Concrete fix — apply BOTH layers** (the fp32 cast is load-bearing; `clamp_min(1e-12)` alone still underflows under fp16, so it only works *after* the cast). Mirrors the existing `regime_encoder.py:138-145` fp32-under-AMP precedent. Replace `quant_moe_model.py:724-740`:

```python
if self.ir_aux_lambda > 0.0:
    # FP32 book/vol math: under amp_fp16 the +1e-12 denom floor underflows to 0
    # (fp16 min subnormal ~6e-8) -> all-equal day -> 0/0 -> NaN permanently poisons
    # the detached EMA buffers. (cf. regime_encoder.py:138-145 fp32-under-AMP precedent.)
    with torch.autocast(device_type=p.device.type, enabled=False):
        p32 = p.float(); y32 = y.float()
        pc = p32 - p32.mean()
        w_book = pc / pc.abs().sum().clamp_min(1e-12)        # fp32 clamp_min cannot underflow; no 0/0
        r_book = (w_book * y32).sum()
        var = (self.ir_ema_r2.float() - self.ir_ema_r.float() ** 2).clamp_min(0.0)
        vol = torch.sqrt(var.detach() + self.ir_aux_var_eps).reshape(())
        l_ir = -(r_book / vol)
    if self.training and torch.isfinite(r_book):            # guard: a bad day can NEVER poison EMA/total_loss
        ramp = min(1.0, float(self.ir_step.item()) / max(1.0, float(self.ir_aux_ramp_steps)))
        lam_eff = self.ir_aux_lambda * ramp
        total_loss = total_loss + lam_eff * l_ir
        with torch.no_grad():
            rd = r_book.detach(); dec = self.ir_aux_ema_decay
            self.ir_ema_r.mul_(dec).add_(rd * (1.0 - dec))
            self.ir_ema_r2.mul_(dec).add_(rd * rd * (1.0 - dec))
            self.ir_step += 1
```
Default-off (`ir_aux_lambda=0`) stays byte-identical to anchor. The `isfinite(r_book)` guard MUST sit in-block before the buffer writes — the adapter-level loss check (`model_adapter.py:1733`) is too late because the `mul_/add_` already ran.

**Pre-launch verification (cheap, do not skip):** extend `scripts/test_ir_aux.py` with a regression case that runs the forward under `torch.autocast(device_type, torch.float16)` on (a) an all-equal cross-section and (b) `valid.sum()==1`, asserting `l_ir` finite, `ir_ema_r/ir_ema_r2` stay finite, and `loss is None` respectively. Current 9/9 PASS runs CPU/fp32 on `torch.randn` only — it exercises none of the fp16/degenerate/poison paths and would not have caught this.

> Note: the two separately-filed items ("fp16 0/0 NaN poisons EMA" blocker + "denominator eps 1e-12 is dtype-fragile" high) are the **same root cause**; this single fix closes both.

---

## 2. NICE-TO-HAVE (real, but does not corrupt/leak the n=3 — apply opportunistically, ideally pre-launch since most are zero-risk)

- **NH-1 — Provenance: `ir_aux_*` + `time_tau_mlp_out_scale` dropped from resolved config** (`work_flow.py` `MODEL_CONFIG_KEYS_FULL` ~`:264-325`; filter `:408-415`). High-rated provenance gap but does NOT affect training (model reads the live config object, `quant_moe_model.py:124`). **Strongly recommended pre-launch — 5-line, zero-risk allowlist append** after `:324` (`"readout_stock_attn_ffn",`): add `"ir_aux_lambda"`, `"ir_aux_ramp_steps"`, `"ir_aux_var_eps"`, `"ir_aux_ema_decay"`, `"time_tau_mlp_out_scale"`. Matches the existing 2026-06-07 pool/readout provenance precedent. Without it the ablated knob + anchor-defining τ0.5 are absent from `run_conf_resolved`/MLflow note/paper report (λ still recoverable post-hoc from the `ir_lambda_eff` metric + launch env, so not a blocker).
- **NH-2 — Vol-floor drift / effective-LR creep.** Detached `vol` makes the aux coefficient `lam_eff/vol`; as IR improves, vol falls and the realistic effective coeff drifts ~5-8× from init (catastrophic 80-1000× is NOT realistic — needs vol≈1e-3, impossible with z-scored labels). Cheap hardening via the existing knob: set `ir_aux_var_eps=2.5e-3` in the override (caps coeff ≈ 0.08/0.05). **Pre-register an abort:** kill+rerun if the already-logged `ir_ema_vol` (`:759`) drops below ~0.03 or `ir_lambda_eff` (`:760`) exceeds ~2× nominal. The MF-1 fp32+`clamp_min` rewrite also removes the related near-collapse 1/S gradient-blowup concern.
- **NH-3 — ramp_steps unit is per-day FORWARDS (microbatches), not optimizer steps** (`ir_step` increments per forward `:740`; `grad_accum_steps=5` default → 5× the optimizer-step count). Default `5000`≈1.7 epochs in the *actual* units, which is the intended warmup. Reword the `model_configuration.py:120` comment ("over this many PER-DAY FORWARD steps = microbatches; ≠ optimizer steps") and **confirm any per-run `ir_aux_ramp_steps` override is set in forward/day units (~2960/epoch), not optimizer-step units** (a footgun: 5× faster ramp if misread).
- **NH-4 — IR metrics absent from the persisted `train_curve` artifact** (`model_adapter.py:2247,2253-2261` capture filter excludes `ir_*`/`loss_ir`). They ARE in MLflow per-epoch scalars (`train/loss_ir`, `train/ir_lambda_eff`, …), so the n=3 is fully analyzable. Only patch the capture filter if your readout consumes `train_curve` rather than raw MLflow scalars.
- **NH-5 — Arm separation depends on `QIB_RUN_SETTING`** (`_build_experiment_name` omits `ir_aux_lambda`, `work_flow.py:460-495`). Both arms collapse to the same experiment name unless run_setting differs. The diagnostic runner already sets a distinct setting per entry; just **register treatment/control as two distinct SETTINGS** (e.g. `ir_aux_008` vs `ir_aux_000`) so names differ. λ is also logged as an MLflow param, so arms are recoverable regardless.
- **NH-6 — Monitoring readability:** `loss_ir = -r_book/vol` is **negative for a good model** and is computed (not summed) in eval too. Document "more-negative loss_ir = better"; treat `ir_book_return`/`ir_ema_vol` as the primary monitors.
- **NH-7 — Duplicate (upsample-with-replacement) names not de-duped at the loss site** (`quant_moe_model.py:725-727`; uniform replace=True at `sampler.py:74`). Verified **variance-only, not directional/leakage bias** (multiplicities ⫫ label); matches MSE's treatment; the documented SHOULD-not-MUST approximation. Accept as-is for n=3; optional future de-dup if `ir_ema_vol` looks noisy.
- **NH-8 — `_sanity_check_batch` seeds the EMA once in train mode** (`model_adapter.py:642,650`). INERT for this launch (`debug_sanity_check=False` default, not set by the override). Snapshot/restore the 3 buffers only for future runs that enable it.

## 3. DISMISSED (false-positives — verified "code already handles it"; do NOT spend edits)

- **Gradient sign** — `l_ir=-(r_book/vol)`, vol detached+positive ⇒ descent maximizes book return; long winners/short losers. Correct.
- **`reshape(())` broadcasting** — keeps loss 0-dim; cosmetic/defensive. Removing it would NOT crash `.backward()` (a numel-1 loss backprops fine); only trips the unit-test assert. No bug.
- **Leakage invariants 1/2/5** — EMA updates train-only under `no_grad` (`:731,735-740`); eval excludes `l_ir` from `total_loss` + freezes buffers; `ir_aux_lambda=0` ⇒ byte-identical anchor (`:724`, buffers `persistent=False`). Hold.
- **GATE all-clear (all 5 invariants)** — re-verified at integration level (predict path `head(h_pooled)` independent of l_ir; predict() calls net without labels; valid epoch `train=False`). Hold.
- **No double-counting with MSE** — un-normalized MSE (scale/level, diagonal) vs scale/shift-invariant l_ir (name-coupling); complementary, not redundant.
- **EMA variance `clamp_min(0)` + causal ordering** — fp cancellation can't make var negative; current-day `r_book` never normalizes itself (vol from steps 0..k-1). Correct.
- **EMA cold-start init (var=1)** — conservative high-vol warmup, doubly damped with the ramp; sign-safe. No change (do NOT add bias-correction — it fights the intended warmup).
- **EMA window (~100 days) / decay 0.99** — sane; informational.
- **Config field plumbing PASS** — `QuantMoEConfig` declares+coerces all 4 fields, model reads via `getattr`, override JSON splats unfiltered into the constructor (`model_adapter.py:1357`). Intact.
- **readout-OFF (`temporal_readout=""`) bare backbone PASS** — `""` allow-listed; falls to last-step + factor-pool else-branch; readout/stock-attn skipped; scoring = `head(h_pooled)`. Correct.
- **l_ir composes additively, doesn't disturb l_aux/l_reg PASS** — added last, train-side only, never reads/writes the other terms.
- **`valid.sum()<2` path** — doubly safe (adapter pre-filters; model guards at `:652`, `total_loss` stays None). No change; the real degenerate-but-≥2 collapse is closed by MF-1's in-block isfinite guard.
- **Dedup variance (leakage lane)** — same-day in-batch label, no look-ahead, never touches the prediction path; cannot manufacture a false validation signal. Accepted approximation.

**Bottom line: apply MF-1 (fp32 book/vol math + in-block `isfinite` EMA guard) and add the fp16/degenerate regression test before launching. Recommended same-PR: NH-1 (provenance append) and NH-3 (ramp_steps comment/units check) — both zero-risk. Everything else can follow.**