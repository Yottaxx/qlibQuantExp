---
name: quant-hypothesis
description: Turn a quant research idea into a falsifiable hypothesis card with budget and promotion criteria. Invoke before any GPU spend when the user proposes a change to RST-MoE (architecture, loss, feature, training tweak), or says things like "what if we tried X", "let's add Y", "I want to try Z". Writes a structured card to experiments.md and a row to experiments_ledger.jsonl, and refuses near-duplicates of recently-decided cards.
---

# /quant-hypothesis

Entry point of the iteration loop. Every change to the RST-MoE system starts here, with a falsifiable claim and a budget — never with "let me just try it."

## When to use

- User proposes a change ("try predictive coding", "swap MSE for ListMLE", "add news embedding").
- User asks "what should we try next?" — generate one or more cards.
- A failure-mode loop-back from a downstream skill.

Do NOT use for trivial bug fixes, refactoring, or formatting changes.

## What to read first

1. `experiments.md` — current experiment log style.
2. `motivation_novelty.md` — the project's research thesis (RST-MoE = regime-conditioned axis-specialized MoE).
3. `DESIGN.md` — current architecture and known caveats.
4. `baselines.md` — what SOTA already does (to avoid reinventing).
5. `kdd_experiments_plan.md` — promotion thresholds already in use.
6. `experiments_ledger.jsonl` — last 30 days of cards (via `module.utils.runlog.iter_ledger`).

## Algorithm

1. Restate the user's idea in one falsifiable sentence ("X improves rank_ic on CSI300 t+5").
2. Identify the **config delta** the idea implies (the smallest set of `run_matrix.yaml` overrides that would test it). If the idea cannot be expressed as a config delta, ask the user to refine it.
3. Compute `config_hash` of the delta and call `runlog.find_duplicate_card(config=delta, since_days=60)`.
   - If a duplicate row with verdict in {`promote`, `reject`} is found within 60 days: surface it to the user, name the prior outcome, and ask whether anything material has changed. Only proceed if the user explicitly confirms.
4. Issue a `card_id` via `runlog.new_card_id()`.
5. Render the card using the template below and append to `experiments.md`.
6. Append a ledger row:
   ```python
   from module.utils.runlog import append_ledger
   append_ledger(
       skill="quant-hypothesis",
       stage="hypothesis",
       card_id=card_id,
       hypothesis=one_line_claim,
       config=config_delta,
       verdict="info",
       notes="<falsifier in one line>",
   )
   ```
7. Print the `card_id` and the next-step suggestion (`/quant-leakage-audit` then `/quant-minimal-repro`).

## Card template

```markdown
## {card_id}: {one-line claim}

- **Mechanism**: why we expect this to help (1–3 sentences, cite the design lens — Kaiming/LeCun/Hinton/Ilya/Jeff Dean or a quant-firm rationale)
- **Falsifier**: a specific result that would disprove the claim (e.g., "rank_ic_delta < 0 on ≥3/5 seeds, or gate collapses with time_ratio outside [0.3, 0.7]")
- **Config delta**: the minimal `run_matrix.yaml` override set
- **Minimal test (proxy)**: epochs, universe slice, seeds for `/quant-minimal-repro`
- **Expected delta vs current best**: rank_ic from 0.0786 → ?
- **Budget**: GPU-hours total (proxy + ablation + walk-forward)
- **Promotion criteria**: e.g. "rank_ic_delta ≥ +0.6%, ≥3/5 seeds above baseline, regime-bucket spread non-collapsed"
- **Risk / failure mode**: what could go wrong (regime collapse, gate collapse, overfit, leakage); each must have a corresponding diagnostic in `/quant-regime-diagnose`
```

## Output contract

- Appended block in `experiments.md` matching the template.
- One row in `experiments_ledger.jsonl` with `stage="hypothesis"`.
- Stdout: `card_id` and suggested next skill.

## Failure modes / loop-back

- **Idea not expressible as config delta** → ask user; do not write a card.
- **Duplicate of recent reject** → require explicit user override.
- **Falsifier missing or not specific** → push back; cards without falsifiers are not allowed in the ledger.
