---
name: qx-notebook-the-anomaly
description: Use daily-or-weekly when there is a SURPRISE in data, training, diagnostics, or literature — anything that does not fit the current model. Anomalies not written down within 48 hours are usually lost. Records a dated narrative entry; what was expected, what happened, three candidate explanations, what to try next. Vertical-line markup when an entry graduates into a paper or proposal (Faraday convention). Keep this empirical-speculation notebook STRICTLY separate from factual records; tracked facts (commit/sweep/IC) go to qx-research-ledger. For testable formulations use qx-forge-a-sharp-falsifier. For cross-field hypotheses use qx-transpose-from-another-field. Triggers; "huh, that's weird", "didn't expect Y", "IC dropped in 2018Q3", any unexpected result.
license: MIT
---

# Notebook the Anomaly

## Overview

Anomalies are research's most perishable resource. Within 48 hours of being noticed, most of them are forgotten or rationalized. This skill captures one anomaly as a dated entry in `shared/notebook/YYYY-MM-DD.md` — raw, fast, ugly is fine.

This is Faraday's **Idea Book**, not his Diary. Speculation lives here. Facts live in `shared/ledger.md` via `qx-research-ledger`. Confusing them is the most common research-hygiene failure; the split is non-negotiable.

## Posture

- Raw over polished. A sloppy entry that exists beats a beautiful entry that was never written.
- Don't resolve the anomaly while writing it. The point is to *preserve* it for later examination.
- Three candidate explanations, not one. The first one that pops to mind is rarely the right one; the third candidate is where the value usually is.
- Mark entries with a vertical-line `|` in the margin when they later graduate into a falsifier card or paper section — Faraday's own convention. Searchable cross-reference.

## Quick Reference

A notebook entry is three short blocks plus a one-line next step:

| Block | Content |
|---|---|
| **Observed** | What I saw — concrete numbers, dates, plots, file paths. No interpretation. |
| **Expected** | What I would have predicted before seeing this. Make the prior explicit. |
| **Three explanations** | At least three candidate causes; the third must be non-obvious. |
| **Next step** | One concrete action — even if it's just "check X in dataset Y". |

## Dialogue flow

When the user surfaces something that did not fit ("router collapsed to time-only in epoch 12", "RankIC fell off a cliff in 2018Q3", "expert cosine sim is 0.97"), do this:

1. **Locate the file.** Today's notebook is `shared/notebook/YYYY-MM-DD.md`. Append, never edit. If the file does not exist, create it with a top-line date header.
2. **Insist on the three-block structure.** Resist the user's tendency to leap to one explanation. If they offer one, ask: "and if that's wrong, what else could it be?"
3. **Keep the entry short.** 100–300 words. The notebook's value is in *quantity over depth* — a hundred small entries beat ten polished ones.
4. **Cross-reference.** If the anomaly relates to an existing falsifier or analogy, link it: `Related: shared/falsifiers/A1-temporal-readout.md`.
5. **Suggest the right next skill.** Surprise that *might* admit a falsifier? → suggest `qx-forge-a-sharp-falsifier`. Surprise that smells like another field's pattern? → suggest `qx-transpose-from-another-field`. Surprise that's just data? → suggest a `qx-research-ledger` append.

## Output

Append to `shared/notebook/YYYY-MM-DD.md` using `templates/anomaly_entry.md`. Multiple entries per file are encouraged — they form a daily log.

## What DOESN'T go here

- "Sweep S1 finished, RankIC = 0.082" — that's a fact, send to `qx-research-ledger`.
- "I decided to demote A5 to Tier-2" — that's a decision, send to `qx-research-ledger` with `kind=decision`.
- "Maxwell's electromagnetism is like our regime model" — that's an analogy, send to `qx-transpose-from-another-field`.
- "I want to test that A1 improves RankIC by 0.003" — that's a falsifier, send to `qx-forge-a-sharp-falsifier`.

The notebook is *for surprises that do not yet have a structured home*.

## Posture line

> "Keep notes of all that occurs to you, however trifling it may at the time appear; not one observation in a thousand may be useful, but the loss of that one is what we cannot afford." — Faraday's working principle, paraphrased from his correspondence.

## See also

- `qx-research-ledger` — the Diary half of Faraday's notebook system. Facts and decisions.
- `qx-forge-a-sharp-falsifier` — when an anomaly is sharp enough to test, this is where it gets a binding prediction.
- `qx-transpose-from-another-field` — when an anomaly smells like a known pattern from another science, this is where the analogy gets built.
- `shared/lineages.md` §3 Faraday (Idea Books).
