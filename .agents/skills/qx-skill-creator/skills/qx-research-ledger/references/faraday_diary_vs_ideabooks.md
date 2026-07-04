# Faraday's Diary vs Idea Books — the Non-Negotiable Split

The philosophical basis for the strict separation between `shared/ledger.md` (Diary) and `shared/notebook/` (Idea Books).

## Faraday's actual practice

Michael Faraday (1791–1867) kept his lab notes in two parallel systems for 42 years:

### The *Diary* (also called *Researches in Experimental Series*)

- **Sequentially numbered experiments**. The numbering continued unbroken across the entire 42-year career, eventually reaching ~30,000.
- Each entry: date, conditions, materials, procedure, result. **Strictly factual.**
- Append-only — Faraday did not go back and "fix" entries. If a prior result was found wrong, he wrote a new entry referencing the old one.
- Entries that later graduated into a published paper were marked with **vertical lines in the left margin** — a manual cross-reference. (Faraday's Diary survives in 7 volumes; the vertical-line marks are visible in scans.)

### The *Idea Books* (and loose memoranda)

- Speculation, conjecture, "what if" questions, half-formed theories.
- Not numbered, not strictly dated, not factual.
- This is where Faraday let himself be wrong, vague, or wild.

## Why the split mattered

> "I cannot pretend that I am yet wholly convinced; but I do not wish, at this moment, to push the matter so as to retard further inquiry. … I will, therefore, write the present results in my Notebook as a record only, and not at present consider them as facts." — Faraday, in a working note, 1837

The split lets the same person:
- (a) Have **publishable, defensible** records of what actually happened (the Diary).
- (b) Have **freedom to be wrong** in private speculation (the Idea Books).

Mixing these is the most common research-hygiene failure. A notebook that conflates the two becomes neither factually defensible nor speculatively free.

## Operational consequences for this package

- `shared/ledger.md` is the Diary. **Facts only.** Append-only. Sequentially numbered.
- `shared/notebook/*.md` are the Idea Books. **Speculation only.** Append-only by date.
- A `kind=decision` row in the ledger may *cite* a notebook entry, but the row itself only records the *outcome of the decision*, not the reasoning. The reasoning is in the notebook (or the cited artifact).
- When in doubt: if you would publish it in a paper without further qualification, it's Diary. If you would only show it in an informal conversation, it's Idea Book.

## Why not one mixed notebook?

Mixed notebooks fail in two predictable ways:

1. **Speculation contaminates the factual record.** Future-you reads a 6-month-old entry and can't tell whether the IC=0.082 is what was measured or what was hypothesized.
2. **Factual discipline kills speculation.** If the same notebook holds the publishable record, the speculative side is consciously or unconsciously sanitized — robbing future-you of the wild hypotheses that turn out to be right.

The two skills enforce the split mechanically: `qx-research-ledger` refuses to record interpretation; `qx-notebook-the-anomaly` refuses to record bare facts.

## See also

- `ledger_schema.md` — the 5 entry kinds.
- `skills/notebook-the-anomaly/SKILL.md` — the Idea Book skill.
- `shared/lineages.md` §3 Faraday.
