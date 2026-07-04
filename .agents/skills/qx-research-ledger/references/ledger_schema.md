# Ledger Schema — the 5 Entry Kinds

The `shared/ledger.md` file uses a single markdown-table format with the columns `id | kind | key | context`. The `kind` field is one of 5 values; each has required content for the `key` and `context` columns. A 6th kind `correction` exists for fixing prior mistakes without editing in place.

## The 5 (+1) kinds

### `anchor`

A commit blessed as a reproducible baseline.

**Required in `key`:** `sha=<commit-sha>` and the headline metric(s). Example: `sha=3675090 ic_raw=0.0706 rank_ic=0.0786`.
**Required in `context`:** branch name, universe, horizon, seed count, "current" or "superseded" status, one cited artifact path.

Only one row per project can be `current` at a time; when a new anchor is added, the prior `current` row is **not edited** — instead, a new `decision` row supersedes it explicitly.

### `sweep_start`

A sweep dispatched.

**Required in `key`:** `claim_id` (from a falsifier card) plus `n_runs` (seeds × universes × arms).
**Required in `context`:** settings tag, expected completion estimate, cited falsifier card path.

### `sweep_done`

A sweep finished. **This is the row that increments the cumulative trial counter** for future Deflated-Sharpe-style haircut calculations.

**Required in `key`:** `claim_id` plus `status` (complete / partial / failed) and `n_runs_completed`.
**Required in `context`:** list of MLflow run IDs (or external path to manifest), one-line outcome.

### `result`

A measured metric value reported.

**Required in `key`:** `claim_id` plus `metric_name=<value>` for each reported metric.
**Required in `context`:** universe, seed count, statistical test result if applicable, cited falsifier card path.

### `decision`

A judgment made: adopt / reject / defer / classify / amend.

**Required in `key`:** a one-phrase summary of the decision.
**Required in `context`:** the reasoning (short!) and ≥ 1 cited artifact path.

### `correction`

A fix to a prior row (because the ledger is append-only).

**Required in `key`:** `corrects: id=<N>` where N is the row being corrected.
**Required in `context`:** what was wrong with the prior row, what the correct value is.

## Sequencing rules

- `id` is globally sequential. Increment from the last row in the file.
- New date headings group rows; create a `## YYYY-MM-DD` heading when the first entry of a day is written.
- Within a day, rows are append-only and ordered by time of event (not time of writing).

## What the ledger does NOT contain

- Speculation ("I think this means…") — that's `qx-notebook-the-anomaly`.
- Long narratives — keep `context` ≤ 200 chars; put the long story in the cited artifact.
- Subjective interpretation — Diary, not Idea Book.

## See also

- `templates/ledger_entry.md` — the row format.
- `references/faraday_diary_vs_ideabooks.md` — the philosophical basis for the strict fact/speculation split.
