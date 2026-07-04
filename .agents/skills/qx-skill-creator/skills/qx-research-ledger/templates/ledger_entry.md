| {{id}} | `{{kind}}` | {{key — concise facts: sha / claim_id / metric=value}} | {{context — ≤ 200 chars; must cite ≥ 1 file path if kind=decision/result/sweep_*}} |

---

*Template for a single row. Append to the appropriate `## YYYY-MM-DD` section of `shared/ledger.md`.*

**The 5 kinds, in order of typical sequence:**

1. `anchor` — when a baseline is frozen
2. `sweep_start` — when a sweep is dispatched (cites a falsifier card)
3. `sweep_done` — when the sweep finishes (increments cumulative trial count)
4. `result` — when metrics are computed and decision-rule applied
5. `decision` — when an outcome is judged (adopt/reject/defer/amend)

Plus `correction` — when a prior row was wrong (cites the original id).

See `references/ledger_schema.md` for the required content in each kind.
