# Imagination Skill Contract

Every skill in this package must satisfy this contract. `qx-skill-creator` refuses to scaffold a skill that violates it.

## Required properties

### 1. Generative output

The skill must produce a concrete artifact in `shared/` — a new file or an append to an existing file. Skills whose only output is "PASS/FAIL", "verification", or a console log are rejected.

| Compliant patterns | Non-compliant patterns |
|---|---|
| Writes `shared/analogies/<concept>.md` | Returns "OK" / "FAIL" |
| Appends to `shared/ledger.md` | Runs a checker that prints "config valid" |
| Appends to `shared/hardcore.md` | Lints a file format |
| Appends to `shared/notebook/<date>.md` | Verifies a schema |

### 2. Named lineage

The skill descends from a named thinker, lab, paper, or heuristic — listed in `shared/lineages.md`. If a new skill draws on a thinker not yet in `lineages.md`, the skill creation must include adding a new §N entry.

### 3. Sibling cross-reference

The skill's SKILL.md `description` field must reference at least **two** sibling skills by name. The K-Dense scientific-skills trio (`scientific-brainstorming` → `hypothesis-generation` → `hypogenic`) is the model; sibling routing prevents over-triggering and helps Claude pick the right tool.

### 4. Dialogue posture

The SKILL.md body must include a "Posture" section explicitly defining the agent's stance. Examples:
- "Equal thought partner, not instructor"
- "Facts only, no interpretation"
- "Raw over polished"

Without an explicit posture, the body becomes a checklist by default.

### 5. Bounded description

The `description` field is ≤ 1024 chars (Anthropic hard limit, enforced). Within that, it must:
- State **what** the skill does AND **when** to use it (Anthropic guidance).
- Be "pushy" — combat undertriggering.
- Cross-reference siblings (rule 3).
- Optionally include "Do NOT use when..." for delicate boundaries with siblings.

### 6. Bounded body

The SKILL.md body is **≤ 200 lines** (per Anthropic real-world convention; `pdf/SKILL.md` is ~160 lines). Heavy detail goes in `references/`.

### 7. Repeated-need provenance

A new skill must be justified by **≥ 3 prior manual executions** of the underlying thinking move. "We might want this" is not sufficient. The `qx-skill-creator` flow should ask: "show me three notebook / ledger entries where you did this by hand."

## Anti-patterns (immediate refusal)

- Heavy `MUST/NEVER` blocks in all-caps.
- Outputs labeled "validation", "audit", "compliance check".
- No mapped lineage in `lineages.md`.
- Description doesn't name any sibling skill.
- "Could be useful" without 3 manual-use witnesses.
- SKILL.md body > 200 lines.

## See also

- `anthropic_skill_creator_extracts.md` — extracts from the canonical Anthropic source.
- `shared/lineages.md` — the lineage registry.
