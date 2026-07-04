---
name: qx-skill-creator
description: Use when adding a new skill to the qlibQuantExp-skills package, or revising an existing one. Enforces the package's imagination-over-compliance ethos; every new skill must name its lineage (which thinker / lab / paper / heuristic), define a GENERATIVE output (artifact under shared/), specify its dialogue posture, and cross-reference at least two siblings. Refuses to scaffold a skill whose body is a checklist or whose output is a "validation". Every package mutation is logged as a kind=decision row via qx-research-ledger; if the new skill would change the project's identity rather than its tooling, classify it through qx-protect-or-relax-the-hard-core first. Triggers; "add a new skill", "I keep doing X — should it be a skill?", "this thinking move keeps repeating". The only authorized path to mutate this package; manual edits to MANIFEST.yaml or shared/lineages.md are forbidden.
license: MIT
---

# Skill Creator — Meta-skill for this Package

## Overview

This skill scaffolds new skills, revises existing ones, and ensures the package does not drift into compliance machinery. It is descended from Anthropic's official `skill-creator` and ARIS's `meta-optimize`, adapted with this package's specific discipline: **imagination over compliance, lineage required, sibling cross-reference required**.

If you find yourself writing a SKILL.md by hand without going through this skill, stop. The conventions are easy to break by accident; this skill exists to make them easy to keep.

## Posture

- A new skill must earn its place. "We could add a skill that…" is not a reason. **Repeated manual use (3+ cycles)** is the threshold.
- Refuse to scaffold a skill whose output is a checklist, a validator pass/fail, or a "verification". This package's outputs are *generative artifacts* — new analogies, new hypotheses, new entries to a manifesto. If you can't name what the new skill *produces* in concrete file terms, you don't have a skill yet.
- Every skill names its lineage. If the proposed skill doesn't draw on a named thinker / lab / paper, find one or shelve the idea.

## Quick Reference — the 6 phases

| Phase | Question | Output |
|---|---|---|
| 1. Identify the repeated move | What thinking move has been done by hand ≥ 3 times? Describe it as a verb-phrase. | one sentence |
| 2. Name the lineage | Which thinker, lab, paper, or heuristic does this descend from? | a citation |
| 3. Specify the generative artifact | What concrete file (`shared/<dir>/<name>.md`?) does this skill produce? | a file-path pattern |
| 4. Draft the description | Pushy, sibling-aware, ≤ 1024 chars, "what AND when" | the description string |
| 5. Scaffold from template | Copy `templates/new_skill/` into `skills/<name>/`; fill in description and posture | new directory created |
| 6. Seed a demo entry | Write one example artifact to demonstrate format | seed file in `shared/` |
| 7. Update package | Append to `shared/lineages.md`, update `MANIFEST.yaml`, append `decision` row to `shared/ledger.md` | three append-only updates |
| 8. Symlink | Create `~/.claude/skills/qx-<name>/` symlink | one symlink |

## Dialogue flow

**Phase 1 — Identify the repeated move.** The user said "I keep doing X". Get specific: which 3+ recent sessions? What was the input, what was the output? Write the move as a verb-phrase ("transpose a mechanism from another field", "split a claim into its prohibition"). If you can't name the move sharply, it's not ready to be a skill.

**Phase 2 — Name the lineage.** Open `shared/lineages.md` and check: is there an existing thinker that already covers this move? If yes, the new skill is a sibling of an existing one, not a new branch. If no, identify the thinker, lab, or paper this draws from and prepare a new §N entry. **Without a lineage, do not proceed.**

**Phase 3 — Specify the generative artifact.** What does this skill *write*? Existing patterns:
- A new file per use (`shared/analogies/<concept>.md`, `shared/falsifiers/<claim>.md`)
- An append to a single file (`shared/hardcore.md`, `shared/ledger.md`)
- An append to a dated file (`shared/notebook/YYYY-MM-DD.md`)

A skill that doesn't write to `shared/` is suspicious — it may be a checklist masquerading as a thinking tool. Push back.

**Phase 4 — Draft the description.** Use the Anthropic pattern observed in real SKILL.md files:
- ≤ 1024 chars (validated automatically).
- States **what** AND **when** to use it.
- Pushy (combat undertriggering) but bounded ("for X use Y" cross-references to siblings).
- Triggers field lists the actual user phrases that should fire it.
- Includes "Do NOT use when..." if the boundary with siblings is delicate.

Worked examples in `references/anthropic_skill_creator_extracts.md`.

**Phase 5 — Scaffold from template.** Copy `templates/new_skill/` to `skills/<name>/`. Fill in `SKILL.md` from the description in Phase 4. Create empty `references/` and `templates/` subdirs. The body of `SKILL.md` should follow the existing pattern: Overview → Posture → Quick Reference → Dialogue flow → Output → Posture line → See also.

**Phase 6 — Seed a demo entry.** Generate one example of the new skill's output and write it to the corresponding `shared/` location. This makes the format concrete for future invocations. Without a demo, future-Claude has to guess what the artifact should look like.

**Phase 7 — Update package metadata.**
- Append a §N to `shared/lineages.md` (the new thinker's entry, or a sub-entry under an existing one).
- Update `MANIFEST.yaml` to register the new skill name, version 0.1, today's date.
- Append a `decision` row to `shared/ledger.md` citing what was added and why.

**Phase 8 — Symlink.** Create `~/.claude/skills/qx-<name>/` → `/Users/yotta/work/qlibQuantExp-skills/skills/<name>/`. Verify with `ls -la ~/.claude/skills/qx-*`.

## What this skill REFUSES

- Skills whose output is "PASS/FAIL" or a validation report.
- Skills without a named lineage.
- Skills that don't cross-reference at least two siblings.
- Skills whose body is a heavy checklist with many MUSTs.
- Manual edits to `MANIFEST.yaml` or `shared/lineages.md` outside this flow.
- Skills proposed after only 1–2 manual uses. Wait for the 3rd.

## Posture line

> "Currently Claude has a tendency to 'undertrigger' skills — to not use them when they'd be useful. To combat this, please make the skill descriptions a little bit 'pushy'." — Anthropic, official `skill-creator/SKILL.md`

> *On not over-engineering:* "Try to explain to the model why things are important in lieu of heavy-handed musty MUSTs. If you find yourself writing ALWAYS or NEVER in all caps, that's a yellow flag." — same source.

## See also

- `qx-research-ledger` — every skill addition is logged here as a `kind=decision` row.
- `qx-protect-or-relax-the-hard-core` — if a new skill changes the package's *identity*, that's a hard-core amendment, classify it there first.
- Anthropic's official skill-creator — extracts in `references/anthropic_skill_creator_extracts.md`.
- `shared/lineages.md` §5 Anthropic / ARIS.
