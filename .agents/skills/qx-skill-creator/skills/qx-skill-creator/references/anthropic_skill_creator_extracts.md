# Anthropic Skill-Creator Extracts

Verbatim quotes from Anthropic's official `skill-creator/SKILL.md` and related public docs. The source of truth for the format conventions this package adopts.

Source: https://raw.githubusercontent.com/anthropics/skills/main/skills/skill-creator/SKILL.md

## On the description field

> "description: When to trigger, what it does. This is the primary triggering mechanism - include both what the skill does AND specific contexts for when to use it. All 'when to use' info goes here, not in the body."

## On pushy triggering

> "Note: currently Claude has a tendency to 'undertrigger' skills — to not use them when they'd be useful. To combat this, please make the skill descriptions a little bit 'pushy'."

## On progressive disclosure

> "1. Metadata (~100 words) — always in context.
> 2. SKILL.md body (<500 lines ideal) — in context whenever skill triggers.
> 3. Bundled resources — as needed, unlimited."

This package uses **<200 lines** as a tighter target (matching what Anthropic actually does in `pptx`, `pdf`, `brand-guidelines`).

## On bundled resources

> "Bundled Resources (optional)
> ├── scripts/    — Executable code for deterministic/repetitive tasks
> ├── references/ — Docs loaded into context as needed
> └── assets/     — Files used in output (templates, icons, fonts)"

This package uses `templates/` instead of `assets/`, in keeping with the artifact-generation theme.

## On voice and style

> "Try to explain to the model why things are important in lieu of heavy-handed musty MUSTs. If you find yourself writing ALWAYS or NEVER in all caps, or using super rigid structures, that's a yellow flag."

> "Cool? Cool." (literal closing line of skill-creator/SKILL.md)
> "Good luck!" (skill-creator concluding line)

The casual register is intentional — over-formal SKILL.md text correlates with skills that nobody invokes.

## On when MUST is OK

The official `xlsx/SKILL.md` uses:

> "Every Excel model MUST be delivered with ZERO formula errors"

MUST is reserved for **hard quality gates that, if violated, render the output unusable**. Not for stylistic preferences.

## On routing patterns

The official `pptx/SKILL.md` is mostly a router that points to `editing.md` and `pptxgenjs.md` in `references/`. The body itself is short. This is the canonical pattern for skills with complex internal logic.

K-Dense's three-skill scientific suite uses the same routing principle across skills (instead of within one skill): `scientific-brainstorming` → `hypothesis-generation` → `hypogenic`, each cross-referencing the others.

## What this package adopts verbatim

- Frontmatter fields: `name`, `description`, `license`.
- The "what AND when" framing for descriptions.
- Pushy triggering.
- `references/` and `templates/` subdirectories.
- Casual register; MUST reserved for hard quality gates.
- Sibling cross-reference instead of mega-skill bloat.

## What this package adds

- The **lineage** requirement (every skill cites a named thinker / paper).
- The **generative artifact** requirement (every skill writes to `shared/`).
- The **posture** section in the body (explicit thinking stance).
- The **≤ 200 line** body target (tighter than Anthropic's "ideal <500").

## See also

- `imagination_skill_contract.md` — the requirements derived from these extracts.
- https://github.com/anthropics/skills — the canonical public source.
