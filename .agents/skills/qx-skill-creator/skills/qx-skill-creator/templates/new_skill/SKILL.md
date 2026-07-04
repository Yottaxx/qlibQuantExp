---
name: "qx-{{kebab-name}}"
description: '{{Use when ... — what AND when. Pushy. Sibling-aware. ≤ 1024 chars. Include 2+ sibling skill names. State explicit triggers; "phrase A", "phrase B". For Y use qx-other-sibling.}}'
license: Internal
---

# {{Title Case Name}}

## Overview

{{One paragraph: what this skill does and why it exists. Reference the repeated manual move that justified scaffolding.}}

## Posture

- {{Stance 1 — "X over Y" or "we care about X, not Y"}}
- {{Stance 2}}
- {{Stance 3}}

## Quick Reference

| Phase | What we do | What we write |
|---|---|---|
| 1. {{Phase name}} | {{action}} | {{output snippet}} |
| 2. {{Phase name}} | {{action}} | {{output snippet}} |
| ... | ... | ... |

Detailed procedure in `references/{{name}}_protocol.md`.

## Dialogue flow

**Phase 1 — {{Name}}.** {{1-3 sentences on what to do, with example questions or operations.}}

**Phase 2 — {{Name}}.** {{1-3 sentences.}}

**Phase 3 — {{Name}}.** {{1-3 sentences.}}

{{... add more phases as needed, but body should stay ≤ 200 lines total.}}

## Output

{{Write to `shared/<dir>/<name>.md` using `templates/{{template_name}}.md`. Append-only to existing file? New file per use? Be specific.}}

## Posture line (cite at the start of dialogue)

> "{{Verbatim quote from the named thinker / lab / paper}}" — {{citation}}

## See also

- `qx-sibling-1` — {{when to route there instead}}
- `qx-sibling-2` — {{when to route there instead}}
- `shared/lineages.md` §{{N}} {{thinker name}}.
