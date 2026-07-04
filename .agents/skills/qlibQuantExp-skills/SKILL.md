---
name: qlibQuantExp-skills
description: Workspace entry point for the qlibQuantExp research skills package. Use at session start, when orienting on RST-MoE experiments, or when deciding which qx-* thinking skill should handle a research event. First read MANIFEST.yaml, shared/ledger.md, and shared/hardcore.md. Route facts and decisions to qx-research-ledger, surprises to qx-notebook-the-anomaly, testable claims to qx-forge-a-sharp-falsifier, hard-core changes to qx-protect-or-relax-the-hard-core, cross-field idea generation to qx-transpose-from-another-field, and package evolution to qx-skill-creator.
license: MIT
---

# qlibQuantExp Skills Package

## Overview

This is the workspace entry point for the qlibQuantExp research skills package.
It does not replace the six child skills; it tells Codex how to route research
work into them.

On session entry, read:

1. `MANIFEST.yaml`
2. `shared/ledger.md`
3. `shared/hardcore.md`

Then use the child skill that matches the research event.

## Routing

| Event | Use |
|---|---|
| A tracked fact happened: baseline, run result, decision, sweep state | `qx-research-ledger` |
| A surprising result needs preserving before it is explained away | `qx-notebook-the-anomaly` |
| A soft idea needs a falsifiable, pre-registered decision rule | `qx-forge-a-sharp-falsifier` |
| A proposal touches the project's hard core or risks monster-barring | `qx-protect-or-relax-the-hard-core` |
| The team needs a fresh mechanism from another field | `qx-transpose-from-another-field` |
| The package itself needs a new or revised skill | `qx-skill-creator` |

## Posture

Keep Faraday's split intact: facts go to the ledger, speculation goes to the
notebook. Do not turn this package into a validator. Its purpose is to preserve
research memory and generate sharper hypotheses.

## See Also

- `skills/qx-research-ledger/SKILL.md`
- `skills/notebook-the-anomaly/SKILL.md`
- `skills/forge-a-sharp-falsifier/SKILL.md`
- `skills/protect-or-relax-the-hard-core/SKILL.md`
- `skills/transpose-from-another-field/SKILL.md`
- `skills/qx-skill-creator/SKILL.md`
