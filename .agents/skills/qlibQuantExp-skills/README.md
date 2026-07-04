# qlibQuantExp-skills

An imagination-driven Claude Code skills package for the qlibQuantExp quant research project (RST-MoE on CSI300/CSI800, t+5 horizon).

**Posture:** tools for *thinking*, not templates for *verification*. Compliance reduces error; imagination produces new knowledge. Each skill embeds a scientific move with a named pedigree (Polya, Lakatos, Popper, Faraday, Gentner, Reflexion/MLAgentBench) and writes a generative artifact under `shared/`.

## The 6 skills

| Skill | Lineage | Generative artifact |
|---|---|---|
| **`qx-transpose-from-another-field`** | Gentner structure-mapping | `shared/analogies/<concept>.md` |
| **`qx-forge-a-sharp-falsifier`** | Popper, *Conjectures and Refutations* | `shared/falsifiers/<claim-id>.md` |
| **`qx-notebook-the-anomaly`** | Faraday's Idea Books, Darwin's Notebooks B–D | `shared/notebook/YYYY-MM-DD.md` |
| **`qx-protect-or-relax-the-hard-core`** | Lakatos, *Methodology of Scientific Research Programmes* | `shared/hardcore.md` (manifesto) |
| **`qx-research-ledger`** | Faraday's Diary + Index; Reflexion / MLAgentBench | `shared/ledger.md` (append-only) |
| **`qx-skill-creator`** | Anthropic skill-creator; ARIS meta-optimize | mutates the package itself |

## Session entry point

**Always start a new session by invoking `qx-research-ledger` in survey mode.** It reads the last 10–20 ledger rows and produces a one-paragraph state summary ("the frontier is X, pending sweeps are Y, last baseline was Z"). This is what prevents future-Claude from redesigning macV3 from scratch.

## Reading order for first-time arrival

1. This file.
2. `shared/lineages.md` — the 6 thinkers and what they contribute.
3. `shared/hardcore.md` — the project's Lakatosian manifesto (what is core, what is belt, what is a red flag).
4. `shared/ledger.md` — most recent rows tell you what's been decided.
5. The 6 SKILL.md files, when each is first invoked.

## How a research week composes

```
Session start → qx-research-ledger (survey)
Monday        → qx-notebook-the-anomaly       (record weekend reading surprises)
Tuesday       → qx-transpose-from-another-field (analogies from one anomaly)
Wednesday     → qx-forge-a-sharp-falsifier     (convert into pre-registered prohibition)
                + qx-research-ledger (append)   (sweep_start)
Thursday      → run experiments (outside this package)
                + qx-research-ledger (append)   (sweep_done, result)
Friday        → qx-protect-or-relax-the-hard-core (verdict touches core? amend manifesto)
                + qx-research-ledger (append)   (decision)
As needed     → qx-skill-creator               (when a thinking move repeats 3+ times)
```

## Package layout

```
qlibQuantExp-skills/
├── README.md              (this file)
├── MANIFEST.yaml          (skill registry: name, version)
├── shared/
│   ├── lineages.md        (the 6 thinkers)
│   ├── hardcore.md        (Lakatos manifesto)
│   ├── ledger.md          (Faraday Diary + Index)
│   ├── notebook/          (Faraday Idea Books, dated)
│   ├── analogies/         (Gentner structure-mapped analogies)
│   └── falsifiers/        (Popper falsifier cards)
└── skills/
    ├── transpose-from-another-field/
    ├── forge-a-sharp-falsifier/
    ├── notebook-the-anomaly/
    ├── protect-or-relax-the-hard-core/
    ├── qx-research-ledger/
    └── qx-skill-creator/
```

Each `skills/<name>/` contains: `SKILL.md`, `references/` (single-source-of-truth design docs), `templates/` (artifact scaffolds), optional `scripts/`.

## Installation

```bash
# 1. Symlink each skill into Claude Code's discovery path:
for skill in transpose-from-another-field forge-a-sharp-falsifier notebook-the-anomaly protect-or-relax-the-hard-core qx-research-ledger qx-skill-creator; do
  # name in symlink = name in SKILL.md frontmatter (with qx- prefix)
  name=$(awk '/^name:/{print $2; exit}' "skills/$skill/SKILL.md")
  ln -snf "$(pwd)/skills/$skill" "$HOME/.claude/skills/$name"
done

# 2. Verify:
ls -la ~/.claude/skills/qx-*
```

## Design principles

This package is consciously not a compliance machine. There are no validators, no anti-pattern checklists, no schema enforcers. The contracts are SKILL.md-described conventions, not runtime gates. The reasoning:

- **Imagination cannot be validated.** A "validator" for analogies would reject the productive ones (Maxwell's vortices were wrong as physics but right as inspiration).
- **Schema enforcement kills append-only-ness.** Append-only artifacts (ledger, hardcore, notebook) must tolerate format drift — they grow over months. Strict schemas freeze them.
- **The Lakatosian hard core is a thinking discipline, not a checklist.** It exists in `shared/hardcore.md` as a manifesto, not as a runtime rule.

What's deferred (and earns its place later via `qx-skill-creator`):
- Deflated Sharpe Ratio (Bailey & López de Prado)
- HAC t-stats (Newey-West)
- Purged + embargoed CV (López de Prado)
- Multi-testing haircut (Harvey/Liu/Zhu)
- Automated sweep dispatch / aggregation skills

These become useful when `n_trials` in the ledger exceeds ~20, i.e., when the project is approaching publication. Until then, the imagination-first orbit suffices.

## See also

- The plan that authored this package: `~/.claude/plans/skills-anthropic-skills-sota-skills-sot-greedy-gadget.md`
- The qlibQuantExp research repo: `/Users/yotta/work/qlibQuantExp`
- The night plan that triggered this package: `qlibQuantExp/analysis/night_plan_*.md`
