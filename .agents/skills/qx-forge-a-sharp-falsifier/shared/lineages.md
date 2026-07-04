# Lineages — the 6 Thinkers Behind This Package

Every skill in this package descends from a named scientific move with a working pedigree.
When uncertain about a skill's intent, read its lineage entry here first.

This file is **navigational** — it is referenced by every SKILL.md and by `qx-skill-creator` when scaffolding new skills.

---

## 1. Dedre Gentner — Structure-Mapping Theory

**Skill descended:** `transpose-from-another-field`

> "An analogy is a mapping of knowledge from one domain (the base) into another (the target) which conveys that a system of relations that holds among the base objects also holds among the target objects. … In analogy, what is mapped are relations, particularly higher-order relational structure."
>
> — *Gentner, "Structure-Mapping: A Theoretical Framework for Analogy" (1983)*

**Why it earns its place.** Most "analogies" in quant research are surface-level word matches (e.g., "regime" sounds like "weather"). Gentner's structure-mapping discipline forces analogies to carry **relational structure** — preserving how the parts of one system act on each other when transposed. Three historical breakthroughs that obeyed this rule: Maxwell's mechanical vortex model of electromagnetism (1861), Bohr's solar-system atom (1913), Darwin's importation of Malthus's "struggle for existence" from demography into biology (1838). The systematicity principle is what separates a productive analogy from a misleading one.

---

## 2. Karl Popper — Conjectures and Refutations

**Skill descended:** `forge-a-sharp-falsifier`

> "Every good scientific theory is a prohibition: it forbids certain things to happen. The more a theory forbids, the better it is. … Confirmations should count only if they are the result of risky predictions; that is to say, if, unenlightened by the theory in question, we should have expected an event which was incompatible with the theory — an event which would have refuted the theory."
>
> — *Popper, "Conjectures and Refutations" (1963), §1*

**Why it earns its place.** Quant research is permeated with soft claims ("the model captures momentum reversion"). Without an ex-ante decision rule, the same result can be retrofitted as support for any of several theories — the "conventionalist twist" Popper warned about. A sharp falsifier names exactly what the theory **forbids** before any data is touched, with the metric, window, and threshold pinned down. The output is a numbered prohibition, not a hope.

---

## 3. Michael Faraday — Diary and Idea Books

**Skill descended:** `notebook-the-anomaly` (Idea Books) and `qx-research-ledger` (Diary + Index)

> Faraday kept two strictly separate categories of notebooks across 42 years of work (1820–1862): the *Diary* — sequentially numbered experiments with date, conditions, results — and *Idea Books* / loose memoranda where speculation, conjecture, and stray thoughts went. The Diary entries that graduated to publication were marked with vertical lines in the margin.

**Why it earns its place.** Two distinct disciplines — recording facts and recording speculations — share the word "notebook" in casual English but were rigorously separated in Faraday's actual practice. Mixing them is the most common research-hygiene failure. `notebook-the-anomaly` is the Idea Book: raw speculation about surprises. `qx-research-ledger` is the Diary + Index: numbered, dated, append-only record of what actually happened. Both are essential; conflating them is fatal.

---

## 4. Imre Lakatos — Methodology of Scientific Research Programmes

**Skill descended:** `protect-or-relax-the-hard-core`

> "A research programme is successful if all this leads to a *progressive problemshift*; unsuccessful if it leads to a *degenerating problemshift*. … The programme has a hard core that the negative heuristic forbids us to modify, and a protective belt of auxiliary hypotheses that the positive heuristic suggests we modify."
>
> — *Lakatos, "Falsification and the Methodology of Scientific Research Programmes" (1970)*

Also drawing on *Proofs and Refutations* (1976) — the dialectic of monster-barring (ad-hoc redefinition), exception-barring (carving out), and lemma-incorporation (locate which sub-claim fails, absorb the counterexample).

**Why it earns its place.** macV3's failure mode in qlibQuantExp was a textbook degenerating problemshift: 15k+ insertions in one diff, multiple new conditional branches, all defending the central regime-routing hypothesis without producing novel predictions. A Lakatos discipline would have flagged this as a candidate degenerating move *before* it was implemented. The skill maintains a living manifesto naming what is **core** (untouchable without paradigm shift) vs **belt** (free to modify), and red-flags monster-barring moves at the moment they are proposed.

---

## 5. Anthropic / ARIS — Skill-Creator Pattern

**Skill descended:** `qx-skill-creator`

> "Currently Claude has a tendency to 'undertrigger' skills — to not use them when they'd be useful. To combat this, please make the skill descriptions a little bit 'pushy'."
>
> — *Anthropic's official `skill-creator/SKILL.md`*

> "`meta-optimize` — Analyzes usage logs to propose SKILL.md improvements"
>
> — *ARIS (wanshuiyin/Auto-claude-code-research-in-sleep)*

**Why it earns its place.** A skills package that cannot evolve becomes stale within months. Anthropic's official `skill-creator` is the canonical pattern for scaffolding new skills with sound triggers; ARIS adds self-improvement. `qx-skill-creator` combines both: it is the **only authorized path** to mutate this package's shape, enforces the imagination-over-compliance ethos, and requires every new skill to declare its own lineage entry here.

---

## 6. MLAgentBench / Reflexion — Research Log and Verbal Memory

**Skill descended:** `qx-research-ledger` (Diary half), `notebook-the-anomaly` (Idea Book half)

> "Agents verbally reflect on task feedback signals, then maintain their own reflective text in an episodic memory buffer to induce better decision-making in subsequent trials."
>
> — *Shinn et al., "Reflexion" (NeurIPS 2023)*

> "Research log: a write-only record of attempts, hypotheses tested, and results observed."
>
> — *MLAgentBench (Huang et al., 2023)*

**Why it earns its place.** The modern agentic-research literature converges on a simple primitive: a persistent, write-only memory that survives across sessions. This package implements that primitive as two complementary files — `shared/ledger.md` for what happened (Faraday's Diary) and `shared/notebook/*.md` for what was thought about it (Faraday's Idea Books). Together they form the cross-session memory that lets a returning Claude (or human) reconstruct the trajectory without re-litigating resolved questions.

---

## Cross-reference map (skill → lineage entries)

| Skill | Primary lineage | Secondary lineage |
|---|---|---|
| `transpose-from-another-field` | §1 Gentner | — |
| `forge-a-sharp-falsifier` | §2 Popper | §4 Lakatos (when claim touches hard core) |
| `notebook-the-anomaly` | §3 Faraday (Idea Books) | §6 Reflexion / MLAgentBench |
| `protect-or-relax-the-hard-core` | §4 Lakatos | — |
| `qx-research-ledger` | §3 Faraday (Diary + Index) | §6 MLAgentBench / Reflexion |
| `qx-skill-creator` | §5 Anthropic skill-creator | §5 ARIS meta-optimize |

Every new skill added via `qx-skill-creator` must append a §N entry here.
