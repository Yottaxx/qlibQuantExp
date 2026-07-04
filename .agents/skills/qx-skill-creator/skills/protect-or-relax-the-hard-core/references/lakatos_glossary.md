# Lakatos Glossary — Hard Core, Belt, Heuristics, Monster-Barring

The vocabulary for the `protect-or-relax-the-hard-core` skill, with verbatim quotes from Lakatos.

## Hard core

> "All scientific research programmes may be characterized by their 'hard core'. The negative heuristic of the programme forbids us to direct the modus tollens at this 'hard core'." — Lakatos (1970)

In practice: the 3–6 commitments that, if surrendered, mean we are working on a different research programme.

## Protective belt

> "Instead, we must use our ingenuity to articulate or even invent 'auxiliary hypotheses', which form a protective belt around this core, and we must redirect the modus tollens to these." — Lakatos (1970)

In practice: every architectural detail, hyperparameter, and design choice that *implements* the hard core but is not the hard core itself.

## Negative heuristic

The set of rules forbidding modification of the hard core. *"Do not aim refutations at the hard core."* Lakatos viewed this as the price of methodological commitment — without it, every result becomes a referendum on the entire programme, and nothing is learned.

## Positive heuristic

> "The positive heuristic consists of a partially articulated set of suggestions or hints on how to change, develop the 'refutable variants' of the research programme, how to modify, sophisticate, the 'refutable' protective belt." — Lakatos (1970)

In practice: the programme's *agenda* — the modifications to the belt that are likely to produce novel predictions.

## Progressive vs degenerating problemshift

> "The programme is *progressive* if each new theory in the sequence has excess empirical content over its predecessor; that is, predicts some novel, hitherto unexpected facts. … *degenerating* if it does not — if all that the theoretician offers are theories which merely accommodate, post hoc, the empirical content of an earlier theory."

In practice:
- **Progressive**: a change to the belt that produces *new testable predictions*, not just patches.
- **Degenerating**: changes to the belt that *accommodate* failed predictions without producing new ones — ad-hoc rescues.

## Monster-barring (from *Proofs and Refutations*, 1976)

> "The Monster-Barrer's reaction is to *change* the definition in such a way that the counter-example is no longer a counter-example."

In practice: when a result fails on a specific case (e.g., a regime, a market period), the move to *redefine* what counts as a valid case in order to preserve the theory. This is the **worst** of the rescue moves because it preserves the theory by emptying it of empirical content.

The dialectic of *Proofs and Refutations* moves from:
1. **Monster-barring** (ad-hoc redefinition) — bad
2. **Exception-barring** (carve out the failing cases explicitly) — better but still defensive
3. **Lemma-incorporation** (locate which sub-claim of the proof actually fails, absorb the counterexample into the theory's structure) — the genuine progressive move

## Quick decision guide for the skill

| Proposed move | Pattern | Classify as |
|---|---|---|
| New auxiliary hypothesis that **adds predictions** | Lakatos positive heuristic at work | progressive ✅ |
| New auxiliary hypothesis that **only fixes** a failure, no new predictions | Belt patching without content | degenerating ⚠️ |
| Redefining what counts as a test case after seeing the result | Monster-barring | red flag 🚩 |
| Excluding a regime from evaluation | Exception-barring | warn; require explicit ex-ante naming |
| Identifying *which* sub-claim of the theory the counterexample actually breaks, then revising that sub-claim | Lemma-incorporation | progressive ✅ |
| Touching the hard core | Paradigm shift | record explicitly; new programme |

## See also

- `templates/hardcore_amendment.md` — the format for a core amendment.
- `templates/red_flag_entry.md` — the format for a monster-barring log entry.
- Lakatos, I. (1970). "Falsification and the methodology of scientific research programmes." In *Criticism and the Growth of Knowledge*, ed. Lakatos & Musgrave.
- Lakatos, I. (1976). *Proofs and Refutations*. Cambridge University Press.
