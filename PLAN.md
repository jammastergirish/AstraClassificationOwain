# Plan — hardening the findings before the writeup

## STATUS: ✅ all four tasks complete (see RESULTS.md Experiments 14–18)

- **Task 1 — proxy-checks:** ✅ `is_question` and the conjunction both **genuine** (97% / 100% match intended). With `first_word_verb`, 3/3 non-trivial survivors genuine.
- **Task 2 — confounds:** ✅ zero-shot at chance (genuine ICL); prior-only articulation fails (no leakage); survivor/contrast seed CIs don't overlap. `first_word_verb` marginal (one seed 80%).
- **Task 3 — rule hunt:** ✅ found a 2nd learnable sub-symbolic rule (`hunt_first_letter_ae`, 100%) → counterfactual shows it's **also a proxy** (replication). Bonus: arbitrary cut (a–m) not learnable → feature needs a semantic handle.
- **Task 4 — code audit:** ✅ **no critical bug.** Fixed M1 (split novel/in-vocab → cleaner 63% proxy + memorisation probe); report per-cell now (M2/M3); `DIGITS` dup left as-is (cosmetic).

Original plan below.

---

Four tasks. Tasks 1–3 are independent and can run in parallel; Task 4 (code audit)
runs **after** them so it reviews the final codebase. Each task says what it is, why,
the concrete steps, what "done" looks like, how delegatable it is, and rough cost.

Guiding principle (from how this project has gone): **agents do the legwork; the
scientific calls stay with us.** We just watched a result reverse under scrutiny
(Experiment 11), so every agent's findings get reviewed before we believe them.

---

## Task 1 — Proxy-check the remaining ≥90% rules

**Why.** After Experiment 11, "the model articulated the rule correctly" no longer
proves it *learned the intended rule* — it may have learned an on-distribution proxy
that merely matches the articulation. We verified `first_word_verb` is genuine; the
other ≥90% rules are still unchecked.

**Rules to check:** `is_question`, `contains_digit_and_lowercase` (conjunction),
`all_lowercase`, `contains_number`.

**Steps.** For each rule, name a plausible proxy and build inputs where the intended
rule and the proxy **disagree**, then classify (no-CoT) and measure which rule
governs behaviour — exactly the `fig5` method.
- `is_question` — intended "ends with `?`"; proxy "contains a `?` anywhere". Put a
  `?` mid-sentence (no trailing `?`) → intended False, proxy True.
- conjunction — intended "digit AND lowercase"; proxies "digit only" / "lowercase
  only". Test the digit-but-uppercase and lowercase-but-no-digit cells directly.
- `all_lowercase`, `contains_number` — likely genuine (the proxy ≈ the rule); a quick
  confirmation cell each.

**Done when:** each rule is labelled genuine or proxy, with a CI on the discriminating
cell, in one table (parallel to `fig5`).

**Delegatable:** medium — the proxy *hypotheses* are a judgment call (we set them);
execution and measurement are mechanical.

**Cost:** ~$2–3, ~10–15 min.

---

## Task 2 — Close the confound gaps the brief expects

**Why.** Step 1 currently rests on single-seed accuracies and an incomplete control
set. These are cheap and materially harden the result.

**Steps.**
1. **Zero-shot baseline (C2).** Classify with *no* examples (instruction only) for
   every rule → ICL contribution = few-shot − zero-shot. Uses `classify_one([], x)`.
2. **Prior-only articulation (C7).** Ask for the rule given *zero* (or shuffled-label)
   examples. If the model "articulates" it without learning, articulation success is
   contaminated by priors. Run on the ≥90% rules.
3. **Seed bars / CIs on Step 1.** Re-run the sweep across ≥3 few-shot seeds; report
   mean ± 95% CI per rule (replaces the single-seed numbers in `fig1`).

**Done when:** a zero-shot-vs-few-shot table, a prior-only-articulation table, and a
Step-1 sweep with error bars.

**Delegatable:** high — mechanical, follows existing script patterns.

**Cost:** ~$3–5, ~15–20 min (the seed CIs dominate).

---

## Task 3 — Hunt a second sub-symbolic-but-learnable rule

**Why.** The proxy finding currently rests on a *single* rule (`first_letter_vowel`).
A second example would make it robust; failing to find one (after a real search) is
itself a reportable result about how rare that seam is.

**Steps.** Design candidates that are sub-symbolic yet plausibly learnable via a
*salient* position or a holistic cue, run Step-1 on them, and for any that clear ~90%
run articulation + the counterfactual. Candidate seeds: "first letter is in {a,e}",
"first word is ≤3 letters" (length), "starts with two consonants", "first and last
word share a starting letter". Most will fail Step 1 (expected); we want the rare one
that doesn't.

**Done when:** ≥1 additional learnable sub-symbolic rule found and characterised
(genuine vs proxy), **or** a documented negative ("we tried N, all failed Step 1").

**Delegatable:** low — exploratory, needs design judgment; best done with us in the
loop or by an agent whose candidate list and results we review.

**Cost:** variable, ~$3–5.

---

## Task 4 — Code audit (runs AFTER 1–3)

**Why.** Before we lean on these numbers (and before the report), confirm the harness
is correct. A wrong labeler or a train/test leak would invalidate conclusions.

**Steps.** Read all of `src/articulation/` and `experiments/` and verify:
- every labeler actually implements its stated `human_articulation`;
- generators produce valid, balanced inputs; no degenerate cases;
- **no train/test leakage** (few-shot vs test seeds disjoint; constructed cells sane);
- cache keys are complete (no silent collisions across experiments);
- classification is genuinely no-CoT / single-token; articulation parsing is sound;
- CIs / proportions computed correctly; no off-by-one in cell logic.

**Done when:** a findings report by severity, and any correctness bugs fixed.

**Delegatable:** high — classic read-only code-review task, **no API cost**. Depends
on Tasks 1–3 having landed their code.

**Cost:** ~0 API (reviewer tokens only).

---

## Dependencies & ordering

```
Task 1 ─┐
Task 2 ─┼─► Task 4 (code audit of the whole, final codebase)
Task 3 ─┘
```

Tasks 1–3 independent (parallelisable). Task 4 last. Each task **reports results for
our review** rather than finalising scientific claims on its own.
