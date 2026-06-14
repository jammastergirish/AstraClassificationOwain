# Articulating in-context classification rules

Research question (after Owain Evans' brief): **are there classification tasks an
LLM can learn very accurately in-context (>90% on held-out, in-distribution
inputs) without being able to articulate the rule it has learned?**

Subject model: **Claude Opus 4.8** (`claude-opus-4-8`). Step 1 classification runs
with **no chain-of-thought** (single-token answer); Step 2 articulation may use CoT.

> The written report must be the author's own work (AI only for typo/grammar
> checks). This repo is the experimental harness and figure pipeline, not a draft.

## Setup

```bash
cp .env.example .env          # then paste your ANTHROPIC_API_KEY
uv sync                       # creates the venv, installs deps + this package
uv run python experiments/check_rules.py     # local, no API/key needed
uv run python experiments/smoke_test.py --rule contains_number
```

## What's here

| Path | Role |
|---|---|
| `src/articulation/rules/` | one `(input_space, labeler)` per rule + `RULE_REGISTRY` |
| `src/articulation/inputs.py` | input-space generators + 50/50 balanced sampler |
| `src/articulation/model.py` | Claude primitives: `classify_one`, `articulate_one` |
| `src/articulation/controls.py` | confound controls (shuffled labels, …) |
| `src/articulation/cache.py` | on-disk response cache (reruns are free) |
| `experiments/check_rules.py` | local balance sanity check (no API) |
| `experiments/smoke_test.py` | one rule end-to-end (classify + control + articulate) |

Rules span four categories chosen to span the hypothesised axis — **introspective
accessibility**: `lexical` (semantic, expect articulate ✓), `syntactic` (~),
`character` (positional/counting, the interesting classify-✓/articulate-✗ zone),
`confounded` (for the Step 3 faithfulness probe).

## Three steps

1. **Apply** — few-shot classify, no CoT, single token, neutral instruction.
2. **Recognize / Generate** — multiple-choice, then free-form articulation.
3. **Faithfulness** — does the *articulated* rule predict the model's *behavioral*
   rule (counterfactual minimal pairs, articulation-as-classifier, the
   spurious-correlate experiment)?

## Confound battery (the controls that make the result credible)

Protecting **Step 1 (real in-context learning?)**: **C1** shuffled-label · **C2**
zero-shot baseline · **C3** balanced classes + per-class accuracy · **C4**
synthetic inputs + programmatic labelers (no contamination) · **C5** seed/order
variance · **C6** held-out test + binomial CIs.

Protecting **Step 2 (real articulation failure?)**: **C7** prior-only articulation
(the dual of C1) · **C8** neutral Step-1 instruction + larger separate example set ·
**C9** judge validation (lenient vs strict, equivalence not string match) · **C10**
recognition-vs-generation split with on-distribution near-miss distractors.

Protecting **Step 3 (faithful?)**: three rules — *intended* / *articulated* /
*behavioral* — and the science is in the mismatches. The underdetermination
confound: with finite examples many rules fit, so failing to state the *intended*
rule may be *correct* (it learned an equivalent rule); counterfactual/OOD probing
reveals the *behavioral* rule.
