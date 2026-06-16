# Articulating in-context classification rules

Research question (after Owain Evans' brief): **are there classification tasks an
LLM can learn very accurately in-context (>90% on held-out, in-distribution
inputs) without being able to articulate the rule it has learned?**

Subject model: **Claude Opus 4.8** (`claude-opus-4-8`). Step 1 classification runs
with **no chain-of-thought** (single-token answer); Step 2 articulation may use CoT.
An open-weights arm replicates the key finding on a local Hugging Face model (Gemma)
and adds linear probes for the mechanistic version of the story.

## Setup

```bash
cp .env.example .env          # then paste your ANTHROPIC_API_KEY
uv sync                       # creates the venv, installs deps + this package
uv run python experiments/check_rules.py     # local, no API/key needed
uv run python experiments/smoke_test.py --rule contains_number
```

Closed-model experiments need only `ANTHROPIC_API_KEY`; responses are cached on disk,
so reruns are free. The open-weights arm is optional and heavier:

```bash
uv sync --extra gpu           # adds torch / transformers / scikit-learn
# Gemma is gated: accept its licence and add HF_TOKEN to .env
uv run python experiments/gemma_classify.py --model google/gemma-4-31B-it
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
| `experiments/sweep_classify.py`, `scale_fewshot.py` | Step-1 accuracy sweeps |
| `experiments/faithfulness_*.py`, `proxy_checks.py`, `confounds_*.py` | Step-3 + confound batteries |
| `experiments/gemma_classify.py`, `gemma_counterfactual.py` | open-weights replication |
| `experiments/gemma_probe.py`, `gemma_probe_figure.py` | linear probes on the open model |
| `experiments/make_figures.py` | redraw the figures in `figures/` |

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

## Confound battery

A set of controls makes each step credible. **Step 1 (real in-context learning?):**
shuffled-label and zero-shot baselines, balanced classes with per-class accuracy,
synthetic inputs + programmatic labelers (no contamination), seed/order variance, and
held-out test sets with binomial CIs. **Step 2 (real articulation failure?):**
prior-only articulation, a neutral Step-1 instruction, judge validation by equivalence
(not string match), and a recognition-vs-generation split with near-miss distractors.
**Step 3 (faithful?):** three rules are tracked — *intended* / *articulated* /
*behavioral* — and the science is in the mismatches. With finite examples many rules
fit, so failing to state the *intended* rule may be *correct* (an equivalent rule was
learned); counterfactual/OOD probing recovers the *behavioral* rule.
