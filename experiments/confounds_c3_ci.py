"""(C3-CI) Step-1 classification accuracy with seed confidence intervals.

Re-run few-shot classification across 3 seeds (0,1,2), n_test=40 each, with the
test seed offset (1000+seed) so each seed sees a different held-out set. Report
the mean accuracy and a 95% normal-approx CI per rule, pooling all predictions
across seeds (N = total parsed predictions).

Focused rule set: the 6 survivors PLUS four chance-level rules for contrast.

  uv run python experiments/confounds_c3_ci.py
"""

from __future__ import annotations

import math

from tqdm import tqdm

from articulation import model
from articulation.inputs import sample_balanced
from articulation.rules import RULE_REGISTRY

SURVIVORS = [
    "contains_number",
    "all_lowercase",
    "is_question",
    "first_word_verb",
    "contains_digit_and_lowercase",
    "first_letter_vowel",
]
CONTRAST = [
    "last_char_vowel",
    "even_word_count",
    "double_letter",
    "second_word_starts_vowel",
]
RULES = SURVIVORS + CONTRAST

N_FEWSHOT = 50
N_TEST = 40
SEEDS = [0, 1, 2]


def run_seed(rule, seed):
    fewshot = sample_balanced(rule, N_FEWSHOT, seed=seed)
    test = sample_balanced(rule, N_TEST, seed=1000 + seed)
    correct = unparsed = 0
    for x, y in tqdm(test, desc=f"{rule.key}.s{seed}", leave=False, disable=None):
        pred, _ = model.classify_one(fewshot, x)
        if pred is None:
            unparsed += 1
            continue
        if pred == y:
            correct += 1
    n_ok = N_TEST - unparsed
    return correct, n_ok


def main():
    rows = []
    for key in RULES:
        rule = RULE_REGISTRY[key]
        per_seed_acc = []
        tot_correct = tot_n = 0
        for seed in SEEDS:
            correct, n_ok = run_seed(rule, seed)
            per_seed_acc.append(correct / n_ok if n_ok else 0.0)
            tot_correct += correct
            tot_n += n_ok
        p = tot_correct / tot_n if tot_n else 0.0
        hw = 1.96 * math.sqrt(p * (1 - p) / tot_n) if tot_n else 0.0
        group = "survivor" if key in SURVIVORS else "contrast"
        rows.append((group, key, p, hw, tot_correct, tot_n, per_seed_acc))
        seedstr = " ".join(f"{a:.0%}" for a in per_seed_acc)
        print(
            f"{group:<9} {key:<28} mean={p:>5.0%}  "
            f"95%CI=[{max(0.0, p - hw):.0%},{min(1.0, p + hw):.0%}]  "
            f"(+/-{hw:.0%})  N={tot_n}  seeds[{seedstr}]"
        )

    print("\n=== C3-CI summary (sorted by mean acc) ===")
    rows.sort(key=lambda r: r[2], reverse=True)
    print(f"{'group':<9} {'rule':<28} {'mean':>5} {'95% CI':>14} {'N':>4}")
    for group, key, p, hw, tc, tn, _ in rows:
        ci = f"[{max(0.0, p - hw):.0%},{min(1.0, p + hw):.0%}]"
        print(f"{group:<9} {key:<28} {p:>5.0%} {ci:>14} {tn:>4}")


if __name__ == "__main__":
    main()
