"""(C2) Zero-shot / prior-only classification baseline.

For EVERY rule in RULE_REGISTRY, classify a held-out test set with NO few-shot
examples (model.classify_one([], x)). This isolates how much the model relies on
priors vs in-context learning: if zero-shot accuracy is already high, the rule is
solved by priors, not by ICL.

  uv run python experiments/confounds_c2_zeroshot.py
"""

from __future__ import annotations

import math

from tqdm import tqdm

from articulation import model
from articulation.inputs import sample_balanced
from articulation.rules import RULE_REGISTRY

N_TEST = 40
TEST_SEED = 1000

SURVIVORS = {
    "contains_number",
    "all_lowercase",
    "is_question",
    "first_word_verb",
    "contains_digit_and_lowercase",
    "first_letter_vowel",
}


def run(rule):
    test = sample_balanced(rule, N_TEST, seed=TEST_SEED)
    correct = unparsed = 0
    for x, y in tqdm(test, desc=rule.key, leave=False, disable=None):
        pred, _ = model.classify_one([], x)  # ZERO-SHOT
        if pred is None:
            unparsed += 1
            continue
        if pred == y:
            correct += 1
    n_ok = N_TEST - unparsed
    acc = correct / n_ok if n_ok else 0.0
    return acc, correct, n_ok, unparsed


def main():
    rows = []
    for key, rule in RULE_REGISTRY.items():
        acc, correct, n_ok, unparsed = run(rule)
        rows.append((rule.category, key, acc, correct, n_ok, unparsed))
        print(
            f"{rule.category:>10} {key:<28} zs_acc={acc:>5.0%}  "
            f"({correct}/{n_ok})  unparsed={unparsed}"
        )

    rows.sort(key=lambda r: r[2], reverse=True)
    print("\n=== C2 zero-shot, sorted by accuracy ===")
    print(f"{'zs_acc':>7} {'rule':<28} {'cat':<11} {'survivor':<9} flag")
    for cat, key, acc, correct, n_ok, unparsed in rows:
        # 95% CI half width for a single rule (N=n_ok)
        hw = 1.96 * math.sqrt(acc * (1 - acc) / n_ok) if n_ok else 0.0
        surv = "yes" if key in SURVIVORS else ""
        flag = "  <- HIGH zero-shot (priors)" if acc >= 0.80 else ""
        print(
            f"{acc:>6.0%} {key:<28} {cat:<11} {surv:<9} "
            f"[{acc - hw:.0%},{min(1.0, acc + hw):.0%}]{flag}"
        )


if __name__ == "__main__":
    main()
