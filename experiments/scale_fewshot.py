"""Few-shot scaling: does a character rule become classifiable with more examples?

Classification accuracy only (no articulation), with the SAME held-out test set
across all n_fewshot so differences are due to example count alone.

  uv run python experiments/scale_fewshot.py
"""

from __future__ import annotations

from tqdm import tqdm

from articulation import model
from articulation.inputs import sample_balanced
from articulation.rules import RULE_REGISTRY

RULES = ["last_char_vowel", "all_lowercase"]
N_FEWSHOT = [18, 50, 100, 200]
N_TEST = 50


def accuracy(rule, fewshot, test):
    correct = unparsed = 0
    for x, y in tqdm(
        test, desc=f"{rule.key} n={len(fewshot)}", leave=False, disable=None
    ):
        pred, _ = model.classify_one(fewshot, x)
        if pred is None:
            unparsed += 1
        elif pred == y:
            correct += 1
    n_ok = len(test) - unparsed
    return correct, n_ok, unparsed


def main():
    rows = []
    for key in RULES:
        rule = RULE_REGISTRY[key]
        test = sample_balanced(rule, N_TEST, seed=1000)
        for n in N_FEWSHOT:
            fewshot = sample_balanced(rule, n, seed=0)
            correct, n_ok, unparsed = accuracy(rule, fewshot, test)
            acc = correct / n_ok if n_ok else 0.0
            rows.append((key, n, correct, n_ok, unparsed, acc))
            print(
                f"{key:<16} n_few={n:>3}  {correct:>2}/{n_ok:<2} = {acc:>5.0%}  unparsed={unparsed}"
            )

    print("\n=== summary ===")
    for key, n, correct, n_ok, unparsed, acc in rows:
        print(
            f"{key:<16} n_few={n:>3}  acc={acc:>5.0%}  ({correct}/{n_ok}, unparsed={unparsed})"
        )


if __name__ == "__main__":
    main()
