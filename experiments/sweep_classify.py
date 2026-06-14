"""Step 1 sweep: classification accuracy for every registered rule at a fixed
example count. No articulation. Uses the same seeds as scale_fewshot.py so any
overlapping (rule, n) cells reuse the on-disk cache.

  uv run python experiments/sweep_classify.py
"""

from __future__ import annotations

from tqdm import tqdm

from articulation import model
from articulation.inputs import sample_balanced
from articulation.rules import RULE_REGISTRY

N_FEWSHOT = 50
N_TEST = 50


def run(rule):
    fewshot = sample_balanced(rule, N_FEWSHOT, seed=0)
    test = sample_balanced(rule, N_TEST, seed=1000)
    correct = unparsed = 0
    per = {True: [0, 0], False: [0, 0]}  # [correct, total] per true class
    for x, y in tqdm(test, desc=rule.key, leave=False, disable=None):
        pred, _ = model.classify_one(fewshot, x)
        if pred is None:
            unparsed += 1
            continue
        per[y][1] += 1
        if pred == y:
            correct += 1
            per[y][0] += 1
    n_ok = N_TEST - unparsed
    acc = correct / n_ok if n_ok else 0.0
    return acc, correct, n_ok, per, unparsed


def main():
    rows = []
    for key, rule in RULE_REGISTRY.items():
        acc, correct, n_ok, per, unparsed = run(rule)
        rows.append((rule.category, key, acc))
        print(
            f"{rule.category:>10} {key:<16} acc={acc:>5.0%}  ({correct}/{n_ok})  "
            f"T:{per[True][0]}/{per[True][1]} F:{per[False][0]}/{per[False][1]}  unparsed={unparsed}"
        )

    rows.sort(key=lambda r: r[2], reverse=True)
    print("\n=== sorted by accuracy ===")
    for cat, key, acc in rows:
        flag = "  <- clears 90%" if acc >= 0.90 else ""
        print(f"{acc:>5.0%}  {key:<16} ({cat}){flag}")


if __name__ == "__main__":
    main()
