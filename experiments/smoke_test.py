"""End-to-end smoke test for one rule (uses the API; needs ANTHROPIC_API_KEY).

  uv run python experiments/smoke_test.py --rule contains_number
  uv run python experiments/smoke_test.py --rule last_char_vowel --n-test 40

Runs: Step 1 classification accuracy on a held-out test set (no CoT, single
token), the C1 shuffled-label control on a subset, and one Step 2 free-form
articulation. Keeps n small to stay cheap.
"""

from __future__ import annotations

import argparse

from tqdm import tqdm

from articulation import controls, model
from articulation.inputs import sample_balanced
from articulation.rules import RULE_REGISTRY


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rule", default="contains_number", choices=sorted(RULE_REGISTRY))
    ap.add_argument("--n-fewshot", type=int, default=18)
    ap.add_argument("--n-test", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rule = RULE_REGISTRY[args.rule]
    fewshot = sample_balanced(rule, args.n_fewshot, seed=args.seed)
    test = sample_balanced(rule, args.n_test, seed=args.seed + 1000)

    print(f"Rule: {rule.key}  ({rule.category})")
    print(f"Reference articulation: {rule.human_articulation}")
    print(f"Few-shot n={len(fewshot)}, held-out test n={len(test)}\n")

    # --- Step 1: classification (no CoT, single token) -------------------
    correct = unparsed = 0
    per = {True: [0, 0], False: [0, 0]}  # [correct, total] per true class
    for x, y in tqdm(test, desc="Step 1 classify"):
        pred, _ = model.classify_one(fewshot, x)
        if pred is None:
            unparsed += 1
            continue
        per[y][1] += 1
        if pred == y:
            correct += 1
            per[y][0] += 1
    n_ok = len(test) - unparsed
    acc = correct / n_ok if n_ok else 0.0
    print(f"Step 1 classification accuracy: {correct}/{n_ok} = {acc:.1%}")
    print(
        f"  per-class  True: {per[True][0]}/{per[True][1]}  "
        f"False: {per[False][0]}/{per[False][1]}  unparsed: {unparsed}"
    )

    # --- C1: shuffled-label control --------------------------------------
    shuf = controls.shuffled_labels(fewshot, seed=args.seed)
    sub = test[:20]
    c = sum(
        1
        for x, y in tqdm(sub, desc="C1 control")
        if model.classify_one(shuf, x)[0] == y
    )
    print(
        f"  [C1 shuffled-label] acc on {len(sub)}: {c}/{len(sub)} = {c / len(sub):.1%} "
        f"(should fall toward 50% if classification is real ICL)"
    )

    # --- Step 2: free-form articulation ----------------------------------
    print("\nStep 2 articulation (free-form):")
    print(f"  {model.articulate_one(fewshot)}")


if __name__ == "__main__":
    main()
