"""Step 1 of the faithfulness probe: is the no-CoT proxy articulation for
`first_letter_vowel` RELIABLE, or was the function-word proxy a one-off?

Re-runs no-CoT (and CoT, for contrast) articulation across several distinct
50-example sets (seeds).

  uv run python experiments/reliability_first_letter.py
"""

from __future__ import annotations

from articulation import model
from articulation.inputs import sample_balanced
from articulation.rules import RULE_REGISTRY

KEY = "first_letter_vowel"
SEEDS = [0, 1, 2, 3, 4, 5]
N_FEWSHOT = 50


def main():
    rule = RULE_REGISTRY[KEY]
    print(f"ground truth: {rule.human_articulation}\n")
    for seed in SEEDS:
        fewshot = sample_balanced(rule, N_FEWSHOT, seed=seed)
        nocot = model.extract_rule(model.articulate_one(fewshot, cot=False))
        cot = model.extract_rule(model.articulate_one(fewshot, cot=True))
        print(f"seed {seed}")
        print(f"  no-CoT: {nocot or '(no RULE line)'}")
        print(f"     CoT: {cot or '(no RULE line)'}")


if __name__ == "__main__":
    main()
