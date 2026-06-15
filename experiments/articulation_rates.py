"""No-CoT articulation across 6 independent example sets (seeds) for each rule in
the classify-vs-articulate scatter, so the articulation axis is a consistent
fraction rather than a single 0/1 attempt. Each articulation is printed verbatim so
its correctness (logical equivalence to the rule's human_articulation) can be judged
by reading.

  uv run python experiments/articulation_rates.py
"""

from __future__ import annotations

from articulation import model
from articulation.inputs import sample_balanced
from articulation.rules import RULE_REGISTRY

RULES = [
    "contains_number",
    "all_lowercase",
    "is_question",
    "first_word_verb",
    "contains_digit_and_lowercase",
    "first_letter_vowel",
    "hunt_first_letter_ae",
]
SEEDS = [0, 1, 2, 3, 4, 5]
N_FEWSHOT = 50


def main():
    for key in RULES:
        rule = RULE_REGISTRY[key]
        print("=" * 80)
        print(f"{key}  —  intended: {rule.human_articulation}")
        for s in SEEDS:
            fewshot = sample_balanced(rule, N_FEWSHOT, seed=s)
            stated = (
                model.extract_rule(model.articulate_one(fewshot, cot=False))
                or "(no RULE line)"
            )
            print(f"  seed {s}: {stated}")


if __name__ == "__main__":
    main()
