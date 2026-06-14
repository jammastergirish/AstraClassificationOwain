"""Step 2: articulation for the rules that cleared Step 1, comparing no-CoT
(single-shot, matches Step 1's regime) vs CoT (reasoning allowed). Prints the
full CoT output so we can see *how* it articulates (esp. the conjunction).

Same 50-example set the model classified well on (seed=0).

  uv run python experiments/articulate_survivors.py
"""

from __future__ import annotations

from articulation import model
from articulation.inputs import sample_balanced
from articulation.rules import RULE_REGISTRY

SURVIVORS = [
    "contains_number",
    "all_lowercase",
    "is_question",
    "first_word_verb",
    "contains_digit_and_lowercase",  # conjunction, 96%
    "first_letter_vowel",  # salient-position character, 88%
]
N_FEWSHOT = 50


def extract_rule(text: str) -> str | None:
    for line in reversed(text.splitlines()):
        s = line.strip()
        if s.upper().startswith("RULE:"):
            return s[len("RULE:") :].strip()
    return None


def main():
    for key in SURVIVORS:
        rule = RULE_REGISTRY[key]
        fewshot = sample_balanced(rule, N_FEWSHOT, seed=0)
        nocot = model.articulate_one(fewshot, cot=False)
        cot = model.articulate_one(fewshot, cot=True)
        print("=" * 72)
        print(f"rule key      : {key}")
        print(f"ground truth  : {rule.human_articulation}")
        print(f"no-CoT stated : {extract_rule(nocot) or '(no RULE: line)'}")
        print(f"  CoT stated  : {extract_rule(cot) or '(no RULE: line)'}")
        if key in ("contains_digit_and_lowercase", "first_letter_vowel"):
            print("--- full CoT output ---")
            print(cot)
        print()


if __name__ == "__main__":
    main()
