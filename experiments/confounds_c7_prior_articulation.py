"""(C7) Prior-only articulation.

For the 6 survivor rules, ask the model to articulate the hidden rule (no-CoT)
under two CONTAMINATED conditions where the examples carry no valid signal:
  (i)  ZERO examples.
  (ii) shuffled-label examples (controls.shuffled_labels of a normal 50-example
       set, seed 0) -> inputs are real but labels are scrambled.

If the model still "states" the true rule, the articulation is driven by priors,
not by the in-context evidence. Compare against the true human_articulation.

  uv run python experiments/confounds_c7_prior_articulation.py
"""

from __future__ import annotations

from articulation import controls, model
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
N_FEWSHOT = 50


def main():
    rows = []
    for key in SURVIVORS:
        rule = RULE_REGISTRY[key]

        # (i) zero examples
        zero_text = model.articulate_one([], cot=False)
        zero_rule = model.extract_rule(zero_text) or "(no RULE: line)"

        # (ii) shuffled-label examples
        fewshot = sample_balanced(rule, N_FEWSHOT, seed=0)
        shuf = controls.shuffled_labels(fewshot, seed=0)
        shuf_text = model.articulate_one(shuf, cot=False)
        shuf_rule = model.extract_rule(shuf_text) or "(no RULE: line)"

        rows.append((key, rule.human_articulation, zero_rule, shuf_rule))

        print("=" * 78)
        print(f"rule key            : {key}")
        print(f"ground truth        : {rule.human_articulation}")
        print(f"(i)  zero-example    : {zero_rule}")
        print(f"(ii) shuffled-label  : {shuf_rule}")
        print()

    print("\n=== C7 summary ===")
    for key, gt, zero_rule, shuf_rule in rows:
        print(f"\n[{key}]")
        print(f"   truth   : {gt}")
        print(f"   zero    : {zero_rule}")
        print(f"   shuffled: {shuf_rule}")


if __name__ == "__main__":
    main()
