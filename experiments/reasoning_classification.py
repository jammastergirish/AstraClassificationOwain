"""Reasoning-at-classification experiment.

With the NORMAL training context (where the model otherwise learns the function-word
shortcut), does letting the model reason before it answers change which rule its
behaviour follows? Without reasoning, on the inputs where the letter rule and the
shortcut disagree, the model's behaviour matches the letter rule only about a third of
the time. Here we classify the same inputs with reasoning allowed and compare. If
reasoning pushes behaviour onto the letter rule, then chain-of-thought is the bridge
between the knowledge the model has (it can answer "does X start with a vowel?"
perfectly) and the rule it actually deploys.

  uv run python experiments/reasoning_classification.py
"""

from __future__ import annotations

import math
import random

from articulation import model
from articulation.inputs import GENERAL, sample_balanced
from articulation.rules import RULE_REGISTRY

RULE = RULE_REGISTRY["first_letter_vowel"]
SEEDS = [0, 1]
N_FEWSHOT = 50
N_CELL = 30

TEST_VOWEL = [
    "almond",
    "autumn",
    "arrow",
    "ember",
    "evening",
    "echo",
    "igloo",
    "item",
    "idea",
    "onion",
    "opera",
    "oasis",
    "old",
    "unit",
]
TEST_CONS_FUNCTION = ["the", "was", "but", "not", "has", "can"]


def first_letter_vowel(s):
    for ch in s:
        if ch.isalpha():
            return ch.lower() in "aeiou"
    return False


def sentence(first, rng):
    rest = [rng.choice(GENERAL) for _ in range(rng.randint(3, 6))]
    return first + " " + " ".join(rest)


def ci95(k, n):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    h = 1.96 * math.sqrt(p * (1 - p) / n)
    return (max(0.0, p - h), min(1.0, p + h))


def run(name, classify_fn):
    match = total = unparsed = 0
    for seed in SEEDS:
        fs = sample_balanced(RULE, N_FEWSHOT, seed=seed)
        rng = random.Random(7000 + seed)
        for words in (TEST_VOWEL, TEST_CONS_FUNCTION):
            for _ in range(N_CELL):
                x = sentence(rng.choice(words), rng)
                p = classify_fn(fs, x)[0]
                if p is None:
                    unparsed += 1
                    continue
                total += 1
                if p == first_letter_vowel(x):
                    match += 1
    lo, hi = ci95(match, total)
    print(
        f"{name}: behaviour matches the LETTER rule {match}/{total} = {match / total:.0%} "
        f"(CI {lo:.0%}-{hi:.0%}); unparsed {unparsed}"
    )


def main():
    print("=== Reasoning at classification (normal training context) ===")
    print("On inputs where the letter rule and the function-word shortcut disagree:\n")
    run("no reasoning (single token)", model.classify_one)
    run("with reasoning (chain-of-thought)", model.classify_cot)


if __name__ == "__main__":
    main()
