"""Decorrelation experiment.

The model learns a function-word shortcut for `first_letter_vowel` only because, in
the normal training examples, vowel-initial words are mostly function words. Here we
remove that shortcut from the TRAINING examples: positive examples use vowel-initial
CONTENT words (apple, eagle, ...), and negative examples use consonant-initial words
including FUNCTION words (by, the, ...). The function-word shortcut now mislabels the
training examples, so it no longer fits, and the only simple rule that does fit is the
real one ("the first letter is a vowel").

We then test both the normal and the decorrelated training on the same NOVEL held-out
inputs where the letter rule and the function-word shortcut disagree, and we ask the
model to state the rule after decorrelated training.

  uv run python experiments/decorrelation.py
"""

from __future__ import annotations

import math
import random

from articulation import model
from articulation.inputs import GENERAL, sample_balanced
from articulation.rules import RULE_REGISTRY

RULE = RULE_REGISTRY["first_letter_vowel"]
SEEDS = [0, 1, 2]
N_FEWSHOT = 50
N_CELL = 40

# Training first-words and test first-words are kept disjoint so that the test
# measures generalisation of the rule, not memorisation of specific words.
TRAIN_VOWEL = [
    "apple",
    "acorn",
    "anchor",
    "eagle",
    "elephant",
    "engine",
    "island",
    "ivory",
    "indigo",
    "ocean",
    "otter",
    "oven",
    "umbrella",
    "urn",
]
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
TRAIN_CONS = [
    "by",
    "to",
    "with",
    "for",
    "dog",
    "table",
    "black",
    "wolf",
    "river",
    "peach",
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


def normal_fewshot(seed):
    return sample_balanced(RULE, N_FEWSHOT, seed=seed)


def decorrelated_fewshot(seed):
    rng = random.Random(6000 + seed)
    half = N_FEWSHOT // 2
    items = [(sentence(rng.choice(TRAIN_VOWEL), rng), True) for _ in range(half)]
    items += [
        (sentence(rng.choice(TRAIN_CONS), rng), False) for _ in range(N_FEWSHOT - half)
    ]
    rng.shuffle(items)
    return items


def cell_a(rng):  # novel vowel-content first word -> intended True
    return sentence(rng.choice(TEST_VOWEL), rng)


def cell_b(rng):  # consonant function-word first word -> intended False
    return sentence(rng.choice(TEST_CONS_FUNCTION), rng)


def ci95(k, n):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    h = 1.96 * math.sqrt(p * (1 - p) / n)
    return (max(0.0, p - h), min(1.0, p + h))


def eval_condition(name, fewshot_fn):
    match = total = 0
    a_true = a_n = b_false = b_n = 0
    for seed in SEEDS:
        fs = fewshot_fn(seed)
        rng = random.Random(5000 + seed)
        for _ in range(N_CELL):
            x = cell_a(rng)
            p = model.classify_one(fs, x)[0]
            if p is None:
                continue
            total += 1
            a_n += 1
            if p is True:
                a_true += 1
            if p == first_letter_vowel(x):
                match += 1
        for _ in range(N_CELL):
            x = cell_b(rng)
            p = model.classify_one(fs, x)[0]
            if p is None:
                continue
            total += 1
            b_n += 1
            if p is False:
                b_false += 1
            if p == first_letter_vowel(x):
                match += 1
    lo, hi = ci95(match, total)
    print(f"{name}")
    print(
        f"   behaviour matches the LETTER rule: {match}/{total} = {match / total:.0%} (CI {lo:.0%}-{hi:.0%})"
    )
    print(
        f"   cell A (novel vowel content, intended True):    said True  {a_true}/{a_n}"
    )
    print(
        f"   cell B (consonant function, intended False):    said False {b_false}/{b_n}"
    )


def main():
    print("=== Decorrelation: does removing the shortcut force the letter rule? ===\n")
    eval_condition("NORMAL training (original distribution)", normal_fewshot)
    print()
    eval_condition("DECORRELATED training (shortcut removed)", decorrelated_fewshot)
    print(
        "\n=== What rule does the model state after DECORRELATED training? (no reasoning) ==="
    )
    for seed in SEEDS:
        fs = decorrelated_fewshot(seed)
        stated = model.extract_rule(model.articulate_one(fs, cot=False))
        print(f"   seed {seed}: {stated}")


if __name__ == "__main__":
    main()
