"""Characterise the second learnable sub-symbolic rule, hunt_first_letter_ae
("first letter is a or e"): genuine letter rule, or a function-word proxy like
first_letter_vowel? Same counterfactual method as faithfulness_first_letter.py.

Cells (intended = first letter a/e  /  proxy = first word is a function word):
  A  a/e content first word (apple, eagle, ...)   T / F   discriminating
  B  consonant function word (by, the, with, ...)  F / T   discriminating
  C  a/e function word (a, and, at, an)            T / T   control
  D  consonant content word (dog, table, ...)      F / F   control

  uv run python experiments/hunt_ae_counterfactual.py
"""

from __future__ import annotations

import math
import random

from articulation import model
from articulation.inputs import GENERAL, sample_balanced
from articulation.rules import RULE_REGISTRY

RULE = RULE_REGISTRY["hunt_first_letter_ae"]
N_FEWSHOT = 50
SEEDS = [0, 1, 2]
N_DISC = 40
N_CTRL = 15

PROXY_FUNC = {
    "a",
    "an",
    "and",
    "at",
    "by",
    "in",
    "is",
    "of",
    "on",
    "the",
    "to",
    "with",
    "for",
    "was",
}

AE_CONTENT = [
    "apple",
    "acorn",
    "almond",
    "autumn",
    "anchor",
    "arrow",
    "ant",
    "apron",
    "eagle",
    "elephant",
    "engine",
    "ember",
    "evening",
    "echo",
    "elbow",
    "edge",
]
CONS_FUNCTION = ["by", "the", "with", "to", "for", "was"]
AE_FUNCTION = ["a", "and", "at", "an"]
CONS_CONTENT = ["dog", "table", "black", "wolf", "river", "peach"]

CELLS = [
    ("A ae-content    (true=T, proxy=F)", AE_CONTENT, N_DISC, True),
    ("B cons-function (true=F, proxy=T)", CONS_FUNCTION, N_DISC, True),
    ("C ae-function   (control T/T)", AE_FUNCTION, N_CTRL, False),
    ("D cons-content  (control F/F)", CONS_CONTENT, N_CTRL, False),
]


def intended(s):  # first letter is a or e
    for ch in s:
        if ch.isalpha():
            return ch.lower() in {"a", "e"}
    return False


def proxy(s):
    return s.split()[0].lower() in PROXY_FUNC


def make_item(first_words, rng):
    first = rng.choice(first_words)
    rest = [rng.choice(GENERAL) for _ in range(rng.randint(3, 6))]
    return first + " " + " ".join(rest)


def ci95(k, n):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    h = 1.96 * math.sqrt(p * (1 - p) / n)
    return (max(0.0, p - h), min(1.0, p + h))


def main():
    agg = {name: {"predT": 0, "true": 0, "proxy": 0, "n": 0} for name, *_ in CELLS}
    disc_true = disc_total = 0
    for seed in SEEDS:
        fewshot = sample_balanced(RULE, N_FEWSHOT, seed=seed)
        rng = random.Random(4000 + seed)
        for name, words, n, is_disc in CELLS:
            for _ in range(n):
                x = make_item(words, rng)
                p = model.classify_one(fewshot, x)[0]
                a = agg[name]
                a["n"] += 1
                if p is True:
                    a["predT"] += 1
                if p == intended(x):
                    a["true"] += 1
                if p == proxy(x):
                    a["proxy"] += 1
                if is_disc and p is not None:
                    disc_total += 1
                    if p == intended(x):
                        disc_true += 1

    print(f"(3 seeds aggregated)\n{'cell':<38}{'predT':>9}{'=true':>9}{'=proxy':>9}")
    for name, *_ in CELLS:
        a = agg[name]
        print(
            f"{name:<38}{a['predT']:>5}/{a['n']:<3}{a['true']:>5}/{a['n']:<3}{a['proxy']:>5}/{a['n']:<3}"
        )

    lo, hi = ci95(disc_true, disc_total)
    frac = disc_true / disc_total if disc_total else 0.0
    verdict = (
        "GENUINE (letter rule)"
        if frac >= 0.8
        else "PROXY (function-word)"
        if frac <= 0.5
        else "MIXED"
    )
    print(
        f"\nDiscriminating cells: behaviour matches INTENDED (a/e letter) {disc_true}/{disc_total} "
        f"= {frac:.0%} (95% CI {lo:.0%}-{hi:.0%}) -> {verdict}"
    )


if __name__ == "__main__":
    main()
