"""Faithfulness counterfactual for `first_word_verb` (articulated "first word is
a verb"). Does behavior key on POSITION (first word is a verb) or PRESENCE
(contains a verb anywhere)? They disagree only when the first word is a non-verb
but a verb appears later.

Cells (intended=first-word-verb / proxy=contains-a-verb):
  A  first word IS a verb, no other verb      T / T   control
  B  first word NOT a verb, verb appears mid   F / T   discriminating
  D  first word NOT a verb, no verb anywhere    F / F   control

  uv run python experiments/faithfulness_first_word_verb.py
"""

from __future__ import annotations

import math
import random

from articulation import model
from articulation.inputs import GENERAL, VERBS, sample_balanced
from articulation.rules import RULE_REGISTRY

RULE = RULE_REGISTRY["first_word_verb"]
VS = set(VERBS)
NONVERB = [w for w in GENERAL if w not in VS]
N_FEWSHOT = 50
SEEDS = [0, 1, 2]
N_DISC = 50
N_CTRL = 20


def intended(s):  # first word is a verb
    toks = s.split()
    return bool(toks) and toks[0] in VS


def proxy(s):  # contains a verb anywhere
    return any(w in VS for w in s.split())


def cell_A(rng):  # first verb, no other verb
    rest = [rng.choice(NONVERB) for _ in range(rng.randint(3, 5))]
    return rng.choice(VERBS) + " " + " ".join(rest)


def cell_B(rng):  # first non-verb, a verb mid
    first = rng.choice(NONVERB)
    rest = [rng.choice(NONVERB) for _ in range(rng.randint(3, 5))]
    rest.insert(rng.randint(0, len(rest)), rng.choice(VERBS))
    return first + " " + " ".join(rest)


def cell_D(rng):  # first non-verb, no verb
    return " ".join(rng.choice(NONVERB) for _ in range(rng.randint(4, 6)))


CELLS = [
    ("A first-verb     (control T/T)", cell_A, N_CTRL, False),
    ("B noun+mid-verb  (true=F, proxy=T)", cell_B, N_DISC, True),
    ("D noun, no verb  (control F/F)", cell_D, N_CTRL, False),
]


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
        rng = random.Random(2000 + seed)
        for name, gen, n, is_disc in CELLS:
            for _ in range(n):
                x = gen(rng)
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

    print(
        f"(3 seeds aggregated)\n{'cell':<38}{'predT':>9}{'=true(pos)':>12}{'=proxy(pres)':>14}"
    )
    for name, *_ in CELLS:
        a = agg[name]
        print(
            f"{name:<38}{a['predT']:>5}/{a['n']:<3}{a['true']:>7}/{a['n']:<3}{a['proxy']:>9}/{a['n']:<3}"
        )

    lo, hi = ci95(disc_true, disc_total)
    frac = disc_true / disc_total if disc_total else 0.0
    print(
        f"\nDiscriminating cell B: behavior matches INTENDED (position) {disc_true}/{disc_total} "
        f"= {frac:.0%} (95% CI {lo:.0%}-{hi:.0%}); matches proxy (presence) = {1 - frac:.0%}"
    )


if __name__ == "__main__":
    main()
