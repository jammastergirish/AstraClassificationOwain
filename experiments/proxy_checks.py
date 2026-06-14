"""Task 1: proxy-check the remaining >=90% rules. Same counterfactual method as
fig5 — build inputs where the intended rule and a plausible proxy DISAGREE, then
see which governs behaviour. (all_lowercase / contains_number deprioritised: their
proxy is ~the rule itself.)

  uv run python experiments/proxy_checks.py
"""

from __future__ import annotations

import math
import random

from articulation import model
from articulation.inputs import DIGITS, GENERAL, sample_balanced
from articulation.rules import RULE_REGISTRY

N_FEWSHOT = 50
SEEDS = [0, 1, 2]
N_DISC = 40
N_CTRL = 15


def ci95(k, n):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    h = 1.96 * math.sqrt(p * (1 - p) / n)
    return (max(0.0, p - h), min(1.0, p + h))


# ===== is_question: intended "ends with ?"  vs  proxy "contains ? anywhere" ====
def iq_intended(s):
    return s.strip().endswith("?")


def iq_mid_q(rng):  # ? attached mid-sentence, not trailing -> intended F, proxy T
    w = [rng.choice(GENERAL) for _ in range(rng.randint(4, 6))]
    i = rng.randint(0, len(w) - 2)
    w[i] = w[i] + "?"
    return " ".join(w)


def iq_end_q(rng):  # control: both True
    return " ".join(rng.choice(GENERAL) for _ in range(rng.randint(4, 6))) + "?"


def iq_no_q(rng):  # control: both False
    return " ".join(rng.choice(GENERAL) for _ in range(rng.randint(4, 6)))


# ===== conjunction: intended "digit AND lowercase" vs single-feature proxies =====
def cj_intended(s):
    return any(c.isdigit() for c in s) and s == s.lower()


def cj_digit_upper(
    rng,
):  # digit present but UPPERCASE -> intended F, "digit-only" proxy T
    toks = [rng.choice(GENERAL) for _ in range(rng.randint(3, 5))]
    toks.insert(rng.randint(0, len(toks)), rng.choice(DIGITS))
    return " ".join(toks).upper()


def cj_lower_nodigit(
    rng,
):  # all lower, NO digit -> intended F, "lowercase-only" proxy T
    return " ".join(rng.choice(GENERAL) for _ in range(rng.randint(4, 6)))


def cj_digit_lower(rng):  # control: both -> intended T
    toks = [rng.choice(GENERAL) for _ in range(rng.randint(3, 5))]
    toks.insert(rng.randint(0, len(toks)), rng.choice(DIGITS))
    return " ".join(toks)


def cj_upper_nodigit(rng):  # control: neither -> intended F
    return " ".join(rng.choice(GENERAL) for _ in range(rng.randint(4, 6))).upper()


CHECKS = {
    "is_question": {
        "intended": iq_intended,
        "cells": [
            ("mid-?  (intended=F, proxy 'contains ?'=T)", iq_mid_q, N_DISC, True),
            ("end-?  (control T)", iq_end_q, N_CTRL, False),
            ("no-?   (control F)", iq_no_q, N_CTRL, False),
        ],
    },
    "contains_digit_and_lowercase": {
        "intended": cj_intended,
        "cells": [
            ("digit+UPPER (intended=F, 'digit-only'=T)", cj_digit_upper, N_DISC, True),
            (
                "lower+nodigit (intended=F, 'lower-only'=T)",
                cj_lower_nodigit,
                N_DISC,
                True,
            ),
            ("digit+lower (control T)", cj_digit_lower, N_CTRL, False),
            ("UPPER+nodigit (control F)", cj_upper_nodigit, N_CTRL, False),
        ],
    },
}


def main():
    for key, spec in CHECKS.items():
        rule = RULE_REGISTRY[key]
        intended = spec["intended"]
        print("=" * 72)
        print(f"RULE: {key}  -- {rule.human_articulation}")
        disc_genuine = disc_total = 0
        for name, gen, n, is_disc in spec["cells"]:
            predT = match_intended = total = 0
            for seed in SEEDS:
                fewshot = sample_balanced(rule, N_FEWSHOT, seed=seed)
                rng = random.Random(3000 + seed)
                for _ in range(n):
                    x = gen(rng)
                    p = model.classify_one(fewshot, x)[0]
                    if p is None:
                        continue
                    total += 1
                    if p is True:
                        predT += 1
                    if p == intended(x):
                        match_intended += 1
                    if is_disc:
                        disc_total += 1
                        if p == intended(x):
                            disc_genuine += 1
            print(
                f"  {name:<44} predT {predT:>3}/{total:<3}  matches-intended {match_intended}/{total}"
            )
        lo, hi = ci95(disc_genuine, disc_total)
        frac = disc_genuine / disc_total if disc_total else 0.0
        verdict = "GENUINE" if frac >= 0.8 else "PROXY" if frac <= 0.5 else "MIXED"
        print(
            f"  => discriminating cells match INTENDED {disc_genuine}/{disc_total} "
            f"= {frac:.0%} (CI {lo:.0%}-{hi:.0%}) -> {verdict}"
        )


if __name__ == "__main__":
    main()
