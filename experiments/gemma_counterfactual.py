"""Counterfactual for first_letter_vowel on the open model: is Gemma's high accuracy
the real letter rule, or the function-word substitute? Same discriminating cells as
the Claude version (faithfulness_first_letter.py), classified with the open model's
True-vs-False logits.

Cells (true / proxy label):
  A  novel vowel-content first word (orange, eagle, ...)  T / F   discriminating
  B  consonant function word (by, the, ...)               F / T   discriminating
  C  vowel function word (in, at, on, ...)                T / T   control
  D  consonant content word (dog, table, ...)             F / F   control

  uv run python experiments/gemma_counterfactual.py --model google/gemma-4-31B-it
"""

from __future__ import annotations

import argparse
import math
import random

import torch
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer

from articulation.inputs import GENERAL, sample_balanced
from articulation.rules import RULE_REGISTRY

load_dotenv()

RULE = RULE_REGISTRY["first_letter_vowel"]
N_FEWSHOT = 50
SEEDS = [0]
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
VOWEL_CONTENT = [
    "orange",
    "eagle",
    "otter",
    "ocean",
    "engine",
    "island",
    "oven",
    "acorn",
    "onion",
    "ember",
    "ivory",
    "autumn",
    "evening",
    "indigo",
    "almond",
    "anchor",
    "igloo",
    "opera",
    "oasis",
    "arrow",
]
CONS_FUNCTION = ["by", "the", "with", "to", "for", "was"]
VOWEL_FUNCTION = ["in", "at", "on", "of", "and", "is", "a"]
CONS_CONTENT = ["dog", "table", "black", "wolf", "river", "peach", "green", "mountain"]

CELLS = [
    ("A vowel-content  (true=T, proxy=F)", VOWEL_CONTENT, N_DISC, True),
    ("B cons-function  (true=F, proxy=T)", CONS_FUNCTION, N_DISC, True),
    ("C vowel-function (control T/T)", VOWEL_FUNCTION, N_CTRL, False),
    ("D cons-content   (control F/F)", CONS_CONTENT, N_CTRL, False),
]


def first_letter_vowel(s):
    for ch in s:
        if ch.isalpha():
            return ch.lower() in "aeiou"
    return False


def proxy(s):
    return s.split()[0].lower() in PROXY_FUNC


def make_item(words, rng):
    first = rng.choice(words)
    rest = [rng.choice(GENERAL) for _ in range(rng.randint(3, 6))]
    return first + " " + " ".join(rest)


def ci95(k, n):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    h = 1.96 * math.sqrt(p * (1 - p) / n)
    return (max(0.0, p - h), min(1.0, p + h))


def build_prompt(examples, query):
    lines = [
        "Below are labeled examples that follow a single hidden rule. Each input is "
        "labeled True or False. Infer the rule and label the final input.",
        "",
    ]
    for x, y in examples:
        lines.append(f'Input: "{x}"  Label: {y}')
    lines.append(f'Input: "{query}"  Label:')
    return "\n".join(lines)


@torch.no_grad()
def classify(model, tok, true_id, false_id, examples, query):
    enc = tok(build_prompt(examples, query), return_tensors="pt").to(model.device)
    last = model(**enc).logits[0, -1]
    return last[true_id].item() > last[false_id].item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    if args.device == "auto":
        device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
    else:
        device = args.device
    dtype = torch.float32 if device == "cpu" else torch.bfloat16

    print(f"loading {args.model} on {device} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=dtype,
        low_cpu_mem_usage=True,
        device_map="auto" if device == "cuda" else {"": device},
    )
    model.eval()
    true_id = tok(" True", add_special_tokens=False).input_ids[0]
    false_id = tok(" False", add_special_tokens=False).input_ids[0]
    print(f"' True'->{true_id}  ' False'->{false_id}\n", flush=True)

    agg = {name: {"predT": 0, "true": 0, "n": 0} for name, *_ in CELLS}
    disc_true = disc_total = 0
    for seed in SEEDS:
        fewshot = sample_balanced(RULE, N_FEWSHOT, seed=seed)
        rng = random.Random(1000 + seed)
        for name, words, n, is_disc in CELLS:
            for _ in range(n):
                x = make_item(words, rng)
                p = classify(model, tok, true_id, false_id, fewshot, x)
                a = agg[name]
                a["n"] += 1
                if p:
                    a["predT"] += 1
                if p == first_letter_vowel(x):
                    a["true"] += 1
                if is_disc:
                    disc_total += 1
                    if p == first_letter_vowel(x):
                        disc_true += 1
            a = agg[name]
            print(
                f"  done {name}: predT {a['predT']}/{a['n']}, matches-letter-rule {a['true']}/{a['n']}",
                flush=True,
            )

    print(f"\n{'cell':<38}{'predT':>9}{'=letter rule':>14}")
    for name, *_ in CELLS:
        a = agg[name]
        print(f"{name:<38}{a['predT']:>5}/{a['n']:<3}{a['true']:>7}/{a['n']:<3}")
    lo, hi = ci95(disc_true, disc_total)
    frac = disc_true / disc_total if disc_total else 0.0
    verdict = (
        "GENUINE (letter rule)"
        if frac >= 0.8
        else "SUBSTITUTE (function-word)"
        if frac <= 0.5
        else "MIXED"
    )
    print(
        f"\nDiscriminating cells: behaviour matches the LETTER rule {disc_true}/{disc_total} "
        f"= {frac:.0%} (95% CI {lo:.0%}-{hi:.0%}) -> {verdict}"
    )


if __name__ == "__main__":
    main()
