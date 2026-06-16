"""Linear probes on Gemma for the first_letter_vowel feature.

Two parts, one model load:

  BASELINE (cheap, ~5 min) -- feed each word on its own, read the residual stream at
  its last token, and train a linear probe to recover "first letter is a vowel". This
  is a methodology check: the feature is trivially present, so a working probe must
  score near-ceiling. If it can't recover this, no null result downstream is trustworthy.

  IN-CONTEXT (~90 min) -- feed the full 50-shot classification prompt plus a query
  sentence, read the residual stream at the answer position (the token right after
  'Label:', where the model commits to True/False), and probe for BOTH cues at once:
  the intended rule (first letter is a vowel) and the substitute (first word is a
  function word). This asks whether, at the decision point, the two competing rules are
  each linearly written down -- the representation-level version of the underdetermination
  story, which a closed model can never answer.

Rigour: train/test are split by WORD (held-out words, not held-out items) so the probe
must read the concept, not memorise vocabulary; and the word set is a balanced 2x2 of
vowel/consonant x content/function, so "first letter is a vowel" is orthogonal to "is a
function word" and a high score is genuinely the letter, not the proxy.

Activations are cached to .cache/ so the probing can be re-run (other layers, C, targets)
without re-running the model.

  uv run python experiments/gemma_probe.py --model google/gemma-4-31B-it
"""

from __future__ import annotations

import argparse
import os
import random
from collections import defaultdict

import numpy as np
import torch
from dotenv import load_dotenv
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from transformers import AutoModelForCausalLM, AutoTokenizer

from articulation.inputs import GENERAL, sample_balanced
from articulation.rules import RULE_REGISTRY

load_dotenv()

RULE = RULE_REGISTRY["first_letter_vowel"]
N_FEWSHOT = 50
N_SENT_INCONTEXT = 3  # sentences per word at the decision point
PROBE_SEEDS = range(5)  # independent held-out-word splits, averaged
TEST_FRAC = 0.3
C_REG = 0.1  # L2 strength for the logistic-regression probe
CACHE = ".cache"

# Balanced 2x2: vowel/consonant first letter x content/function word. 20 words per cell.
WORDS = {
    "vowel_content": [
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
    ],
    "vowel_function": [
        "a",
        "an",
        "and",
        "as",
        "at",
        "in",
        "into",
        "is",
        "it",
        "if",
        "of",
        "off",
        "on",
        "onto",
        "or",
        "our",
        "us",
        "are",
        "am",
        "all",
    ],
    "cons_content": [
        "dog",
        "table",
        "black",
        "wolf",
        "river",
        "peach",
        "green",
        "mountain",
        "hammer",
        "bottle",
        "silver",
        "candle",
        "garden",
        "pencil",
        "ladder",
        "window",
        "basket",
        "rocket",
        "tiger",
        "forest",
    ],
    "cons_function": [
        "the",
        "to",
        "with",
        "by",
        "for",
        "was",
        "that",
        "this",
        "but",
        "not",
        "has",
        "can",
        "will",
        "would",
        "should",
        "my",
        "your",
        "his",
        "her",
        "from",
    ],
}
FUNCTION = set(WORDS["vowel_function"]) | set(WORDS["cons_function"])


def is_vowel(w):
    return w[0].lower() in "aeiou"


def is_function(w):
    return w in FUNCTION


def make_item(word, rng):
    rest = [rng.choice(GENERAL) for _ in range(rng.randint(3, 6))]
    return word + " " + " ".join(rest)


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
def hidden_stack(model, tok, text):
    """Residual stream at the last token, every layer: returns [n_layers+1, hidden]."""
    enc = tok(text, return_tensors="pt").to(model.device)
    out = model(**enc, output_hidden_states=True)
    return torch.stack([h[0, -1].float().cpu() for h in out.hidden_states]).numpy()


def probe_per_layer(X, y, words, cell_of):
    """Held-out-word linear probe at every layer. X: [n, L, H]. Returns (mean, std) per
    layer over PROBE_SEEDS independent word-level splits, balanced within each cell."""
    n, L, H = X.shape
    words_by_cell = defaultdict(set)
    for w in words:
        words_by_cell[cell_of[w]].add(w)
    accs = np.zeros((len(list(PROBE_SEEDS)), L))
    for si, seed in enumerate(PROBE_SEEDS):
        rng = random.Random(seed)
        test_words = set()
        for cell, ws in words_by_cell.items():
            wl = sorted(ws)
            rng.shuffle(wl)
            k = max(1, int(round(TEST_FRAC * len(wl))))
            test_words.update(wl[:k])
        tr = [i for i, w in enumerate(words) if w not in test_words]
        te = [i for i, w in enumerate(words) if w in test_words]
        for l in range(L):
            scaler = StandardScaler().fit(X[tr, l, :])
            clf = LogisticRegression(C=C_REG, max_iter=2000, class_weight="balanced")
            clf.fit(scaler.transform(X[tr, l, :]), y[tr])
            accs[si, l] = clf.score(scaler.transform(X[te, l, :]), y[te])
    return accs.mean(0), accs.std(0)


def report(title, mean, std):
    best = int(np.argmax(mean))
    print(
        f"\n  {title}: best layer {best} = {mean[best]:.0%} (+/-{std[best]:.0%}); "
        f"final layer {mean[-1]:.0%}; chance 50%"
    )
    # compact per-layer line so the whole curve is visible
    step = max(1, len(mean) // 16)
    cells = [f"L{l}:{mean[l]:.0%}" for l in range(0, len(mean), step)]
    print("    " + "  ".join(cells))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    os.makedirs(CACHE, exist_ok=True)

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

    all_words = [(w, cell) for cell, ws in WORDS.items() for w in ws]
    cell_of = {w: cell for cell, ws in WORDS.items() for w in ws}

    # ---- BASELINE: bare word, read its own representation ----
    print("\n===== BASELINE: bare-word probe for first-letter-vowel =====", flush=True)
    Xb, wb = [], []
    for i, (w, _) in enumerate(all_words):
        Xb.append(hidden_stack(model, tok, w))
        wb.append(w)
        if (i + 1) % 20 == 0:
            print(f"  baseline {i + 1}/{len(all_words)}", flush=True)
    Xb = np.stack(Xb)
    yb = np.array([is_vowel(w) for w in wb])
    np.savez(f"{CACHE}/gemma_probe_baseline.npz", X=Xb, words=np.array(wb), y_vowel=yb)
    mean, std = probe_per_layer(Xb, yb, wb, cell_of)
    report("vowel feature (bare word)", mean, std)

    # ---- IN-CONTEXT: decision-point representation, both cues ----
    print(
        "\n===== IN-CONTEXT: answer-position probe (few-shot prompt) =====", flush=True
    )
    fewshot = sample_balanced(RULE, N_FEWSHOT, seed=0)
    rng = random.Random(0)
    Xi, wi = [], []
    total = len(all_words) * N_SENT_INCONTEXT
    for i, (w, _) in enumerate(all_words):
        for _ in range(N_SENT_INCONTEXT):
            query = make_item(w, rng)
            Xi.append(hidden_stack(model, tok, build_prompt(fewshot, query)))
            wi.append(w)
        if (i + 1) % 10 == 0:
            print(f"  in-context {len(wi)}/{total}", flush=True)
    Xi = np.stack(Xi)
    yv = np.array([is_vowel(w) for w in wi])
    yf = np.array([is_function(w) for w in wi])
    np.savez(
        f"{CACHE}/gemma_probe_incontext.npz",
        X=Xi,
        words=np.array(wi),
        y_vowel=yv,
        y_function=yf,
    )
    mv, sv = probe_per_layer(Xi, yv, wi, cell_of)
    mf, sf = probe_per_layer(Xi, yf, wi, cell_of)
    report("intended rule: first letter is a vowel", mv, sv)
    report("substitute: first word is a function word", mf, sf)


if __name__ == "__main__":
    main()
