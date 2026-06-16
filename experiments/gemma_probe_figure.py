"""Plot the Gemma linear-probe layer curves from the cached activations (no model needed).

Reads .cache/gemma_probe_{baseline,incontext}.npz written by gemma_probe.py, re-runs the
held-out-word probe per layer (deterministic given the same seeds), and draws three
curves: the baseline vowel probe (methodology check), and the in-context probes for the
intended rule and the function-word substitute at the decision point.

  uv run python experiments/gemma_probe_figure.py
"""

from __future__ import annotations

import random
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

PROBE_SEEDS = range(5)
TEST_FRAC = 0.3
C_REG = 0.1

# The same function-word set gemma_probe.py used, to reconstruct each word's 2x2 cell.
FUNCTION = {
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
}


def cell_of(w):
    v = "V" if w[0].lower() in "aeiou" else "C"
    f = "func" if w in FUNCTION else "content"
    return f"{v}_{f}"


def probe_per_layer(X, y, words):
    n, L, H = X.shape
    words_by_cell = defaultdict(set)
    for w in words:
        words_by_cell[cell_of(w)].add(w)
    accs = np.zeros((len(list(PROBE_SEEDS)), L))
    for si, seed in enumerate(PROBE_SEEDS):
        rng = random.Random(seed)
        test_words = set()
        for ws in words_by_cell.values():
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
    return accs.mean(0)


def main():
    base = np.load(".cache/gemma_probe_baseline.npz", allow_pickle=True)
    inc = np.load(".cache/gemma_probe_incontext.npz", allow_pickle=True)

    bw = [str(w) for w in base["words"]]
    iw = [str(w) for w in inc["words"]]
    base_vowel = probe_per_layer(base["X"], base["y_vowel"], bw)
    inc_vowel = probe_per_layer(inc["X"], inc["y_vowel"], iw)
    inc_func = probe_per_layer(inc["X"], inc["y_function"], iw)

    layers = np.arange(len(inc_vowel))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axhline(50, ls=":", c="grey", lw=1, label="chance (50%)")
    ax.plot(
        layers,
        100 * base_vowel,
        "-o",
        ms=3,
        c="#444",
        label="baseline: vowel, bare word (probe works)",
    )
    ax.plot(
        layers,
        100 * inc_vowel,
        "-o",
        ms=3,
        c="#1f77b4",
        label="in-context: intended rule (first letter is a vowel)",
    )
    ax.plot(
        layers,
        100 * inc_func,
        "-o",
        ms=3,
        c="#d62728",
        label="in-context: substitute (first word is a function word)",
    )
    ax.set_xlabel("layer (residual stream, 0 = embeddings)")
    ax.set_ylabel("held-out-word probe accuracy (%)")
    ax.set_ylim(45, 102)
    ax.set_title("Linear probes on Gemma 4 31B: where each rule is written down")
    ax.legend(loc="lower center", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    out = "figures/fig8_gemma_probe.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")
    print(
        f"baseline vowel  best {100 * base_vowel.max():.0f}% (L{base_vowel.argmax()})"
    )
    print(
        f"in-context vowel best {100 * inc_vowel.max():.0f}% (L{inc_vowel.argmax()}), final {100 * inc_vowel[-1]:.0f}%"
    )
    print(
        f"in-context func  best {100 * inc_func.max():.0f}% (L{inc_func.argmax()}), final {100 * inc_func[-1]:.0f}%"
    )


if __name__ == "__main__":
    main()
