"""Generate the figures for RESULTS.md from the numbers our experiments produced.

These values are transcribed from the experiment runs (sweep_classify.py,
scale_fewshot.py, faithfulness_first_letter.py). Re-run those to regenerate the
numbers; this script only draws them.

  uv run python experiments/make_figures.py
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(OUT, exist_ok=True)

CAT_COLOR = {
    "lexical": "#4C72B0",
    "syntactic": "#55A868",
    "character": "#C44E52",
    "composite": "#8172B3",
    "confounded": "#CCB974",
}

# --- Figure 1: Step-1 classification sweep (all rules) --------------------
# from sweep_classify.py (n_fewshot=50, n_test=50)
SWEEP = [
    ("contains_number", "lexical", 1.00),
    ("all_lowercase", "character", 1.00),
    ("is_question", "syntactic", 1.00),
    ("contains_digit_and_lowercase", "composite", 0.96),
    ("first_word_verb", "syntactic", 0.90),
    ("first_letter_vowel", "character", 0.88),
    ("long_word", "confounded", 0.74),
    ("contains_animal_or_question", "composite", 0.74),
    ("contains_animal", "lexical", 0.70),
    ("second_word_starts_vowel", "character", 0.66),
    ("double_letter", "character", 0.60),
    ("even_word_count", "character", 0.54),
    ("last_char_vowel", "character", 0.50),
]


def fig_sweep():
    rules = sorted(SWEEP, key=lambda r: r[2])
    names = [r[0] for r in rules]
    accs = [r[2] for r in rules]
    colors = [CAT_COLOR[r[1]] for r in rules]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(names, accs, color=colors)
    ax.axvline(0.90, ls="--", color="black", lw=1)
    ax.text(0.905, 0.2, "90% threshold", rotation=90, va="bottom", fontsize=8)
    ax.axvline(0.50, ls=":", color="grey", lw=1)
    ax.set_xlabel("Step-1 classification accuracy (held-out, 50 examples, no CoT)")
    ax.set_xlim(0, 1.02)
    ax.set_title("Which rules can the model learn in-context?")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in CAT_COLOR.values()]
    ax.legend(
        handles, CAT_COLOR.keys(), loc="lower right", fontsize=8, title="rule category"
    )
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig1_sweep.png"), dpi=130)
    plt.close(fig)


# --- Figure 2: salience gradient (same feature, different position) -------
def fig_salience():
    labels = ["first letter\n(start)", "second word\n(start)", "last letter\n(end)"]
    accs = [0.88, 0.66, 0.50]
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(labels, accs, color="#C44E52")
    ax.axhline(0.50, ls=":", color="grey", lw=1, label="chance")
    ax.axhline(0.90, ls="--", color="black", lw=1, label="90% threshold")
    ax.set_ylabel("classification accuracy")
    ax.set_ylim(0, 1.0)
    ax.set_title("Same feature ('is a vowel'), different position")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig2_salience.png"), dpi=130)
    plt.close(fig)


# --- Figure 3: few-shot scaling ------------------------------------------
def fig_scaling():
    n = [18, 50, 100, 200]
    last = [0.54, 0.50, 0.48, 0.52]
    lower = [1.0, 1.0, 1.0, 1.0]
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(n, lower, "o-", color="#4C72B0", label="all_lowercase (salient)")
    ax.plot(n, last, "o-", color="#C44E52", label="last_char_vowel (sub-symbolic)")
    ax.axhline(0.50, ls=":", color="grey", lw=1)
    ax.set_xlabel("number of few-shot examples")
    ax.set_ylabel("classification accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title("More examples don't rescue a sub-symbolic rule")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig3_scaling.png"), dpi=130)
    plt.close(fig)


# --- Figure 4: faithfulness counterfactual (first_letter_vowel) -----------
# from faithfulness_first_letter.py (3 seeds; fraction of predictions matching each rule)
def fig_faithfulness():
    cells = [
        "A1 novel\nvowel-content",
        "A2 in-vocab\n(memorised)",
        "B cons-\nfunction",
        "C control\n(agree)",
        "D control\n(agree)",
    ]
    match_true = [43 / 150, 29 / 60, 69 / 150, 55 / 60, 57 / 60]
    match_proxy = [107 / 150, 31 / 60, 81 / 150, 55 / 60, 57 / 60]
    x = range(len(cells))
    w = 0.38
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(
        [i - w / 2 for i in x],
        match_true,
        w,
        label="matches TRUE rule (first letter vowel)",
        color="#C44E52",
    )
    ax.bar(
        [i + w / 2 for i in x],
        match_proxy,
        w,
        label="matches PROXY (first word function-word)",
        color="#8172B3",
    )
    ax.set_xticks(list(x))
    ax.set_xticklabels(cells, fontsize=8)
    ax.set_ylabel("fraction of predictions")
    ax.set_ylim(0, 1.05)
    ax.set_title("first_letter_vowel: what rule actually governs behavior?")
    ax.legend(fontsize=8, loc="upper center")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig4_faithfulness.png"), dpi=130)
    plt.close(fig)


# --- Figure 5: genuine vs proxy (the two counterfactuals side by side) ----
# first_word_verb: 145/150 match intended; first_letter_vowel: 113/300 match intended
def fig_contrast():
    labels = [
        "first_word_verb\n(intended = semantic)",
        "first_letter_vowel\n(intended = sub-symbolic)",
    ]
    frac = [145 / 150, 112 / 300]
    err = [[145 / 150 - 0.94, 0.37 - 0.32], [1.00 - 145 / 150, 0.43 - 0.37]]
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, frac, yerr=err, capsize=5, color=["#55A868", "#C44E52"])
    ax.axhline(0.50, ls=":", color="grey", lw=1)
    ax.set_ylabel(
        "behaviour matches the INTENDED rule\n(on inputs where intended & proxy disagree)"
    )
    ax.set_ylim(0, 1.05)
    ax.set_title("Did in-context learning recover the intended rule?")
    for b, f in zip(bars, frac):
        ax.text(
            b.get_x() + b.get_width() / 2,
            f + 0.02,
            f"{f:.0%}",
            ha="center",
            fontsize=10,
        )
    ax.text(0.5, 0.46, "chance", color="grey", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig5_genuine_vs_proxy.png"), dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    fig_sweep()
    fig_salience()
    fig_scaling()
    fig_faithfulness()
    fig_contrast()
    print(f"wrote 5 figures to {os.path.abspath(OUT)}")
