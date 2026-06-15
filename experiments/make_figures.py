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


# --- Figure 0: HERO. In-distribution accuracy vs counterfactual genuineness -----
# in-dist = Step-1 accuracy; genuineness = behaviour matching the INTENDED rule on
# discriminating inputs (first_word_verb/is_question/conjunction genuine; the two
# first-letter rules proxy). hunt_first_letter_ae genuineness is the pooled disc
# metric (51%); its novel-content generalisation alone is only 12%.
def fig_hero():
    rules = [
        "first_word\n_verb",
        "is_question",
        "conjunction\n(digit&lower)",
        "first_letter\n_vowel",
        "hunt_first\n_letter_ae",
    ]
    indist = [0.90, 1.00, 0.96, 0.88, 1.00]
    genuine = [0.97, 0.97, 1.00, 0.37, 0.51]
    x = range(len(rules))
    w = 0.38
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.bar(
        [i - w / 2 for i in x],
        indist,
        w,
        label="in-distribution accuracy (Step 1)",
        color="#c4c4c4",
    )
    ax.bar(
        [i + w / 2 for i in x],
        genuine,
        w,
        label="counterfactual genuineness (behaviour matches INTENDED rule)",
        color=["#55A868"] * 3 + ["#C44E52"] * 2,
    )
    ax.axhline(0.50, ls=":", color="grey", lw=1)
    ax.text(4.35, 0.52, "chance", color="grey", fontsize=8)
    ax.axvline(2.5, ls="--", color="black", lw=0.8)
    ax.text(
        1.0,
        1.07,
        "GENUINE  (semantic feature)",
        ha="center",
        fontsize=9,
        color="#3a7a4f",
    )
    ax.text(
        3.5, 1.07, "PROXY  (sub-symbolic)", ha="center", fontsize=9, color="#8a3438"
    )
    ax.set_xticks(list(x))
    ax.set_xticklabels(rules, fontsize=8)
    ax.set_ylim(0, 1.16)
    ax.set_ylabel("accuracy / genuineness")
    ax.set_title(
        "In-distribution accuracy hides which rule was learned;\nthe counterfactual reveals it"
    )
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig0_hero.png"), dpi=130)
    plt.close(fig)


# --- Figure 6: naive classify-vs-articulate scatter (the "glimpse") ------------
# x = Step-1 classification accuracy; y = fraction of 6 single-shot (no-CoT)
# articulations, on 6 independent example sets, that state the INTENDED rule.
# first_letter_vowel 2/6, hunt_first_letter_ae 1/6; everything else 6/6.
def fig_scatter():
    from matplotlib.patches import Rectangle

    pts = [
        ("contains_number", "lexical", 1.00, 1.00, -0.015),
        ("all_lowercase", "character", 1.00, 1.00, 0.0),
        ("is_question", "syntactic", 1.00, 1.00, 0.015),
        ("conjunction", "composite", 0.96, 1.00, 0.0),
        ("first_word_verb", "syntactic", 0.90, 1.00, 0.0),
        ("first_letter_vowel", "character", 0.88, 2 / 6, 0.0),
        ("hunt_first_letter_ae", "character", 1.00, 1 / 6, 0.0),
    ]
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.add_patch(
        Rectangle((0.872, -0.05), 0.183, 0.55, facecolor="#f3d9d9", alpha=0.4, zorder=0)
    )
    ax.text(
        0.975,
        0.20,
        "low score here is misleading:\nthese two rules were learned as a\nsimpler rule (see the hero figure)",
        fontsize=8,
        color="#8a3438",
        ha="center",
    )
    for name, cat, xc, yc, jit in pts:
        ax.scatter(
            xc + jit, yc, s=95, color=CAT_COLOR[cat], edgecolor="black", zorder=3
        )
    ax.annotate(
        "5 genuine rules\n(classify ✓ + articulate intended ✓)",
        (1.0, 1.0),
        xytext=(0.905, 0.74),
        fontsize=8,
        ha="center",
        arrowprops=dict(arrowstyle="->", color="grey"),
    )
    ax.annotate("first_letter_vowel", (0.88, 0.33), xytext=(0.885, 0.42), fontsize=8)
    ax.annotate("hunt_first_letter_ae", (1.0, 0.0), xytext=(0.9, 0.06), fontsize=8)
    handles = [
        plt.Line2D([], [], marker="o", ls="", color=c, label=k)
        for k, c in CAT_COLOR.items()
    ]
    ax.legend(handles=handles, fontsize=7, loc="center left", title="category")
    ax.set_xlim(0.85, 1.05)
    ax.set_ylim(-0.05, 1.12)
    ax.set_xlabel("classification accuracy (Step 1)")
    ax.set_ylabel(
        "fraction of 6 single-shot articulations\nthat state the intended rule"
    )
    ax.set_title("Naive landscape: classify vs articulate, per rule")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig6_scatter.png"), dpi=130)
    plt.close(fig)


# --- Figure 7: the substitute is controllable (Experiments 19-20) --------------
# behaviour matching the intended first-letter-vowel rule, on disagreeing inputs.
def fig_controllable():
    labels = [
        "normal\ntraining",
        "decorrelated\ntraining",
        "normal +\nno reasoning",
        "normal +\nreasoning",
    ]
    vals = [0.39, 0.95, 0.38, 0.91]
    colors = ["#C44E52", "#55A868", "#C44E52", "#55A868"]
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    bars = ax.bar(labels, vals, color=colors)
    ax.axhline(0.5, ls=":", color="grey", lw=1)
    ax.text(3.45, 0.52, "chance", color="grey", fontsize=8)
    ax.axvline(1.5, ls="--", color="black", lw=0.8)
    ax.text(0.5, 1.05, "change the DATA", ha="center", fontsize=9, color="#444")
    ax.text(2.5, 1.05, "change the INFERENCE", ha="center", fontsize=9, color="#444")
    for b, v in zip(bars, vals):
        ax.text(
            b.get_x() + b.get_width() / 2,
            v + 0.02,
            f"{v:.0%}",
            ha="center",
            fontsize=10,
        )
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("behaviour matches the intended\n(first-letter-vowel) rule")
    ax.set_title(
        "The substitute is controllable: removing the shortcut,\nor allowing reasoning, recovers the real rule"
    )
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig7_controllable.png"), dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    fig_hero()
    fig_sweep()
    fig_salience()
    fig_scaling()
    fig_faithfulness()
    fig_contrast()
    fig_scatter()
    fig_controllable()
    print(f"wrote 8 figures to {os.path.abspath(OUT)}")
