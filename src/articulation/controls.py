"""Confound controls.

C1 shuffled-label: permute few-shot labels. If classification accuracy stays
high, the model is using priors, not learning in-context -> premise breaks.

C2 zero-shot / prior-only: call classify_one with examples=[] (handled in model).

C7 prior-only articulation: ask for the rule with shuffled-label (or empty)
examples; if it still "articulates" the rule, articulation is contaminated by
priors rather than learned in-context.
"""

from __future__ import annotations

import random


def shuffled_labels(
    examples: list[tuple[str, bool]], seed: int = 0
) -> list[tuple[str, bool]]:
    rng = random.Random(seed)
    labels = [y for _, y in examples]
    rng.shuffle(labels)
    return [(x, labels[i]) for i, (x, _) in enumerate(examples)]
