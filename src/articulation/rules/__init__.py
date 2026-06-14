"""Rule registry. A Rule is an (input_space, labeler) pair plus metadata.

The labeler is the *intended* rule (ground truth). ``human_articulation`` is the
clean natural-language statement a human would give — used only for reference and
for grading articulations, never shown to the subject model during Step 1 or 2.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class Rule:
    key: str
    category: str  # "lexical" | "syntactic" | "character" | "confounded"
    human_articulation: str
    input_space: Callable[[random.Random], str]
    labeler: Callable[[str], bool]


RULE_REGISTRY: dict[str, Rule] = {}


def register(rule: Rule) -> Rule:
    if rule.key in RULE_REGISTRY:
        raise ValueError(f"duplicate rule key: {rule.key}")
    RULE_REGISTRY[rule.key] = rule
    return rule


# Import submodules to populate the registry. Kept at the bottom so that
# `register` and `Rule` are defined before the submodules import them.
from . import lexical, character, syntactic, confounded, composite, hunt  # noqa: E402,F401
