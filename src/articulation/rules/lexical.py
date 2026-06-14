"""Semantic / lexical rules. Predicted: classify OK, articulate OK."""

from ..inputs import ANIMALS, free_sentence, sentence_maybe_number
from . import Rule, register

_ANIMALS = set(ANIMALS)

register(
    Rule(
        key="contains_number",
        category="lexical",
        human_articulation="The input is labeled True if and only if it contains a digit (0-9).",
        input_space=sentence_maybe_number,
        labeler=lambda s: any(c.isdigit() for c in s),
    )
)

register(
    Rule(
        key="contains_animal",
        category="lexical",
        human_articulation="The input is labeled True if and only if it contains an animal word.",
        input_space=free_sentence,
        labeler=lambda s: any(t in _ANIMALS for t in s.split()),
    )
)
