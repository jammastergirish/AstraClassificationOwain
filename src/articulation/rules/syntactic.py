"""Syntactic / structural rules. Predicted: classify OK, articulate ~mixed."""

from ..inputs import VERBS, free_sentence, maybe_question
from . import Rule, register

_VERBS = set(VERBS)

register(
    Rule(
        key="is_question",
        category="syntactic",
        human_articulation="The input is labeled True if and only if it ends with a question mark.",
        input_space=maybe_question,
        labeler=lambda s: s.strip().endswith("?"),
    )
)

register(
    Rule(
        key="first_word_verb",
        category="syntactic",
        human_articulation="The input is labeled True if and only if its first word is a verb.",
        input_space=free_sentence,
        labeler=lambda s: bool(s.split()) and s.split()[0] in _VERBS,
    )
)
