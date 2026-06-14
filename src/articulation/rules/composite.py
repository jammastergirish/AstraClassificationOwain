"""Boolean-structure rules (Mechanism A): two individually-salient features
combined. The hypothesis is that the model learns the combination (>90% accuracy
on 'A AND B' *requires* using both features, since the False set includes the
A-but-not-B cell) yet articulates only one conjunct."""

from ..inputs import ANIMALS, maybe_question, sentence_number_and_case
from . import Rule, register

_ANIMALS = set(ANIMALS)

register(
    Rule(
        key="contains_digit_and_lowercase",
        category="composite",
        human_articulation="The input is labeled True if and only if it contains a digit AND is entirely lowercase.",
        input_space=sentence_number_and_case,
        labeler=lambda s: any(c.isdigit() for c in s) and s == s.lower(),
    )
)

register(
    Rule(
        key="contains_animal_or_question",
        category="composite",
        human_articulation="The input is labeled True if and only if it contains an animal word OR ends with a question mark.",
        input_space=maybe_question,
        labeler=lambda s: (
            any(t in _ANIMALS for t in s.split()) or s.strip().endswith("?")
        ),
    )
)
