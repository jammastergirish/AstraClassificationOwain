"""Task 3: candidate sub-symbolic rules, hunting for a second learnable-but-
sub-symbolic case to replicate the `first_letter_vowel` proxy phenomenon. Most are
expected to fail Step 1 (that's fine); we want the rare one at a salient position
that the model can learn."""

from ..inputs import free_sentence
from . import Rule, register

_VOWELS = set("aeiou")
_FIRST_HALF = set("abcdefghijklm")


def _first_alpha(s: str) -> str | None:
    for ch in s:
        if ch.isalpha():
            return ch.lower()
    return None


register(
    Rule(
        key="hunt_first_letter_ae",
        category="character",
        human_articulation="The input is labeled True iff its first letter is 'a' or 'e'.",
        input_space=free_sentence,
        labeler=lambda s: _first_alpha(s) in {"a", "e"},
    )
)

register(
    Rule(
        key="hunt_first_letter_first_half",
        category="character",
        human_articulation="The input is labeled True iff its first letter is in the first half of the alphabet (a-m).",
        input_space=free_sentence,
        labeler=lambda s: _first_alpha(s) in _FIRST_HALF,
    )
)

register(
    Rule(
        key="hunt_first_word_short",
        category="character",
        human_articulation="The input is labeled True iff its first word is at most 3 letters long.",
        input_space=free_sentence,
        labeler=lambda s: bool(s.split()) and len(s.split()[0]) <= 3,
    )
)

register(
    Rule(
        key="hunt_last_word_first_vowel",
        category="character",
        human_articulation="The input is labeled True iff its last word starts with a vowel.",
        input_space=free_sentence,
        labeler=lambda s: bool(s.split()) and s.split()[-1][:1].lower() in _VOWELS,
    )
)
