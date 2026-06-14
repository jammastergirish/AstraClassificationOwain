"""Character / positional / counting rules. The interesting zone: predicted to
classify well but be hard for the model to articulate (tokenization smears the
relevant features)."""

from ..inputs import free_sentence, maybe_uppercase
from . import Rule, register

_VOWELS = set("aeiou")


def _last_alpha_is_vowel(s: str) -> bool:
    for ch in reversed(s):
        if ch.isalpha():
            return ch.lower() in _VOWELS
    return False


def _has_double_letter(s: str) -> bool:
    for tok in s.split():
        for i in range(len(tok) - 1):
            if tok[i] == tok[i + 1] and tok[i].isalpha():
                return True
    return False


def _first_letter_is_vowel(s: str) -> bool:
    for ch in s:
        if ch.isalpha():
            return ch.lower() in _VOWELS
    return False


def _second_word_starts_vowel(s: str) -> bool:
    words = s.split()
    if len(words) < 2:
        return False
    for ch in words[1]:
        if ch.isalpha():
            return ch.lower() in _VOWELS
    return False


register(
    Rule(
        key="all_lowercase",
        category="character",
        human_articulation="The input is labeled True if and only if it is entirely lowercase.",
        input_space=maybe_uppercase,
        labeler=lambda s: s == s.lower() and any(c.isalpha() for c in s),
    )
)

register(
    Rule(
        key="last_char_vowel",
        category="character",
        human_articulation="The input is labeled True if and only if its last alphabetic character is a vowel.",
        input_space=free_sentence,
        labeler=_last_alpha_is_vowel,
    )
)

register(
    Rule(
        key="even_word_count",
        category="character",
        human_articulation="The input is labeled True if and only if it has an even number of words.",
        input_space=free_sentence,
        labeler=lambda s: len(s.split()) % 2 == 0,
    )
)

register(
    Rule(
        key="double_letter",
        category="character",
        human_articulation="The input is labeled True if and only if some word contains a doubled letter (e.g. ll, ee).",
        input_space=free_sentence,
        labeler=_has_double_letter,
    )
)

register(
    Rule(
        key="first_letter_vowel",
        category="character",
        human_articulation="The input is labeled True if and only if its first letter is a vowel.",
        input_space=free_sentence,
        labeler=_first_letter_is_vowel,
    )
)

register(
    Rule(
        key="second_word_starts_vowel",
        category="character",
        human_articulation="The input is labeled True if and only if its second word starts with a vowel.",
        input_space=free_sentence,
        labeler=_second_word_starts_vowel,
    )
)
