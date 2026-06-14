"""Input-space generators and a balanced sampler.

Every rule draws its inputs by calling its own ``input_space(rng)`` repeatedly;
``sample_balanced`` rejection-samples to a 50/50 label split so classification
accuracy can't be inflated by a majority-class artifact (confound C3).

All vocabulary is lowercase so the only source of uppercase is the
``maybe_uppercase`` corruption used by the ``all_lowercase`` rule.
"""

from __future__ import annotations

import random

# --- Vocabulary -----------------------------------------------------------
# Subsets are referenced by labelers (animals, verbs, ...). NOUNS deliberately
# includes some 8+ letter words (for the `long_word` rule) and some
# doubled-letter words (for the `double_letter` rule).
FILLER = [
    "the",
    "a",
    "on",
    "in",
    "is",
    "was",
    "and",
    "of",
    "to",
    "with",
    "at",
    "by",
    "for",
]
ANIMALS = [
    "cat",
    "dog",
    "horse",
    "bird",
    "fish",
    "cow",
    "sheep",
    "wolf",
    "fox",
    "bear",
    "mouse",
    "goat",
    "duck",
    "frog",
    "deer",
]
COLORS = [
    "red",
    "blue",
    "green",
    "yellow",
    "black",
    "white",
    "brown",
    "pink",
    "grey",
    "purple",
]
FRUITS = [
    "apple",
    "banana",
    "cherry",
    "grape",
    "lemon",
    "melon",
    "peach",
    "plum",
    "mango",
    "pear",
]
VERBS = [
    "run",
    "jump",
    "sit",
    "walk",
    "sing",
    "read",
    "sleep",
    "cook",
    "build",
    "throw",
    "climb",
    "dance",
    "write",
    "paint",
    "drive",
]
ADJ = [
    "cold",
    "warm",
    "big",
    "small",
    "fast",
    "slow",
    "bright",
    "dark",
    "quiet",
    "loud",
    "soft",
    "hard",
    "old",
    "new",
    "clean",
]
NOUNS = [
    "house",
    "car",
    "table",
    "river",
    "book",
    "window",
    "garden",
    "street",
    "cloud",
    "stone",
    "field",
    "bridge",
    "forest",
    "city",
    # 8+ letter words (long_word rule):
    "mountain",
    "umbrella",
    "elephant",
    "computer",
    "hospital",
    "triangle",
    "sandwich",
    # doubled-letter words (double_letter rule):
    "summer",
    "valley",
    "coffee",
    "rabbit",
    "ladder",
    "mirror",
    "kitten",
]
DIGITS = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "21", "42", "7"]

GENERAL = FILLER + NOUNS + VERBS + ADJ + ANIMALS + COLORS + FRUITS


# --- Generators (each takes a seeded random.Random, returns one input) -----
def _sentence(rng: random.Random, lo: int = 4, hi: int = 8) -> str:
    length = rng.randint(lo, hi)
    return " ".join(rng.choice(GENERAL) for _ in range(length))


def free_sentence(rng: random.Random) -> str:
    return _sentence(rng)


def five_word_sentence(rng: random.Random) -> str:
    """Fixed-format space: exactly 5 words. Use for positional rules."""
    return " ".join(rng.choice(GENERAL) for _ in range(5))


def sentence_maybe_number(rng: random.Random) -> str:
    toks = _sentence(rng).split()
    if rng.random() < 0.45:
        toks.insert(rng.randint(0, len(toks)), rng.choice(DIGITS))
    return " ".join(toks)


def maybe_uppercase(rng: random.Random) -> str:
    s = _sentence(rng)
    r = rng.random()
    if r < 0.34:
        return s.upper()
    if r < 0.50:
        toks = s.split()
        i = rng.randrange(len(toks))
        toks[i] = toks[i].capitalize()
        return " ".join(toks)
    return s


def maybe_question(rng: random.Random) -> str:
    s = _sentence(rng)
    return s + "?" if rng.random() < 0.5 else s


def sentence_number_and_case(rng: random.Random) -> str:
    """Digit-presence and case vary independently -> all four cells for the
    'contains a digit AND is all-lowercase' conjunction."""
    toks = _sentence(rng).split()
    if rng.random() < 0.5:
        toks.insert(rng.randint(0, len(toks)), rng.choice(DIGITS))
    s = " ".join(toks)
    if rng.random() < 0.5:  # corrupt case
        if rng.random() < 0.5:
            s = s.upper()
        else:
            t = s.split()
            i = rng.randrange(len(t))
            t[i] = t[i].capitalize()
            s = " ".join(t)
    return s


# --- Balanced sampling ----------------------------------------------------
def sample_balanced(rule, n: int, seed: int):
    """Return n (input, label) pairs, 50/50 True/False, deterministic in seed."""
    rng = random.Random(seed)
    target_pos = n // 2
    target_neg = n - target_pos
    pos: list[str] = []
    neg: list[str] = []
    cap = n * 2000
    tries = 0
    while len(pos) < target_pos or len(neg) < target_neg:
        tries += 1
        if tries > cap:
            raise RuntimeError(
                f"could not balance rule '{rule.key}' "
                f"(pos={len(pos)}, neg={len(neg)}, want {target_pos}/{target_neg})"
            )
        x = rule.input_space(rng)
        y = rule.labeler(x)
        if y and len(pos) < target_pos:
            pos.append(x)
        elif (not y) and len(neg) < target_neg:
            neg.append(x)
    items = [(x, True) for x in pos] + [(x, False) for x in neg]
    rng.shuffle(items)
    return items
