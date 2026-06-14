"""Isolation probe: does the model HAVE the concept "first letter is a vowel"
when asked directly, outside the classification framing? If it answers correctly
in isolation but does not learn/deploy the rule in-context (where it falls back on
a function-word proxy), then the failure is one of in-context deployment, not
missing knowledge.

  uv run python experiments/isolation_probe.py
"""

from __future__ import annotations

from articulation import cache, model

WORDS_VOWEL = [
    "orange",
    "igloo",
    "apple",
    "eagle",
    "otter",
    "umbrella",
    "oasis",
    "engine",
    "island",
    "acorn",
    "old",
    "onion",
]
WORDS_CONS = [
    "dog",
    "table",
    "by",
    "the",
    "with",
    "wolf",
    "river",
    "peach",
    "green",
    "mountain",
    "black",
    "for",
]

SYSTEM = "Answer with exactly one word: Yes or No. Do not explain."


def truth(word: str) -> bool:
    return word[0].lower() in "aeiou"


def ask(word: str) -> bool | None:
    q = f'Does the word "{word}" start with a vowel (a, e, i, o, u)? Answer Yes or No.'
    payload = {
        "op": "isolation_first_letter",
        "model": model.SUBJECT_MODEL,
        "system": SYSTEM,
        "q": q,
    }
    hit = cache.get(payload)
    if hit is not None:
        return hit["ans"]
    resp = model.client().messages.create(
        model=model.SUBJECT_MODEL,
        max_tokens=5,
        system=[{"type": "text", "text": SYSTEM}],
        messages=[{"role": "user", "content": q}],
    )
    raw = "".join(b.text for b in resp.content if b.type == "text").strip().lower()
    ans = True if raw.startswith("yes") else False if raw.startswith("no") else None
    cache.put(payload, {"ans": ans, "raw": raw})
    return ans


def main():
    correct = total = 0
    for w in WORDS_VOWEL + WORDS_CONS:
        ans = ask(w)
        ok = ans == truth(w)
        correct += ok
        total += 1
        print(
            f"{w:<12} model-says-vowel={str(ans):<5} truth={str(truth(w)):<5} {'OK' if ok else 'WRONG'}"
        )
    print(f"\nIsolation accuracy: {correct}/{total} = {correct / total:.0%}")


if __name__ == "__main__":
    main()
