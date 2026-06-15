"""Claude API primitives for the subject model.

Step 1 (classify): NO chain-of-thought. We omit `thinking` (off by default on
Opus 4.8), force a single-word answer, and cap `max_tokens` so the model cannot
reason in the visible response. The instruction is rule-neutral so it cannot leak
the rule into the Step 2 articulation (confound C8).

Step 2 (articulate): chain-of-thought is allowed and is one of the variables to
sweep later. This scaffold uses a plain (no-CoT) free-form articulation.
"""

from __future__ import annotations

import os

from anthropic import Anthropic
from dotenv import load_dotenv

from . import cache

load_dotenv()

SUBJECT_MODEL = "claude-opus-4-8"

_client: Anthropic | None = None


def client() -> Anthropic:
    global _client
    if _client is None:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError(
                "ANTHROPIC_API_KEY not set. Copy .env.example to .env and add your key."
            )
        # Higher max_retries so transient 429s (e.g. brief concurrency spikes)
        # back off and recover instead of killing a run.
        _client = Anthropic(max_retries=8)
    return _client


CLASSIFY_SYSTEM = (
    "You are a careful classifier. You will be shown labeled examples that all "
    "follow a single hidden rule mapping each input string to True or False. "
    "Infer the rule from the examples and apply it to the final input. "
    "Respond with exactly one word, either True or False, and nothing else. "
    "Do not explain your reasoning."
)

CLASSIFY_COT_SYSTEM = (
    "You are shown labeled examples that all follow a single hidden rule mapping each "
    "input string to True or False. Reason step by step to work out the rule, then "
    "apply it to the final input. End your reply with a final line in exactly this "
    "format:\nANSWER: True\nor\nANSWER: False"
)

ARTICULATE_SYSTEM_COT = (
    "You are shown labeled examples that all follow a single hidden rule mapping "
    "each input string to True or False. Work out the rule. You may reason briefly "
    "first. Then end your reply with a single final line in exactly this format:\n"
    "RULE: <one sentence stating the rule>\n"
    "The rule must correctly predict the label of every example shown."
)

ARTICULATE_SYSTEM_NOCOT = (
    "You are shown labeled examples that all follow a single hidden rule mapping "
    "each input string to True or False. Respond with exactly one line, and "
    "nothing else, in this format:\n"
    "RULE: <one sentence stating the rule>\n"
    "Do not reason, explain, or show any work. Give only the rule line."
)


def _fewshot_block(examples: list[tuple[str, bool]]) -> str:
    lines = ["Here are labeled examples:"]
    for x, y in examples:
        lines.append(f'Input: "{x}"  Label: {y}')
    return "\n".join(lines)


def _text(resp) -> str:
    return "".join(b.text for b in resp.content if b.type == "text").strip()


def _parse_label(raw: str) -> bool | None:
    t = raw.strip().lower()
    if t.startswith("true"):
        return True
    if t.startswith("false"):
        return False
    return None


def _parse_answer(raw: str) -> bool | None:
    """Pull the final `ANSWER: True/False` line out of a chain-of-thought reply."""
    for line in reversed(raw.splitlines()):
        s = line.strip().upper()
        if s.startswith("ANSWER:"):
            v = s[len("ANSWER:") :].strip()
            if v.startswith("TRUE"):
                return True
            if v.startswith("FALSE"):
                return False
    return None


def classify_one(
    examples: list[tuple[str, bool]], query: str, model: str = SUBJECT_MODEL
) -> tuple[bool | None, str]:
    """Single-token, no-CoT classification of `query` given few-shot `examples`.

    Pass `examples=[]` for the zero-shot / prior-only baseline (confound C2).
    Returns (predicted_label_or_None, raw_text).
    """
    prefix = _fewshot_block(examples) if examples else "No examples are provided."
    query_block = (
        f"\nNow classify the next input. Respond with only True or False.\n"
        f'Input: "{query}"  Label:'
    )
    payload = {
        "op": "classify",
        "model": model,
        "system": CLASSIFY_SYSTEM,
        "prefix": prefix,
        "query": query_block,
    }
    hit = cache.get(payload)
    if hit is not None:
        return hit["label"], hit["raw"]

    resp = client().messages.create(
        model=model,
        max_tokens=5,
        system=[{"type": "text", "text": CLASSIFY_SYSTEM}],
        messages=[
            {
                "role": "user",
                "content": [
                    # Stable prefix first so API prompt caching can reuse it across the
                    # held-out test set; the varying query goes last (uncached).
                    {
                        "type": "text",
                        "text": prefix,
                        "cache_control": {"type": "ephemeral"},
                    },
                    {"type": "text", "text": query_block},
                ],
            }
        ],
    )
    raw = _text(resp)
    label = _parse_label(raw)
    cache.put(payload, {"label": label, "raw": raw})
    return label, raw


def classify_cot(
    examples: list[tuple[str, bool]],
    query: str,
    model: str = SUBJECT_MODEL,
    max_tokens: int = 1200,
) -> tuple[bool | None, str]:
    """Chain-of-thought classification: the model may reason, then gives a final
    ANSWER line. Returns (predicted_label_or_None, raw_text)."""
    prefix = _fewshot_block(examples) if examples else "No examples are provided."
    query_block = (
        f'\nNow classify the next input.\nInput: "{query}"\n'
        "Reason step by step, then end with ANSWER: True or ANSWER: False."
    )
    payload = {
        "op": "classify_cot",
        "model": model,
        "system": CLASSIFY_COT_SYSTEM,
        "prefix": prefix,
        "query": query_block,
    }
    hit = cache.get(payload)
    if hit is not None:
        return hit["label"], hit["raw"]

    resp = client().messages.create(
        model=model,
        max_tokens=max_tokens,
        system=[{"type": "text", "text": CLASSIFY_COT_SYSTEM}],
        messages=[{"role": "user", "content": prefix + query_block}],
    )
    raw = _text(resp)
    label = _parse_answer(raw)
    cache.put(payload, {"label": label, "raw": raw})
    return label, raw


def articulate_one(
    examples: list[tuple[str, bool]],
    cot: bool = True,
    model: str = SUBJECT_MODEL,
    max_tokens: int | None = None,
) -> str:
    """Free-form articulation of the rule from `examples`. Returns the raw text.

    cot=True  : may reason, then a final RULE: line (default).
    cot=False : single-shot RULE: line, no reasoning -- the cleaner test of
                whether the model can *state* the rule it uses (matches Step 1's
                no-reasoning regime).
    """
    system = ARTICULATE_SYSTEM_COT if cot else ARTICULATE_SYSTEM_NOCOT
    if max_tokens is None:
        max_tokens = 2000 if cot else 100
    prefix = _fewshot_block(examples)
    payload = {
        "op": "articulate_cot" if cot else "articulate_nocot",
        "model": model,
        "system": system,
        "prefix": prefix,
    }
    hit = cache.get(payload)
    if hit is not None:
        return hit["text"]

    resp = client().messages.create(
        model=model,
        max_tokens=max_tokens,
        system=[{"type": "text", "text": system}],
        messages=[{"role": "user", "content": prefix + "\n\nState the rule."}],
    )
    text = _text(resp)
    cache.put(payload, {"text": text})
    return text


def extract_rule(text: str) -> str | None:
    """Pull the final `RULE: <...>` line out of an articulation, if present."""
    for line in reversed(text.splitlines()):
        s = line.strip()
        if s.upper().startswith("RULE:"):
            return s[len("RULE:") :].strip()
    return None
