"""Open-weights replication: does an open model also learn the function-word
substitute for `first_letter_vowel`?

This loads a Hugging Face causal LM (Gemma or any other) and classifies our rules by
reading the next-token logits for True vs False, reusing the exact same rules and
balanced sampler as the Claude experiments. It is the open-model analogue of
Experiment 4 (the Step-1 sweep): confirm which rules the model can classify, with the
sub-symbolic ones (first_letter_vowel, last_char_vowel) as the rules of interest.

A 31B model is ~62GB in bf16. It loads straight onto the device, so it runs on a
128GB M-series Mac via MPS as well as on a CUDA GPU. Gemma is gated, so accept its
licence and put a read token in `.env` as HF_TOKEN.

  uv sync --extra gpu
  uv run python experiments/gemma_classify.py --model google/gemma-4-31B-it

Once the behaviour is confirmed, the next step is to extract hidden states at the
answer position (the harness already supports --dump-hidden) and train a linear probe
for the vowel feature, which is where the closed-model story gains a mechanistic one.
"""

from __future__ import annotations

import argparse

import torch
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer

from articulation.inputs import sample_balanced
from articulation.rules import RULE_REGISTRY

load_dotenv()  # picks up HF_TOKEN

# Rules to sweep by default: two the model should learn easily, the conjunction, and
# the two sub-symbolic letter rules that are the whole point.
DEFAULT_RULES = [
    "contains_number",
    "all_lowercase",
    "is_question",
    "first_word_verb",
    "contains_digit_and_lowercase",
    "first_letter_vowel",
    "last_char_vowel",
]


def build_prompt(examples, query):
    lines = [
        "Below are labeled examples that follow a single hidden rule. "
        "Each input is labeled True or False. Infer the rule and label the final input.",
        "",
    ]
    for x, y in examples:
        lines.append(f'Input: "{x}"  Label: {y}')
    lines.append(f'Input: "{query}"  Label:')
    return "\n".join(lines)


def answer_token_ids(tok):
    """First-token ids for the two answers. We try a leading space first, since the
    prompt ends in 'Label:' and the model continues with ' True'/' False'."""

    def first_id(s):
        ids = tok(s, add_special_tokens=False).input_ids
        return ids[0]

    return first_id(" True"), first_id(" False")


@torch.no_grad()
def classify(
    model, tok, true_id, false_id, examples, query, dump_hidden=False, debug=False
):
    text = build_prompt(examples, query)
    enc = tok(text, return_tensors="pt").to(model.device)
    out = model(**enc, output_hidden_states=dump_hidden)
    last = out.logits[0, -1]  # logits for the token that comes after 'Label:'
    pred = last[true_id].item() > last[false_id].item()
    if debug:
        top = last.topk(6)
        toks = [repr(tok.decode([i])) for i in top.indices.tolist()]
        print(
            f"    top tokens: {list(zip(toks, [round(v, 1) for v in top.values.tolist()]))}"
            f"  | ' True'={last[true_id]:.1f} ' False'={last[false_id]:.1f}"
        )
    hidden = None
    if dump_hidden:
        # per-layer residual stream at the answer position
        hidden = torch.stack([h[0, -1].float().cpu() for h in out.hidden_states])
    return pred, hidden


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model", required=True, help="HF repo id, e.g. google/gemma-3-27b-it"
    )
    ap.add_argument("--rules", nargs="*", default=DEFAULT_RULES)
    ap.add_argument("--n-fewshot", type=int, default=50)
    ap.add_argument("--n-test", type=int, default=50)
    ap.add_argument(
        "--dump-hidden", action="store_true", help="(for probing) save hidden states"
    )
    ap.add_argument("--device", default="auto", help="auto | cuda | mps | cpu")
    ap.add_argument(
        "--debug", action="store_true", help="print top tokens for the first few items"
    )
    args = ap.parse_args()

    if args.device == "auto":
        device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
    else:
        device = args.device
    # bf16 everywhere except CPU: a 31B model is ~62GB in bf16 (fits a 128GB Mac
    # via MPS), ~124GB in float32 (does not). Gemma overflows in float16, so bf16,
    # not fp16, is the right half-precision choice.
    dtype = torch.float32 if device == "cpu" else torch.bfloat16

    print(f"loading {args.model} on {device} (dtype={dtype}) ...")
    tok = AutoTokenizer.from_pretrained(args.model)
    # Load weights straight onto the target device. On the Mac this avoids holding a
    # CPU copy and an MPS copy at the same time, which for a 62GB model would risk an
    # out-of-memory peak. "auto" shards across multiple GPUs on a CUDA box.
    load_kwargs = {
        "dtype": dtype,  # transformers v5 renamed torch_dtype -> dtype
        "low_cpu_mem_usage": True,
        "device_map": "auto" if device == "cuda" else {"": device},
    }
    model = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs)
    model.eval()
    true_id, false_id = answer_token_ids(tok)
    print(f"device={device}  ' True'->{true_id}  ' False'->{false_id}\n")

    for key in args.rules:
        rule = RULE_REGISTRY[key]
        fewshot = sample_balanced(rule, args.n_fewshot, seed=0)
        test = sample_balanced(rule, args.n_test, seed=1000)
        correct = 0
        for i, (x, y) in enumerate(test):
            dbg = args.debug and i < 3  # show diagnostics for the first few items
            if dbg:
                print(f"  [{key}] input={x!r}  true_label={y}")
            pred, _ = classify(model, tok, true_id, false_id, fewshot, x, debug=dbg)
            correct += pred == y
        print(f"{key:<28} {correct}/{len(test)} = {correct / len(test):.0%}")


if __name__ == "__main__":
    main()
