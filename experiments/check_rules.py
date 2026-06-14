"""Local sanity check — NO API calls, NO key needed.

Verifies every registered rule can produce a balanced sample and prints a couple
of labeled examples so you can eyeball the input space. Run before spending any
tokens:  uv run python experiments/check_rules.py
"""

from articulation.inputs import sample_balanced
from articulation.rules import RULE_REGISTRY


def main() -> None:
    print(f"{len(RULE_REGISTRY)} rules registered.\n")
    for key, rule in RULE_REGISTRY.items():
        items = sample_balanced(rule, n=60, seed=0)
        n_pos = sum(1 for _, y in items if y)
        ex_pos = next(x for x, y in items if y)
        ex_neg = next(x for x, y in items if not y)
        print(f"[{rule.category:>10}] {key:<16} balance {n_pos}/{len(items)} True")
        print(f"             True : {ex_pos!r}")
        print(f"             False: {ex_neg!r}")
    print("\nAll rules balanced OK.")


if __name__ == "__main__":
    main()
