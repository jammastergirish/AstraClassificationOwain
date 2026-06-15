"""Print example inputs (exactly what the model is shown) for every rule.
Local only — no API. Deterministic (seed=0).

  uv run python experiments/dump_examples.py
"""

from articulation.inputs import sample_balanced
from articulation.rules import RULE_REGISTRY

N = 80
for key, rule in RULE_REGISTRY.items():
    items = sample_balanced(rule, N, seed=0)
    pos = [x for x, y in items if y][:3]
    neg = [x for x, y in items if not y][:3]
    print(f"### {key}  —  {rule.human_articulation}")
    for x in pos:
        print(f"    True : {x!r}")
    for x in neg:
        print(f"    False: {x!r}")
    print()
