"""astra-articulation: can an LLM articulate the rules it learns in-context?

Layout:
  rules/      one (input_space, labeler) pair per classification rule + registry
  inputs.py   input-space generators + balanced sampler
  model.py    Claude API primitives: classify_one(), articulate_one()
  cache.py    on-disk response cache (keyed by model + prompt) so reruns are free
  controls.py confound controls (shuffled labels, etc.)
"""
