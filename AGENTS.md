# Agent instructions: RSI JEPA Experiment

Always read `README.md` and `research/RESEARCH_NOTES.md` before changing code. This is a **public research repo** with a runnable temporal JEPA-inspired baseline. Do not mistake it for a Meta model reproduction, an AGI system or a deployed 24/7 agent.

## Priorities

1. **Scientific correctness over more code:** run unit tests, reproduce pilot results and report negative outcomes. Prefer controlled, falsifiable experiments over building agent infrastructure.
2. **Preserve reproducibility:** fixed seeds, train/test trajectories disjoint, explicit configuration, schema-versioned metrics, recording environment, real links to primary sources. Treat hidden simulator states as *evaluation labels only*; don't leak them into JEPA training.
3. **Use honest comparators:** frozen random encoder, raw observation probe, persistence predictor, no-action ablation and fixed/dynamic nuisance. Never compare unnormalized embedding losses from different target encoders as if they share a scale.
4. **Stay small:** CPU-first, no unnecessary downloads, no silent paid provider calls, no training on private data. Bounded runs and killable processes; no indefinite autonomous loops.
5. **Make only verified claims:** measured run outcome, exact test commands, files changed and unsolved failures. No SOTA/AGI/recursive self-improvement claims without evidence.

## Working commands

```bash
python -m pip install -e '.[test]'
python -m pytest -q
python -m jepa_lab --config configs/smoke.json --output reports/smoke.json
python scripts/run_matrix.py --config configs/baseline.json --seeds 7 19 43 --output reports/pilot-matrix.json
```

CI runs smoke only; the multi-seed GitHub Actions workflow is manually dispatched and bounded. Do not alter it to an always-on loop without explicit owner permission. If adding configuration or metrics, update both tests and README.

## Agent ownership and authentication

Never solicit passwords, 2FA codes, session tokens or privileged OAuth links in chat, logs or this repo. Devin CLI / Pi setup is an **optional** owner-controlled workflow in `CLOUD_SETUP.md`. It is not a prerequisite for experiments. The human owner completes authentication in their own terminal if they elect to use these tools. Never bypass provider quotas or publish their authentication materials.

For every contribution, state (a) hypothesis, (b) exact change and (c) measured before/after and tests; retain failed experiments. GitHub is public; don't commit `.env`, private code, weights/datasets or local auth state.
