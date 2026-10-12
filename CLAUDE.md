# Claude / Codex startup: RSI JEPA Experiment

Read `AGENTS.md`, `README.md` and `research/RESEARCH_NOTES.md` first. These specify the current code, scientific controls, trusted sources and research boundaries.

**Default assignment:** run the CPU unit tests, inspect the synthetic temporal JEPA implementation, and execute a bounded falsifiable ablation. Commit code and reports with real results. Avoid generic agent harness engineering until the experiments provide meaningful evidence.

Commands:

```bash
python -m pip install -e '.[test]'
python -m pytest -q
python -m jepa_lab --config configs/smoke.json --output reports/smoke.json
```

This repository is public. Do not publish secrets, model-service tokens or privileged login links. If the owner **explicitly** asks to configure Devin CLI, follow `CLOUD_SETUP.md`, let the owner authenticate personally in a terminal they control, and verify availability/pricing before running it. Devin CLI is optional, and model promotions do not mean unlimited cloud compute.

**Do not claim this repo contains a trained Meta JEPA, an autonomous agent or recursively self-improving intelligence.** It is a reproducible synthetic research sandbox.
