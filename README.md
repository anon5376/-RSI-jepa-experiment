# RSI JEPA Experiment

> **Public, reproducible research on latent predictive models.** Small experiments first; measured comparisons instead of AGI claims.

This repository explores **Joint-Embedding Predictive Architectures (JEPA)** and their limitations, beginning with a tiny CPU-first temporal benchmark. It has real executable code and tests; it is **not** a reproduction of Meta I-JEPA/V-JEPA and does not currently implement recursive self-improvement, a general-purpose conversational AI, or an autonomous training service.

**Status (12 October 2026):** A runnable synthetic action-conditioned temporal JEPA baseline, tests, research review, and bounded GitHub Actions jobs. See [`research/RESEARCH_NOTES.md`](research/RESEARCH_NOTES.md) for primary sources and what the baseline actually tests. Pilot metrics appear in `reports/` with provenance and limitations.

## Architecture

```text
Synthetic dynamical environment             Ground-truth state
  observation(t) + action(t)                   |   (probe only)
       |                                      v
  student encoder ──> z(t) ────> frozen linear probe
       |                  |
       |          + action(t)
       |                  v
       |             predictor
       |                  | z_pred(t+1)
       |                  v
       |             JEPA loss   <── stop-gradient z_teacher(t+1)
       |                                      ^
  EMA student weights ──> target encoder <── observation(t+1)

  + variance/covariance anti-collapse penalties on student embeddings
```

The **coding agent** (Claude, Codex, Devin CLI or Pi) writes and analyzes Python experiments; the **JEPA network** learns predictive embeddings. They are different systems. Connecting them via JSON experiment outputs does not turn JEPA into a text-generating assistant.

## Run it locally

Requires Python 3.10+ and PyTorch 2.x. No paid APIs, downloads of pretrained checkpoints, GPU, human data or login are necessary.

```bash
python -m venv .venv
source .venv/bin/activate
# Install PyTorch appropriate to your device: https://pytorch.org/get-started/locally/
python -m pip install -e '.[test]'
python -m pytest -q
python -m jepa_lab --config configs/smoke.json --output reports/smoke.json
python -m jepa_lab --config configs/baseline.json --output reports/baseline.json
```

Run a bounded 3-seed research matrix (nine runs, potentially several minutes on CPU):

```bash
python scripts/run_matrix.py --config configs/baseline.json --seeds 7 19 43 --output reports/pilot-matrix.json
```

Override individual conditions:

```bash
python -m jepa_lab --config configs/baseline.json --seed 19 --background fixed --output reports/fixed-19.json
python -m jepa_lab --config configs/baseline.json --seed 19 --no-actions --output reports/no-actions-19.json
```

Outputs are schema-versioned JSON with configuration, PyTorch version, CPU metadata, training losses, linear probes, latent prediction errors and embedding spread. Train/test trajectories are disjoint. Labels from the hidden simulated state are **not** used to train the JEPA encoder: only a downstream frozen-feature probe sees them.

## CI and reproducibility

- [CPU smoke workflow](.github/workflows/ci.yml): unit tests + a small reproducible experiment on push/PR.
- [Bounded experiment workflow](.github/workflows/experiment.yml): **manual trigger only**, six CPU jobs spanning three seeds and two nuisance regimes, saving metric artifacts. No unattended agent, no secret credentials and no training on each PR.
- [`EXPERIMENTS.md`](EXPERIMENTS.md) captures hypotheses, controls, measured outcomes and failures.
- [`AGENTS.md`](AGENTS.md) contains the agent execution policy. [`CLAUDE.md`](CLAUDE.md) points Claude at the same rules.

## Pilot interpretation

**Don't optimize for training loss alone.** A representation can predict a constant or persistent background with very low loss while losing information about the simulated underlying state. This project's key measurement is the *held-out* frozen-embedding state probe compared to random features and raw observations. The fixed-background condition tests shortcut learning. Differences among conditions require repeated seeds and defensible analysis.

The phrase **RSI** in the project title describes a research direction. No recursively self-improving AI is implemented or evidenced by these measurements.

## Optional coding-agent infrastructure

Devin CLI, Pi and Codespaces setup are documented separately in [`CLOUD_SETUP.md`](CLOUD_SETUP.md). Those integrations are **optional** and not required to run or reproduce these experiments. Do not put credentials in this public repository. No 24/7 hosted agent is currently running, and the CLI model promotion does not establish free indefinite cloud compute.

## Academic sources

Start with [I-JEPA (CVPR 2023)](https://openaccess.thecvf.com/content/CVPR2023/html/Assran_Self-Supervised_Learning_From_Images_With_a_Joint-Embedding_Predictive_Architecture_CVPR_2023_paper.html), [V-JEPA 2 (2025)](https://arxiv.org/abs/2506.09985), [V-JEPA 2.1 (2026)](https://arxiv.org/abs/2603.14482), [VICReg (2021)](https://arxiv.org/abs/2105.04906), [LeJEPA (2025)](https://arxiv.org/abs/2511.08544), and [JEPA slow-feature failures (2022)](https://arxiv.org/abs/2211.10831). The [research note](research/RESEARCH_NOTES.md) explains each connection and caveat.

**Licensing:** Public visibility is not an open-source license. No reuse license has been selected by the repository owner.
