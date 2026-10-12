# JEPA experiment log

## 2026-10-12 — First synthetic temporal JEPA pilot

**Question:** Can a small latent-prediction model learn a useful hidden-state representation, and does a predictable fixed distractor degrade it?

**Method:** Train a student MLP encoder to predict next-observation embeddings from the current observation and synthetic 2D action. The teacher is an exponential-moving-average copy with stopped gradients. Apply variance/covariance penalties to the student. The hidden state is withheld from self-supervised pretraining and used only to fit an **after-training frozen-feature ridge probe** on separate trajectories.

- **Training:** 192 independent trajectories × 16 transitions; test 64 independent trajectories × 16 transitions. Fixed observation mapping shared across train/test. Train and test trajectories do not overlap.
- **Controls:** random untrained encoder, direct raw-observation probe, next-embedding persistence, no-action predictor and fixed-vs-changing background.
- **Config:** `configs/baseline.json`; train each condition for 30 epochs at seeds 7, 19 and 43.
- **Machine:** Python 3.13.5, PyTorch 2.10.0+cpu, Linux 6.18.44-x86_64, CPU.
- **Command:** `PYTHONPATH=src python scripts/run_matrix.py --config configs/baseline.json --seeds 7 19 43 --output reports/pilot-matrix-20261012.json`.
- **Machine-readable published summary:** [`reports/pilot-summary-20261012.json`](reports/pilot-summary-20261012.json). The Python command generates a more detailed complete JSON matrix locally; the tracked summary records rounded measured values.
- **Unit tests:** 6 passed on the local execution environment. CI runs the tests and smoke experiment independently.

### Measured outcomes (3 seeds; mean ± sample SD for state R²)

| Condition | Learned frozen-state probe R² | Random encoder R² (mean) | Raw-observation R² (mean) | Predictor gain vs persistence |
|---|---:|---:|---:|---:|
| Dynamic background, actions | 0.925 ± 0.029 | 0.680 | 0.955 | 0.530 |
| Dynamic background, **without** actions | 0.916 ± 0.037 | 0.680 | 0.955 | 0.490 |
| Fixed per-trajectory background, actions | 0.846 ± 0.045 | 0.668 | 0.954 | 0.816 |

**Interpretation:**
- Learned features decode state better than randomly initialized features on this synthetic dataset, but **do not beat direct raw-observation probing**.
- When background persists throughout a trajectory, state-probe performance falls despite the high within-run predictive gain: an example of a **predictable shortcut**. The finding is compatible with the phenomenon examined in [Sobal et al. 2022](https://arxiv.org/abs/2211.10831).
- Providing actions modestly improves state R² on the dynamic setting (+0.009 on the three-seed mean) and improves predictor gain. This small pilot does not establish meaningful control or planning.
- Prediction gain **must not be compared as a general success metric across independently trained representations**, especially fixed and dynamic nuisance conditions; target feature spaces and difficulty change.
- A higher variance-normalized predictive accuracy alone can still hide low-quality state representations.

**Limitations:** Only three independent seeds and a synthetic task with known dynamics. No out-of-distribution videos, no real-world agent actions, no learned planner, no evidence of recursively self-improving intelligence. CPU PyTorch/backend differences can affect exact values.

**Next test:** build an environment requiring multiple past frames to infer hidden velocity. Compare single-frame vs temporal context under the same data and compute budget, then repeat at 5+ seeds. This creates a harder benchmark than the present nearly observable 2D system.

---

## Template for subsequent verified experiments

- **Date / hypothesis:**
- **Exact code commit, config, seed and data split:**
- **Independent train/test controls and baselines:**
- **Command, hardware, elapsed time, memory:**
- **Machine-readable artifact and metrics:**
- **Negative results / failures and interpretation:**
- **Next falsifiable experiment:**

Do not write a result until it has been executed. Preserve negative outcomes and failed tests.
