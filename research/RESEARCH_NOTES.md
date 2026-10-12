# Research grounding and falsifiable JEPA hypotheses

**Reviewed 12 October 2026.** Links below point to original papers, author repositories or the publishers. This note separates published work from our own unreviewed toy implementation.

## What is established

| Source | Relevant result or method | Consequence for this repository |
|---|---|---|
| Assran et al., *I-JEPA*, CVPR 2023 ([paper](https://openaccess.thecvf.com/content/CVPR2023/html/Assran_Self-Supervised_Learning_From_Images_With_a_Joint-Embedding_Predictive_Architecture_CVPR_2023_paper.html), [official code](https://github.com/facebookresearch/ijepa)) | Predict embeddings of masked image regions using context, an EMA target branch and a predictor; mask geometry matters. | Predict *representations* rather than raw observations. Our temporal toy does **not** reproduce I-JEPA's image masking or ViT. |
| Bardes, Ponce, LeCun, *VICReg*, 2021 ([paper](https://arxiv.org/abs/2105.04906)) | Variance and covariance terms explicitly discourage representation collapse and redundancy. | Add variance/covariance regularization and monitor actual embedding spread. Our objective is VICReg-inspired; it is not a VICReg implementation. |
| Sobal et al., *JEPAs Focus on Slow Features*, 2022 ([paper](https://arxiv.org/abs/2211.10831)) | In a moving-dot experiment, predictable fixed-background distractors harmed state-relevant representation learning. | Run separate fixed versus changing background conditions. This is a possible failure mode, not a universal theorem about every JEPA. |
| Assran et al., *V-JEPA 2*, 2025 ([paper](https://arxiv.org/abs/2506.09985), [official code](https://github.com/facebookresearch/vjepa2)) | Video representation pretraining, then action-conditioned model adaptation with robot trajectories, for planning. | Condition a small latent predictor on known synthetic actions, but do **not** claim robotic planning. |
| Balestriero and LeCun, *LeJEPA*, 2025 ([paper](https://arxiv.org/abs/2511.08544)) | Presents a teacher-free predictive objective with sketched isotropic Gaussian regularization (SIGReg) and theoretical claims under assumptions. | A distinct future comparator; cannot silently call our EMA+VICReg-like implementation LeJEPA. |
| Mur-Labadia et al., *V-JEPA 2.1*, 2026 ([paper](https://arxiv.org/abs/2603.14482), [official code](https://github.com/facebookresearch/vjepa2)) | Dense predictive loss, deep self-supervision and unified image/video training refine V-JEPA. | Potential future feature hierarchy, premature for this tiny CPU baseline. |

## Our narrower claim

The code is an **action-conditioned temporal JEPA-inspired baseline**: a student encoder encodes observation at time *t*; a predictor receives its latent embedding plus action *a_t*; an EMA target encoder embeds observation at *t+1*. The predictor minimizes squared distance to the stop-gradient target; student representations additionally receive variance and covariance penalties. The simulated hidden state **is not passed to training** and is used only for a frozen-encoder linear probe.

This is not evidence of general intelligence, recursively self-improving intelligence, world understanding, sample-efficient control or parity with published I-JEPA / V-JEPA benchmarks.

## Testable questions, before scaling

1. **Predictive usefulness:** within one trained run, is next-latent prediction MSE lower than the same run's untrained persistence comparator? Metric: `prediction_gain_vs_persistence`; positive is necessary, not sufficient.
2. **Representation relevance:** on disjoint held-out trajectories, does a frozen learned embedding linearly recover the hidden state better than a **random frozen encoder**? Compare `linear_probe_state_r2` with `random_encoder_state_r2`. Raw observation + linear probe is another comparator; a lower JEPA score is not a victory.
3. **Distractor robustness:** does held-out state-probe performance degrade when a nuisance background is held fixed within each trajectory? Compare dynamic/fixed conditions on at least three seeds, recognizing that the generated datasets have different observable distributions.
4. **Action use:** does excluding actions from the predictor hurt latent prediction or state retrieval? Differences may be small if dynamics are mostly predictable from current observations. This is a limited ablation, not a planning test.
5. **Anti-collapse:** track per-coordinate latent standard deviations and normalized prediction errors; low raw MSE is meaningless if the representation shrinks toward a constant. The simple variance term is a heuristic, not a collapse guarantee.

## Evaluation guardrails

- Split by **trajectory**, not adjacent frames; fixed observation-projection mapping and independent train/test random sequences.
- Linear probes train with simulator state *only after* self-supervised training. No test labels in fitting or hyperparameter search.
- Run multiple independent seeds, report mean and sample SD (the first pilot uses 7, 19, 43).
- Normalize predictive MSE by the target-embedding variance. Even normalized MSE alone is not task quality; the target distribution shifts across conditions.
- Include raw-observation and randomly initialized feature probes to detect an apparently impressive but actually trivial representation.
- Treat outcomes as **synthetic pilot observations**, not published benchmarks or model-to-model SOTA claims.
- Track configuration and hardware in JSON; if an experiment fails, preserve the failed result and explanation.

## Next scientific milestones

- **M1:** multi-seed synthetic baseline with state-probe and action/fixed-background ablations (this repository).
- **M2:** hidden-velocity or partially observed dynamical environments where a single frame cannot reveal the full predictive state; compare one-frame vs multiframe context.
- **M3:** systematic collapse and shortcut stress tests: fixed/background-changing, distribution shift, different noise levels and seed variance.
- **M4:** only after M1–M3 hold up, evaluate on a modest public observation stream and compare against direct next-observation prediction and LeJEPA-style alternatives.
