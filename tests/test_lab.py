import math
from dataclasses import replace

import pytest
import torch

from jepa_lab.__main__ import (Config, TemporalJEPA, covariance_penalty,
                               linear_probe_r2, make_splits, run, variance_penalty)


def tiny_cfg() -> Config:
    return Config(train_trajectories=5, test_trajectories=3, steps=3,
                  epochs=1, batch_size=8, hidden_dim=16, embedding_dim=4)


def test_trajectory_split_is_disjoint_and_deterministic():
    cfg = tiny_cfg()
    train, test = make_splits(cfg)
    assert not set(train.trajectory_ids.tolist()) & set(test.trajectory_ids.tolist())
    train_again, test_again = make_splits(cfg)
    assert torch.equal(train.obs, train_again.obs)
    assert torch.equal(test.obs, test_again.obs)
    assert train.obs.shape == (cfg.train_trajectories * cfg.steps, cfg.signal_dims + cfg.distractor_dims)


def test_teacher_is_frozen_and_ema_moves_toward_student():
    model = TemporalJEPA(tiny_cfg())
    assert not any(p.requires_grad for p in model.teacher.parameters())
    before = next(model.teacher.parameters()).detach().clone()
    with torch.no_grad():
        next(model.student.parameters()).add_(1.0)
    model.update_teacher()
    after = next(model.teacher.parameters()).detach()
    assert torch.allclose(after - before, torch.full_like(after, 0.01), atol=1e-6)


def test_action_conditioning_can_be_abated():
    cfg = tiny_cfg()
    model = TemporalJEPA(cfg)
    obs = torch.randn(3, cfg.signal_dims + cfg.distractor_dims)
    zeros, ones = torch.zeros(3, 2), torch.ones(3, 2)
    assert not torch.allclose(model(obs, zeros), model(obs, ones))
    ablation = TemporalJEPA(replace(cfg, use_actions=False))
    assert torch.allclose(ablation(obs, zeros), ablation(obs, ones))


def test_regularizers_and_probe():
    constant = torch.zeros(12, 4)
    varied = torch.randn(12, 4)
    assert variance_penalty(constant) > variance_penalty(varied)
    assert covariance_penalty(constant).item() == 0
    states = torch.randn(20, 2)
    r2 = linear_probe_r2(states[:15], states[:15], states[15:], states[15:])
    assert r2 > 0.999


def test_run_has_finite_metrics_and_valid_sample_counts():
    cfg = tiny_cfg()
    result = run(cfg)
    assert result["sample_counts"] == {"train": 15, "test": 9}
    assert result["schema_version"] == 1
    assert all(math.isfinite(v) for v in result["metrics"].values())


def test_invalid_configuration_is_rejected():
    with pytest.raises(AssertionError):
        make_splits(replace(tiny_cfg(), background="unknown"))
