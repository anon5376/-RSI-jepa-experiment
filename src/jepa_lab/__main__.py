"""CPU-first temporal JEPA sandbox: observations -> teacher embeddings, not pixels.

This is a teaching-scale action-conditioned JEPA variant, not Meta I-JEPA or
V-JEPA reproduction. States exist only in the synthetic simulator and are
excluded from self-supervised pretraining; they are used by the linear probe.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import platform
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class Config:
    seed: int = 7
    train_trajectories: int = 64
    test_trajectories: int = 24
    steps: int = 12
    signal_dims: int = 12
    distractor_dims: int = 8
    hidden_dim: int = 48
    embedding_dim: int = 12
    epochs: int = 12
    batch_size: int = 96
    lr: float = 0.001
    ema: float = 0.99
    variance_weight: float = 1.0
    covariance_weight: float = 0.02
    observation_noise: float = 0.04
    background: str = "dynamic"  # "dynamic" or "fixed" per trajectory
    use_actions: bool = True

    def validate(self) -> None:
        assert self.train_trajectories >= 2 and self.test_trajectories >= 2
        assert self.steps >= 2
        assert self.signal_dims >= 2 and self.distractor_dims >= 0
        assert self.hidden_dim > 0 and self.embedding_dim > 0
        assert self.epochs > 0 and self.batch_size >= 2
        assert self.lr > 0 and 0 <= self.ema < 1
        assert self.variance_weight >= 0 and self.covariance_weight >= 0
        assert self.observation_noise >= 0
        assert self.background in {"dynamic", "fixed"}


@dataclass
class Split:
    obs: Tensor
    actions: Tensor
    next_obs: Tensor
    states: Tensor
    next_states: Tensor
    trajectory_ids: Tensor


def make_splits(cfg: Config) -> tuple[Split, Split]:
    """Disjoint trajectories, shared frozen observation mapping, no labels in SSL.

    Random generators for projection, train and test are isolated so adding
    training trajectories never changes the held-out split.
    """
    cfg.validate()
    pg = torch.Generator().manual_seed(cfg.seed + 11)
    projection = torch.randn(9, cfg.signal_dims, generator=pg) / math.sqrt(9)
    angle = 0.28
    rot = torch.tensor([[math.cos(angle), -math.sin(angle)],
                        [math.sin(angle), math.cos(angle)]], dtype=torch.float32)

    def build(count: int, random_seed: int, id_offset: int) -> Split:
        g = torch.Generator().manual_seed(random_seed)
        rows: list[tuple[Tensor, Tensor, Tensor, Tensor, Tensor, int]] = []

        def observe(s: Tensor, fixed_background: Tensor) -> Tensor:
            feats = torch.stack((s[0], s[1], s[0] ** 2, s[1] ** 2,
                                 s[0] * s[1], torch.sin(s[0]),
                                 torch.sin(s[1]), torch.cos(s[0]),
                                 torch.cos(s[1])))
            signal = 1.5 * torch.tanh(feats @ projection)
            signal = signal + cfg.observation_noise * torch.randn(cfg.signal_dims, generator=g)
            if cfg.distractor_dims:
                background = (fixed_background if cfg.background == "fixed"
                              else torch.randn(cfg.distractor_dims, generator=g))
                return torch.cat((signal, 1.0 * background))
            return signal

        for trajectory in range(count):
            s = torch.randn(2, generator=g)
            fixed_background = torch.randn(cfg.distractor_dims, generator=g)
            for _ in range(cfg.steps):
                action = 0.7 * torch.randn(2, generator=g)
                next_s = 0.96 * (rot @ s) + 0.28 * action
                rows.append((observe(s, fixed_background), action,
                             observe(next_s, fixed_background), s.clone(),
                             next_s.clone(), id_offset + trajectory))
                s = next_s

        return Split(
            obs=torch.stack([r[0] for r in rows]),
            actions=torch.stack([r[1] for r in rows]),
            next_obs=torch.stack([r[2] for r in rows]),
            states=torch.stack([r[3] for r in rows]),
            next_states=torch.stack([r[4] for r in rows]),
            trajectory_ids=torch.tensor([r[5] for r in rows], dtype=torch.long),
        )

    train = build(cfg.train_trajectories, cfg.seed + 101, 0)
    test = build(cfg.test_trajectories, cfg.seed + 202, cfg.train_trajectories)
    return train, test


class Encoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, embedding_dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.GELU(),
                                 nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
                                 nn.Linear(hidden_dim, embedding_dim))

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class TemporalJEPA(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.student = Encoder(cfg.signal_dims + cfg.distractor_dims,
                               cfg.hidden_dim, cfg.embedding_dim)
        self.teacher = copy.deepcopy(self.student)
        self.teacher.requires_grad_(False)
        self.predictor = nn.Sequential(
            nn.Linear(cfg.embedding_dim + 2, cfg.hidden_dim), nn.GELU(),
            nn.Linear(cfg.hidden_dim, cfg.embedding_dim),
        )

    def forward(self, obs: Tensor, actions: Tensor) -> Tensor:
        if not self.cfg.use_actions:
            actions = torch.zeros_like(actions)
        return self.predictor(torch.cat((self.student(obs), actions), dim=-1))

    @torch.no_grad()
    def update_teacher(self) -> None:
        for target, online in zip(self.teacher.parameters(), self.student.parameters()):
            target.lerp_(online, 1.0 - self.cfg.ema)


def variance_penalty(embeddings: Tensor) -> Tensor:
    """VICReg-inspired anti-collapse term; batch must have >= 2 elements."""
    std = torch.sqrt(embeddings.var(dim=0, unbiased=False) + 1e-4)
    return F.relu(1.0 - std).mean()


def covariance_penalty(embeddings: Tensor) -> Tensor:
    x = embeddings - embeddings.mean(dim=0, keepdim=True)
    cov = (x.T @ x) / max(1, x.shape[0] - 1)
    off_diag = cov - torch.diag(torch.diag(cov))
    return off_diag.square().sum() / embeddings.shape[1]


def linear_probe_r2(train_x: Tensor, train_y: Tensor,
                    test_x: Tensor, test_y: Tensor, ridge: float = 0.01) -> float:
    """Frozen features, fitted on train only; reports aggregate held-out R²."""
    train_design = torch.cat((train_x, torch.ones(len(train_x), 1)), dim=1).double()
    test_design = torch.cat((test_x, torch.ones(len(test_x), 1)), dim=1).double()
    y = train_y.double()
    reg = ridge * torch.eye(train_design.shape[1], dtype=torch.float64)
    reg[-1, -1] = 0.0  # unpenalized intercept
    coefficients = torch.linalg.solve(train_design.T @ train_design + reg,
                                      train_design.T @ y)
    prediction = test_design @ coefficients
    errors = (test_y.double() - prediction).square().sum()
    total = (test_y.double() - test_y.double().mean(dim=0)).square().sum()
    return float((1.0 - errors / total).item())


def evaluate(model: TemporalJEPA, random_encoder: Encoder,
             train: Split, test: Split) -> dict[str, float | int]:
    model.eval()
    with torch.no_grad():
        encoded_train = model.student(train.obs)
        encoded_test = model.student(test.obs)
        random_train = random_encoder(train.obs)
        random_test = random_encoder(test.obs)
        pred = model(test.obs, test.actions)
        target = model.teacher(test.next_obs)
        emb_std = encoded_test.std(dim=0, unbiased=False)
        teacher_var = target.var(dim=0, unbiased=False).mean()
        prediction_mse = F.mse_loss(pred, target)
        persistence_mse = F.mse_loss(encoded_test, target)
    return {
        "linear_probe_state_r2": linear_probe_r2(encoded_train, train.states,
                                                 encoded_test, test.states),
        "random_encoder_state_r2": linear_probe_r2(random_train, train.states,
                                                   random_test, test.states),
        "raw_observation_state_r2": linear_probe_r2(train.obs, train.states,
                                                   test.obs, test.states),
        "latent_prediction_mse": float(prediction_mse.item()),
        "latent_persistence_mse": float(persistence_mse.item()),
        "teacher_mean_variance": float(teacher_var.item()),
        "prediction_mse_over_teacher_variance": float((prediction_mse / (teacher_var + 1e-8)).item()),
        "prediction_gain_vs_persistence": float((1 - prediction_mse / (persistence_mse + 1e-8)).item()),
        "embedding_mean_std": float(emb_std.mean().item()),
        "collapsed_dims_std_lt_0_01": int((emb_std < 0.01).sum().item()),
        "embedding_dims": int(emb_std.numel()),
    }


def run(cfg: Config) -> dict:
    cfg.validate()
    torch.manual_seed(cfg.seed)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    train, test = make_splits(cfg)
    model = TemporalJEPA(cfg)
    random_encoder = copy.deepcopy(model.student).eval()
    opt = torch.optim.AdamW([*model.student.parameters(),
                             *model.predictor.parameters()], lr=cfg.lr)
    n = len(train.obs)
    losses: list[float] = []
    shuffle_generator = torch.Generator().manual_seed(cfg.seed + 303)

    for _ in range(cfg.epochs):
        model.train()
        order = torch.randperm(n, generator=shuffle_generator)
        batch_losses: list[float] = []
        for ids in order.split(cfg.batch_size):
            if len(ids) < 2:  # variance needs at least two observations
                continue
            obs, actions, next_obs = train.obs[ids], train.actions[ids], train.next_obs[ids]
            pred = model(obs, actions)
            with torch.no_grad():
                target = model.teacher(next_obs)
            embeddings = model.student(obs)
            loss = (F.mse_loss(pred, target)
                    + cfg.variance_weight * variance_penalty(embeddings)
                    + cfg.covariance_weight * covariance_penalty(embeddings))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            model.update_teacher()
            batch_losses.append(float(loss.item()))
        losses.append(sum(batch_losses) / len(batch_losses))

    metrics = evaluate(model, random_encoder, train, test)
    return {
        "schema_version": 1,
        "experiment": "synthetic_action_conditioned_temporal_jepa",
        "method": "ema_target_student_predictor_with_vicreg_inspired_regularization",
        "status": "measured",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": asdict(cfg),
        "environment": {"python": platform.python_version(),
                        "pytorch": torch.__version__, "device": "cpu",
                        "platform": platform.platform()},
        "sample_counts": {"train": len(train.obs), "test": len(test.obs)},
        "metrics": metrics,
        "training_loss_first_epoch": losses[0],
        "training_loss_last_epoch": losses[-1],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Small reproducible temporal JEPA benchmark")
    parser.add_argument("--config", type=Path, default=Path("configs/smoke.json"))
    parser.add_argument("--output", type=Path, default=Path("reports/smoke.json"))
    parser.add_argument("--seed", type=int, help="Override config random seed")
    parser.add_argument("--background", choices=["dynamic", "fixed"],
                        help="Distractor changes each observation or stays fixed per trajectory")
    parser.add_argument("--no-actions", action="store_true",
                        help="Zero out actions fed to predictor (controlled ablation)")
    args = parser.parse_args()
    cfg = Config(**json.loads(args.config.read_text(encoding="utf-8")))
    if args.seed is not None:
        cfg = replace(cfg, seed=args.seed)
    if args.background is not None:
        cfg = replace(cfg, background=args.background)
    if args.no_actions:
        cfg = replace(cfg, use_actions=False)
    result = run(cfg)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"saved": str(args.output), **result["metrics"]}, indent=2))


if __name__ == "__main__":
    main()
