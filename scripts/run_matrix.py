"""Explicit bounded research matrix; no agent loop, no downloads, no paid APIs."""

from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from jepa_lab.__main__ import Config, run


CONDITIONS = {
    "dynamic_actions": {"background": "dynamic", "use_actions": True},
    "dynamic_no_actions": {"background": "dynamic", "use_actions": False},
    "fixed_actions": {"background": "fixed", "use_actions": True},
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/baseline.json"))
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 19, 43])
    parser.add_argument("--output", type=Path, default=Path("reports/pilot-matrix.json"))
    args = parser.parse_args()
    if not (1 <= len(args.seeds) <= 8):
        parser.error("1 to 8 seeds allowed in a bounded pilot")
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("seeds must be distinct")
    base = Config(**json.loads(args.config.read_text(encoding="utf-8")))
    runs = []
    for condition, changes in CONDITIONS.items():
        for seed in args.seeds:
            result = run(replace(base, seed=seed, **changes))
            runs.append({"condition": condition, "seed": seed, "metrics": result["metrics"],
                         "training_loss_last_epoch": result["training_loss_last_epoch"]})
            print(condition, seed, result["metrics"]["linear_probe_state_r2"], flush=True)
    summary = {}
    fields = ("linear_probe_state_r2", "random_encoder_state_r2",
              "raw_observation_state_r2", "prediction_gain_vs_persistence",
              "prediction_mse_over_teacher_variance", "embedding_mean_std")
    for condition in CONDITIONS:
        group = [run["metrics"] for run in runs if run["condition"] == condition]
        summary[condition] = {field: {"mean": statistics.mean(row[field] for row in group),
                                       "std": statistics.stdev(row[field] for row in group)
                                       if len(group) > 1 else None}
                              for field in fields}
    report = {"schema_version": 1, "status": "measured",
              "generated_at_utc": datetime.now(timezone.utc).isoformat(),
              "base_config": str(args.config), "seeds": args.seeds,
              "conditions": CONDITIONS, "runs": runs, "summary": summary,
              "notes": ["Toy synthetic trajectories only, no real data.",
                        "No single-run performance claim should be generalized.",
                        "Latent MSE across separately trained embeddings is not directly comparable."]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("Saved", args.output)


if __name__ == "__main__":
    main()
