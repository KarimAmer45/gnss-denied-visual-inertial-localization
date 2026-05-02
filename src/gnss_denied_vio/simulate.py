"""Command line entry point for the GNSS-denied localization demo."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import SimulationConfig
from .simulation import run_simulation, save_result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a reproducible GNSS-denied visual-inertial EKF simulation.")
    parser.add_argument("--seed", type=int, default=SimulationConfig.seed, help="Random seed for repeatable sensor noise.")
    parser.add_argument("--duration", type=float, default=SimulationConfig.duration_s, help="Scenario duration in seconds.")
    parser.add_argument("--dt", type=float, default=SimulationConfig.dt_s, help="Simulation timestep in seconds.")
    parser.add_argument("--dropout-start", type=float, default=SimulationConfig.dropout_start_s, help="GNSS outage start time.")
    parser.add_argument("--dropout-duration", type=float, default=SimulationConfig.dropout_duration_s, help="GNSS outage duration.")
    parser.add_argument("--output", type=Path, default=Path("results/example"), help="Directory for plots and metrics.")
    parser.add_argument("--no-plots", action="store_true", help="Skip writing plots and save only metrics.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = SimulationConfig(
        seed=args.seed,
        duration_s=args.duration,
        dt_s=args.dt,
        dropout_start_s=args.dropout_start,
        dropout_duration_s=args.dropout_duration,
    )
    result = run_simulation(config)
    if args.no_plots:
        args.output.mkdir(parents=True, exist_ok=True)
        with (args.output / "metrics.json").open("w", encoding="utf-8") as handle:
            json.dump(result.metrics, handle, indent=2, sort_keys=True)
            handle.write("\n")
    else:
        save_result(result, args.output)

    print("GNSS-denied visual-inertial localization run complete")
    print(f"  output: {args.output}")
    print(f"  GNSS outage: {config.dropout_start_s:.1f}s to {config.dropout_end_s():.1f}s")
    print(f"  fused dropout RMSE: {result.metrics['dropout_fused_position_rmse_m']:.2f} m")
    print(f"  inertial/odom dropout RMSE: {result.metrics['dropout_inertial_odom_position_rmse_m']:.2f} m")
    print(f"  improvement: {result.metrics['dropout_improvement_percent']:.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
