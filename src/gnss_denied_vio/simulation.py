"""End-to-end GNSS-denied visual-inertial localization simulation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np

from .config import SimulationConfig
from .ekf import BAX, BAY, BGZ, VX, VY, X, Y, YAW, ProcessNoise, VisualInertialEKF, wrap_angle
from .plots import render_all_plots
from .sensors import Measurements, GroundTruth, generate_ground_truth, generate_measurements


@dataclass(frozen=True)
class SimulationResult:
    config: SimulationConfig
    truth: GroundTruth
    measurements: Measurements
    fused_state: np.ndarray
    fused_covariance: np.ndarray
    inertial_odom_state: np.ndarray
    metrics: dict[str, float | int]


def run_simulation(config: SimulationConfig) -> SimulationResult:
    truth = generate_ground_truth(config)
    measurements = generate_measurements(config, truth)

    initial_state = _initial_state(truth, measurements)
    initial_covariance = np.diag([2.0, 2.0, 0.8, 0.8, 0.2, 0.08, 0.08, 0.02]) ** 2
    noise = ProcessNoise(
        accel_noise_std=config.imu_accel_noise_std,
        gyro_noise_std=config.imu_gyro_noise_std,
        accel_bias_walk_std=config.accel_bias_walk_std,
        gyro_bias_walk_std=config.gyro_bias_walk_std,
    )

    fused = VisualInertialEKF(initial_state, initial_covariance, noise)
    inertial_odom = VisualInertialEKF(initial_state, initial_covariance, noise)

    steps = config.steps()
    fused_state = np.zeros((steps, 8))
    inertial_odom_state = np.zeros((steps, 8))
    fused_covariance = np.zeros((steps, 8, 8))

    for k in range(steps):
        if k > 0:
            accel_x, accel_y = measurements.imu_accel_body[k - 1]
            gyro_z = measurements.imu_gyro_z[k - 1]
            fused.predict(accel_x, accel_y, gyro_z, config.dt_s)
            inertial_odom.predict(accel_x, accel_y, gyro_z, config.dt_s)

        if measurements.odom_available[k]:
            fused.update_forward_speed(measurements.odom_speed[k], config.odom_speed_std)
            inertial_odom.update_forward_speed(measurements.odom_speed[k], config.odom_speed_std)

        if measurements.visual_available[k]:
            fused.update_visual_pose(measurements.visual_pose[k], config.visual_pos_std, config.visual_yaw_std)

        if measurements.gnss_available[k]:
            fused.update_gnss_position(measurements.gnss_position[k], config.gnss_pos_std)

        fused_state[k] = fused.snapshot()
        inertial_odom_state[k] = inertial_odom.snapshot()
        fused_covariance[k] = fused.covariance_snapshot()

    metrics = _compute_metrics(config, truth, measurements, fused_state, inertial_odom_state)
    return SimulationResult(
        config=config,
        truth=truth,
        measurements=measurements,
        fused_state=fused_state,
        fused_covariance=fused_covariance,
        inertial_odom_state=inertial_odom_state,
        metrics=metrics,
    )


def save_result(result: SimulationResult, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "run.npz",
        t=result.truth.t,
        truth_pose=result.truth.pose,
        fused_state=result.fused_state,
        inertial_odom_state=result.inertial_odom_state,
        gnss_position=result.measurements.gnss_position,
        visual_pose=result.measurements.visual_pose,
        dropout_mask=result.measurements.dropout_mask,
    )
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(result.metrics, handle, indent=2, sort_keys=True)
        handle.write("\n")
    render_all_plots(result, output_dir)


def _initial_state(truth: GroundTruth, measurements: Measurements) -> np.ndarray:
    state = np.zeros(8)
    if measurements.gnss_available[0]:
        state[[X, Y]] = measurements.gnss_position[0]
    else:
        state[[X, Y]] = truth.pose[0, :2]
    state[[VX, VY]] = truth.velocity[0]
    state[YAW] = truth.pose[0, 2]
    state[[BAX, BAY, BGZ]] = 0.0
    return state


def _compute_metrics(
    config: SimulationConfig,
    truth: GroundTruth,
    measurements: Measurements,
    fused_state: np.ndarray,
    inertial_odom_state: np.ndarray,
) -> dict[str, float | int]:
    fused_error = _position_error(fused_state[:, :2], truth.pose[:, :2])
    inertial_error = _position_error(inertial_odom_state[:, :2], truth.pose[:, :2])
    visual_error = _position_error(measurements.visual_pose[:, :2], truth.pose[:, :2])
    visual_error = visual_error[np.isfinite(visual_error)]
    dropout = measurements.dropout_mask

    fused_yaw_error = np.abs(wrap_angle(fused_state[:, YAW] - truth.pose[:, 2]))
    inertial_yaw_error = np.abs(wrap_angle(inertial_odom_state[:, YAW] - truth.pose[:, 2]))

    dropout_fused_rmse = _rmse(fused_error[dropout])
    dropout_inertial_rmse = _rmse(inertial_error[dropout])
    improvement = 100.0 * (1.0 - dropout_fused_rmse / max(dropout_inertial_rmse, 1e-9))

    return {
        "seed": config.seed,
        "duration_s": config.duration_s,
        "dropout_start_s": config.dropout_start_s,
        "dropout_end_s": config.dropout_end_s(),
        "dropout_duration_s": config.dropout_duration_s,
        "gnss_fix_count": int(measurements.gnss_available.sum()),
        "visual_update_count": int(measurements.visual_available.sum()),
        "odom_update_count": int(measurements.odom_available.sum()),
        "fused_position_rmse_m": _rmse(fused_error),
        "inertial_odom_position_rmse_m": _rmse(inertial_error),
        "visual_position_rmse_m": _rmse(visual_error),
        "dropout_fused_position_rmse_m": dropout_fused_rmse,
        "dropout_inertial_odom_position_rmse_m": dropout_inertial_rmse,
        "dropout_improvement_percent": improvement,
        "fused_max_error_m": float(np.max(fused_error)),
        "inertial_odom_max_error_m": float(np.max(inertial_error)),
        "fused_final_error_m": float(fused_error[-1]),
        "inertial_odom_final_error_m": float(inertial_error[-1]),
        "fused_yaw_rmse_rad": _rmse(fused_yaw_error),
        "inertial_odom_yaw_rmse_rad": _rmse(inertial_yaw_error),
    }


def _position_error(estimate_xy: np.ndarray, truth_xy: np.ndarray) -> np.ndarray:
    return np.linalg.norm(estimate_xy - truth_xy, axis=1)


def _rmse(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(values**2)))

