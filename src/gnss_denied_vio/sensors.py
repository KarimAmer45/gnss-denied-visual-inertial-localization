"""Synthetic trajectory and sensor generation for the demo scenario."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import SimulationConfig
from .ekf import wrap_angle


@dataclass(frozen=True)
class GroundTruth:
    t: np.ndarray
    pose: np.ndarray
    velocity: np.ndarray
    accel_body: np.ndarray
    gyro_z: np.ndarray
    speed: np.ndarray


@dataclass(frozen=True)
class Measurements:
    imu_accel_body: np.ndarray
    imu_gyro_z: np.ndarray
    odom_speed: np.ndarray
    visual_pose: np.ndarray
    gnss_position: np.ndarray
    gnss_available: np.ndarray
    visual_available: np.ndarray
    odom_available: np.ndarray
    dropout_mask: np.ndarray


def generate_ground_truth(config: SimulationConfig) -> GroundTruth:
    steps = config.steps()
    t = np.linspace(0.0, config.duration_s, steps)
    dt = config.dt_s

    speed = 3.6 + 0.7 * np.sin(0.08 * t) + 0.25 * np.cos(0.17 * t)
    yaw_rate = 0.055 * np.sin(0.11 * t) + 0.025 * np.cos(0.035 * t)

    yaw = np.zeros(steps)
    yaw[0] = 0.35
    for k in range(1, steps):
        yaw[k] = wrap_angle(yaw[k - 1] + yaw_rate[k - 1] * dt)

    velocity = np.column_stack((speed * np.cos(yaw), speed * np.sin(yaw)))
    pose = np.zeros((steps, 3))
    pose[:, 2] = yaw
    for k in range(1, steps):
        pose[k, 0] = pose[k - 1, 0] + velocity[k - 1, 0] * dt
        pose[k, 1] = pose[k - 1, 1] + velocity[k - 1, 1] * dt

    accel_world = np.column_stack(
        (
            np.gradient(velocity[:, 0], dt, edge_order=2),
            np.gradient(velocity[:, 1], dt, edge_order=2),
        )
    )
    accel_body = np.empty_like(accel_world)
    accel_body[:, 0] = np.cos(yaw) * accel_world[:, 0] + np.sin(yaw) * accel_world[:, 1]
    accel_body[:, 1] = -np.sin(yaw) * accel_world[:, 0] + np.cos(yaw) * accel_world[:, 1]
    gyro_z = np.gradient(np.unwrap(yaw), dt, edge_order=2)
    return GroundTruth(t=t, pose=pose, velocity=velocity, accel_body=accel_body, gyro_z=gyro_z, speed=speed)


def generate_measurements(config: SimulationConfig, truth: GroundTruth) -> Measurements:
    rng = np.random.default_rng(config.seed)
    steps = config.steps()
    dropout_mask = (truth.t >= config.dropout_start_s) & (truth.t <= config.dropout_end_s())

    accel_bias = np.zeros((steps, 2))
    gyro_bias = np.zeros(steps)
    accel_bias[0] = rng.normal(0.0, 0.035, size=2)
    gyro_bias[0] = rng.normal(0.0, 0.003)
    for k in range(1, steps):
        accel_bias[k] = accel_bias[k - 1] + rng.normal(0.0, config.accel_bias_walk_std * np.sqrt(config.dt_s), size=2)
        gyro_bias[k] = gyro_bias[k - 1] + rng.normal(0.0, config.gyro_bias_walk_std * np.sqrt(config.dt_s))

    imu_accel = truth.accel_body + accel_bias + rng.normal(0.0, config.imu_accel_noise_std, size=(steps, 2))
    imu_gyro = truth.gyro_z + gyro_bias + rng.normal(0.0, config.imu_gyro_noise_std, size=steps)

    gnss_position = np.full((steps, 2), np.nan)
    gnss_available = np.zeros(steps, dtype=bool)
    gnss_period = _period_steps(config.gnss_rate_hz, config.dt_s)
    for k in range(0, steps, gnss_period):
        if not dropout_mask[k]:
            gnss_position[k] = truth.pose[k, :2] + rng.normal(0.0, config.gnss_pos_std, size=2)
            gnss_available[k] = True

    visual_pose = np.full((steps, 3), np.nan)
    visual_available = np.zeros(steps, dtype=bool)
    visual_period = _period_steps(config.visual_rate_hz, config.dt_s)
    visual_drift = rng.normal(0.0, 0.15, size=2)
    visual_yaw_drift = rng.normal(0.0, 0.015)
    for k in range(0, steps, visual_period):
        dt_since_update = visual_period * config.dt_s
        visual_drift += rng.normal(0.0, config.visual_drift_std_per_s * dt_since_update, size=2)
        visual_yaw_drift += rng.normal(0.0, 0.0025 * dt_since_update)
        visual_pose[k, :2] = truth.pose[k, :2] + visual_drift + rng.normal(0.0, config.visual_pos_std, size=2)
        visual_pose[k, 2] = wrap_angle(truth.pose[k, 2] + visual_yaw_drift + rng.normal(0.0, config.visual_yaw_std))
        visual_available[k] = True

    odom_speed = np.full(steps, np.nan)
    odom_available = np.zeros(steps, dtype=bool)
    odom_period = _period_steps(config.odom_rate_hz, config.dt_s)
    for k in range(0, steps, odom_period):
        odom_speed[k] = truth.speed[k] + rng.normal(0.0, config.odom_speed_std)
        odom_available[k] = True

    return Measurements(
        imu_accel_body=imu_accel,
        imu_gyro_z=imu_gyro,
        odom_speed=odom_speed,
        visual_pose=visual_pose,
        gnss_position=gnss_position,
        gnss_available=gnss_available,
        visual_available=visual_available,
        odom_available=odom_available,
        dropout_mask=dropout_mask,
    )


def _period_steps(rate_hz: float, dt_s: float) -> int:
    return max(1, int(round(1.0 / (rate_hz * dt_s))))

