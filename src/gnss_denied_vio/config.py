"""Configuration for the repeatable localization scenario."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SimulationConfig:
    """Parameters for a deterministic GNSS-denied localization run."""

    seed: int = 7
    duration_s: float = 90.0
    dt_s: float = 0.1
    dropout_start_s: float = 28.0
    dropout_duration_s: float = 34.0

    imu_accel_noise_std: float = 0.08
    imu_gyro_noise_std: float = 0.006
    accel_bias_walk_std: float = 0.0012
    gyro_bias_walk_std: float = 0.0002

    gnss_rate_hz: float = 1.0
    gnss_pos_std: float = 1.2

    visual_rate_hz: float = 3.0
    visual_pos_std: float = 0.65
    visual_yaw_std: float = 0.05
    visual_drift_std_per_s: float = 0.045

    odom_rate_hz: float = 10.0
    odom_speed_std: float = 0.08

    def steps(self) -> int:
        return int(round(self.duration_s / self.dt_s)) + 1

    def dropout_end_s(self) -> float:
        return self.dropout_start_s + self.dropout_duration_s
