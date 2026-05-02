"""Small EKF used by the simulated visual-inertial localization stack."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


X = 0
Y = 1
VX = 2
VY = 3
YAW = 4
BAX = 5
BAY = 6
BGZ = 7
STATE_SIZE = 8


def wrap_angle(angle: float | np.ndarray) -> float | np.ndarray:
    """Wrap an angle to [-pi, pi)."""

    return (angle + np.pi) % (2.0 * np.pi) - np.pi


@dataclass(frozen=True)
class ProcessNoise:
    accel_noise_std: float
    gyro_noise_std: float
    accel_bias_walk_std: float
    gyro_bias_walk_std: float


class VisualInertialEKF:
    """Planar EKF with IMU prediction and position, yaw, and speed updates."""

    def __init__(self, initial_state: np.ndarray, initial_covariance: np.ndarray, noise: ProcessNoise):
        self.x = initial_state.astype(float).copy()
        self.p = initial_covariance.astype(float).copy()
        self.noise = noise
        self.x[YAW] = float(wrap_angle(self.x[YAW]))

    def predict(self, accel_body_x: float, accel_body_y: float, gyro_z: float, dt: float) -> None:
        accel_x = accel_body_x - self.x[BAX]
        accel_y = accel_body_y - self.x[BAY]
        yaw_rate = gyro_z - self.x[BGZ]

        yaw = self.x[YAW]
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)
        accel_world_x = cos_yaw * accel_x - sin_yaw * accel_y
        accel_world_y = sin_yaw * accel_x + cos_yaw * accel_y

        dt2 = dt * dt
        self.x[X] += self.x[VX] * dt + 0.5 * accel_world_x * dt2
        self.x[Y] += self.x[VY] * dt + 0.5 * accel_world_y * dt2
        self.x[VX] += accel_world_x * dt
        self.x[VY] += accel_world_y * dt
        self.x[YAW] = float(wrap_angle(self.x[YAW] + yaw_rate * dt))

        f = np.eye(STATE_SIZE)
        f[X, VX] = dt
        f[Y, VY] = dt

        d_ax_d_yaw = -accel_world_y
        d_ay_d_yaw = accel_world_x
        f[X, YAW] = 0.5 * d_ax_d_yaw * dt2
        f[Y, YAW] = 0.5 * d_ay_d_yaw * dt2
        f[VX, YAW] = d_ax_d_yaw * dt
        f[VY, YAW] = d_ay_d_yaw * dt

        f[X, BAX] = -0.5 * cos_yaw * dt2
        f[X, BAY] = 0.5 * sin_yaw * dt2
        f[Y, BAX] = -0.5 * sin_yaw * dt2
        f[Y, BAY] = -0.5 * cos_yaw * dt2
        f[VX, BAX] = -cos_yaw * dt
        f[VX, BAY] = sin_yaw * dt
        f[VY, BAX] = -sin_yaw * dt
        f[VY, BAY] = -cos_yaw * dt
        f[YAW, BGZ] = -dt

        q_acc = self.noise.accel_noise_std**2
        q_gyro = self.noise.gyro_noise_std**2
        q = np.diag(
            [
                0.25 * q_acc * dt2 * dt2,
                0.25 * q_acc * dt2 * dt2,
                q_acc * dt2,
                q_acc * dt2,
                q_gyro * dt2,
                self.noise.accel_bias_walk_std**2 * dt,
                self.noise.accel_bias_walk_std**2 * dt,
                self.noise.gyro_bias_walk_std**2 * dt,
            ]
        )
        self.p = f @ self.p @ f.T + q
        self.p = 0.5 * (self.p + self.p.T)

    def update_gnss_position(self, position_xy: np.ndarray, position_std: float) -> None:
        h = np.zeros((2, STATE_SIZE))
        h[0, X] = 1.0
        h[1, Y] = 1.0
        residual = position_xy - self.x[[X, Y]]
        r = np.diag([position_std**2, position_std**2])
        self._update(residual, h, r)

    def update_visual_pose(self, pose_xy_yaw: np.ndarray, position_std: float, yaw_std: float) -> None:
        h = np.zeros((3, STATE_SIZE))
        h[0, X] = 1.0
        h[1, Y] = 1.0
        h[2, YAW] = 1.0
        residual = pose_xy_yaw - self.x[[X, Y, YAW]]
        residual[2] = wrap_angle(residual[2])
        r = np.diag([position_std**2, position_std**2, yaw_std**2])
        self._update(residual, h, r)

    def update_forward_speed(self, speed_mps: float, speed_std: float) -> None:
        yaw = self.x[YAW]
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)
        predicted_speed = cos_yaw * self.x[VX] + sin_yaw * self.x[VY]
        residual = np.array([speed_mps - predicted_speed])

        h = np.zeros((1, STATE_SIZE))
        h[0, VX] = cos_yaw
        h[0, VY] = sin_yaw
        h[0, YAW] = -sin_yaw * self.x[VX] + cos_yaw * self.x[VY]
        r = np.array([[speed_std**2]])
        self._update(residual, h, r)

    def _update(self, residual: np.ndarray, h: np.ndarray, r: np.ndarray) -> None:
        s = h @ self.p @ h.T + r
        k = self.p @ h.T @ np.linalg.inv(s)
        identity = np.eye(STATE_SIZE)

        self.x += k @ residual
        self.x[YAW] = float(wrap_angle(self.x[YAW]))

        joseph = identity - k @ h
        self.p = joseph @ self.p @ joseph.T + k @ r @ k.T
        self.p = 0.5 * (self.p + self.p.T)

    def snapshot(self) -> np.ndarray:
        return self.x.copy()

    def covariance_snapshot(self) -> np.ndarray:
        return self.p.copy()

