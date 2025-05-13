# ruff: noqa: N806
import numpy as np


class KalmanXYZ:
    def __init__(self, dt=1/30, q=5e-3, r=5e-3):
        self.dt = dt
        self.x  = np.zeros(6, dtype=np.float32)               # state
        self.P  = np.eye(6, dtype=np.float32) * 1.            # covariance
        self.Q  = np.eye(6, dtype=np.float32) * q             # motion noise
        self.R  = np.eye(3, dtype=np.float32) * r             # measurement noise
        self.H  = np.hstack([np.eye(3), np.zeros((3, 3))]).astype(np.float32)  # pos-only meas

    def _F(self, dt):  # transition matrix rebuilt each step  # noqa: N802
        F = np.eye(6, dtype=np.float32)
        F[:3, 3:] = np.eye(3, dtype=np.float32) * dt
        return F

    def predict(self, dt=None):
        dt = dt or self.dt
        F = self._F(dt)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self.Q

    def update(self, z):
        y = z.astype(np.float32) - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x += K @ y
        I = np.eye(6, dtype=np.float32)  # noqa: E741
        self.P = (I - K @ self.H) @ self.P
