# ruff: noqa: N806 N803

import numpy as np


class KalmanXYZ:
    def __init__(self, dt=1/30, q=1e-2, r=5e-3):
        self.dt = dt
        self.x  = np.zeros(6)          # state
        self.P  = np.eye(6) * 1.       # covariance
        self.Q  = np.eye(6) * q        # motion noise
        self.R  = np.eye(3) * r        # meas  noise
        self.H  = np.hstack([np.eye(3), np.zeros((3,3))])  # pos-only meas

    def _F(self, dt):                  # transition matrix rebuilt each step  # noqa: N802
        F = np.eye(6)
        F[:3, 3:] = np.eye(3) * dt
        return F

    def predict(self, dt=None):
        dt = dt or self.dt
        F  = self._F(dt)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self.Q

    def update(self, z):
        y  = z - self.H @ self.x
        S  = self.H @ self.P @ self.H.T + self.R
        K  = self.P @ self.H.T @ np.linalg.inv(S)
        self.x += K @ y
        I  = np.eye(6)  # noqa: E741
        self.P = (I - K @ self.H) @ self.P