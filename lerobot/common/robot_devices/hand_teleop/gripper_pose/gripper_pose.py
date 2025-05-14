# ruff: noqa: N803 N806 
from typing import Optional

import numpy as np
from scipy.spatial.transform import Rotation


class GripperPose:
    def __init__(self, pos: np.ndarray, rot: np.ndarray, open_degree: float, keypoints: Optional[list[np.ndarray]] = None):
        assert pos.shape == (3,), "pos must be a 3-element vector"
        assert rot.shape == (3, 3), "rot must be a rh 3x3 matrix in column-vector format"
        self.pos = pos.astype(np.float32)
        self.rot = rot.astype(np.float32)
        self.open_degree = float(open_degree)
        self._keypoints = [kp.astype(np.float32) for kp in keypoints] if keypoints is not None else []

    @staticmethod
    def zero() -> 'GripperPose':
        """Creates a zero pose with identity orientation and gripper closed."""
        return GripperPose(
            pos=np.zeros(3, dtype=np.float32),
            rot=np.eye(3, dtype=np.float32),
            open_degree=0.0
        )
    
    @staticmethod
    def from_matrix(T: np.ndarray, open_degree: float = 0.0, keypoints: Optional[list[np.ndarray]] = None) -> 'GripperPose':
        """Creates a GripperPose from a 4x4 transformation matrix."""
        assert T.shape == (4, 4), "Expected a 4x4 transformation matrix"
        pos = T[:3, 3]
        rot = T[:3, :3]
        return GripperPose(pos, rot, open_degree, keypoints)

    def to_matrix(self) -> np.ndarray:
        """Returns the 4x4 transformation matrix representing the pose."""
        T = np.eye(4, dtype=np.float32)
        T[:3, :3] = self.rot
        T[:3, 3] = self.pos
        return T

    @property
    def rot_euler(self) -> np.ndarray:
        """Returns orientation as intrinsic Euler angles (Z→Y→X), [roll, pitch, yaw] order."""
        return np.degrees(Rotation.from_matrix(self.rot).as_euler("ZYX")[::-1])
    
    @property
    def is_rh(self) -> bool:
        """Returns True if the rotation matrix represents a right-handed coordinate system."""
        return np.isclose(np.linalg.det(self.rot), 1.0, atol=1e-3)

    @property
    def gripper_pts(self) -> list[np.ndarray]:
        """Returns the list of associated keypoints (e.g. origin, tip, thumb base/tip)."""
        return self._keypoints

    def flip_axis(self, axis: str) -> None:
        """Flips the specified intrinsic Euler axis ('x', 'y', or 'z') in-place."""
        assert axis in {'x', 'y', 'z'}, "Axis must be one of 'x', 'y', or 'z'"
        eul = Rotation.from_matrix(self.rot).as_euler("ZYX")
        axis_to_idx = {'z': 0, 'y': 1, 'x': 2}
        eul[axis_to_idx[axis]] *= -1
        self.rot = Rotation.from_euler("ZYX", eul).as_matrix().astype(np.float32)

    def copy(self) -> 'GripperPose':
        """Returns a deep copy of the pose and keypoints."""
        return GripperPose(self.pos.copy(), self.rot.copy(), self.open_degree, [pt.copy() for pt in self._keypoints])

    def clip_(self, safe_range: dict[str, tuple[float, float]]) -> None:
        """In-place clipping of position and gripper open degree based on safe_range dict."""
        self.pos = np.array([np.clip(v, *safe_range[k]) for v, k in zip(self.pos, "xyz", strict=False)], dtype=np.float32)
        self.open_degree = float(np.clip(self.open_degree, *safe_range["g"]))
        
    def _apply_op(self, p_fn, r_fn):
        """Apply a position and rotation operation to pose and keypoints."""
        self.pos = p_fn(self.pos)
        self.rot = r_fn(self.rot)
        self._keypoints = [p_fn(pt) for pt in self._keypoints]

    def transform_pose(self, R, t):
        """Actively apply rotation R and translation t to the pose and keypoints."""
        self._apply_op(lambda p: p + t, lambda r: R @ r)

    def inverse_transform_pose(self, R, t): 
        """Undo an active transform defined by rotation R and translation t.
        Equivalently this gives the transformation to go from initial to the current one.
        """
        self._apply_op(lambda p: p - t, lambda r: r @ R.T )

    def change_basis(self, R, t):
        """Express the pose and keypoints in a new coordinate frame defined by (R, t)."""
        self._apply_op(lambda p: R.T @ (p - t), lambda r: R.T @ r @ R)

    def revert_basis(self, R, t):
        """Revert a previously-applied change of coordinate frame (basis)."""
        self._apply_op(lambda p: R @ p + t, lambda r: R @ r @ R.T)
         


    
