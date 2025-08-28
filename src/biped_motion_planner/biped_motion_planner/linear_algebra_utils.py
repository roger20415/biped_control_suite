import numpy as np
from geometry_msgs.msg import Quaternion, Vector3
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation as _Rot
from typing import Sequence


class LinearAlgebraUtils():

    @staticmethod
    def quaternion_to_rotation_matrix(q: Quaternion) -> NDArray[np.float64]:
        q_norm = LinearAlgebraUtils.normalize_vec([q.x, q.y, q.z, q.w])
        return _Rot.from_quat(q_norm, scalar_first=False).as_matrix().astype(np.float64)
    
    @staticmethod
    def combine_transformation_matrix(R: NDArray[np.float64], t: Vector3) -> NDArray[np.float64]:
        if R.shape != (3, 3):
            raise ValueError(f"R must be 3x3, got shape {R.shape}")
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3,  3] = np.array([t.x, t.y, t.z], dtype=np.float64)
        return T
    
    @staticmethod
    def normalize_vec(vec: Sequence[float]) -> NDArray[np.float64]:
        arr = np.array(vec, dtype=np.float64)
        norm = np.linalg.norm(arr)
        if norm < 1e-12:
            raise ValueError("Cannot normalize zero-length vector")
        return arr / norm
    
    @staticmethod
    def _invert_transformation_matrix(T: NDArray[np.float64]) -> NDArray[np.float64]:
        if T.shape != (4, 4):
            raise ValueError(f"T must be 4x4, got shape {T.shape}")
        R = T[:3, :3]
        t = T[:3,  3]
        T_inv = np.eye(4, dtype=np.float64)
        Rt = R.T
        T_inv[:3, :3] = Rt
        T_inv[:3,  3] = -Rt @ t
        return T_inv