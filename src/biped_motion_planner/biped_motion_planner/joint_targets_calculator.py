import numpy as np
from geometry_msgs.msg import Quaternion, Vector3
from numpy.typing import NDArray
from typing import Mapping
from .linear_algebra_utils import LinearAlgebraUtils

L_FOOT: float = 1.0

class JointTargetsCalculator():
    def calc_joint_targets(self, p_W: Mapping[str, Vector3], q_W_baselink: Quaternion) -> dict[str, float]:
        T_BW = self._calc_T_BW(q_W_baselink, p_W["baselink"])

    def _calc_T_BW(self, q_W_baselink: Quaternion, p_W_baselink: Vector3) -> NDArray[np.float64]:
        R_WB = LinearAlgebraUtils.quaternion_to_rotation_matrix(q_W_baselink)
        T_WB = LinearAlgebraUtils.combine_transformation_matrix(R_WB, p_W_baselink)
        T_BW = LinearAlgebraUtils._invert_transformation_matrix(T_WB)
        return T_BW