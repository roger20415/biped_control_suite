import numpy as np
from geometry_msgs.msg import Quaternion, Vector3
from numpy.typing import NDArray
from typing import Mapping
from .linear_algebra_utils import LinearAlgebraUtils

L_FOOT: float = 1.0

class JointTargetsCalculator():
    def __init__(self):
        self.p_W: dict[str, Vector3] = {}
        self.p_B: dict[str, Vector3] = {}
        self.p_L: dict[str, Vector3] = {}
        self.p_uw: dict[str, tuple[float, float]] = {}

    def calc_joint_targets(self, p_W: Mapping[str, Vector3], q_W_baselink: Quaternion) -> dict[str, float]:
        self.p_W = dict(p_W)
        self.p_W["foot"] = Vector3(p_W["target"].x, p_W["target"].y, p_W["target"].z + L_FOOT)
        self._transform_points_world_to_baselink(q_W_baselink)

    def _transform_points_world_to_baselink(self, q_W_baselink: Quaternion) -> None:
        T_BW = self._calc_T_BW(q_W_baselink, self.p_W["baselink"])
        self.p_B["hip"] = LinearAlgebraUtils.transform_point(T_BW, self.p_W["hip"])
        self.p_B["foot"] = LinearAlgebraUtils.transform_point(T_BW, self.p_W["foot"])

    def _calc_T_BW(self, q_W_baselink: Quaternion, p_W_baselink: Vector3) -> NDArray[np.float64]:
        R_WB = LinearAlgebraUtils.quaternion_to_rotation_matrix(q_W_baselink)
        T_WB = LinearAlgebraUtils.combine_transformation_matrix(R_WB, p_W_baselink)
        T_BW = LinearAlgebraUtils._invert_transformation_matrix(T_WB)
        return T_BW