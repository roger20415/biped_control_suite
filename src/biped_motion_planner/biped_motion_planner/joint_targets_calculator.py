from geometry_msgs.msg import Quaternion, Vector3
from typing import Mapping
from .linear_algebra_utils import LinearAlgebraUtils

L_FOOT: float = 1.0

class JointTargetsCalculator():
    def calc_joint_targets(self, p_W: Mapping[str, Vector3], q_W_baselink: Quaternion) -> dict[str, float]:
        pass