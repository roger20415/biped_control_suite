import numpy as np
from geometry_msgs.msg import Quaternion, Vector3
from numpy.typing import NDArray
from typing import Mapping
from .config import Config
from .linear_algebra_utils import LinearAlgebraUtils

class JointTargetsCalculator():
    def __init__(self):
        self.p_W: dict[str, Vector3] = {} # points in the World frame
        self.p_B: dict[str, Vector3] = {} # points in the Baselink frame
        self.p_L: dict[str, Vector3] = {} # points in the Leg frame
        self.p_uw: dict[str, tuple[float, float]] = {} # points in the (u,w) coordinates in the Leg frame
        self.joint_phi: dict[str, float] = {} # +u-axis rotation in the Leg frame
        self.joint_theta: dict[str, float] = {} # -v-axis rotation in the Leg frame
        self.joint_targets: dict[str, float] = {} # final joint angle command

    def calc_joint_targets(self, p_W: Mapping[str, Vector3], q_W_baselink: Quaternion) -> dict[str, float]:
        self.p_W = dict(p_W)
        self.p_W["foot"] = Vector3(x=p_W["target"].x,
                                   y=p_W["target"].y,
                                   z=p_W["target"].z + Config.L_FOOT)
        self._transform_points_world_to_baselink(q_W_baselink)

    def _transform_points_world_to_baselink(self, q_W_baselink: Quaternion) -> None:
        T_BW = self._calc_T_BW(q_W_baselink, self.p_W["baselink"])
        self.p_B["hip"] = LinearAlgebraUtils.transform_point(T_BW, self.p_W["hip"])
        self.p_B["foot"] = LinearAlgebraUtils.transform_point(T_BW, self.p_W["foot"])

    def _calc_T_BW(self, q_W_baselink: Quaternion, p_W_baselink: Vector3) -> NDArray[np.float64]:
        R_WB = LinearAlgebraUtils.quaternion_to_rotation_matrix(q_W_baselink)
        T_WB = LinearAlgebraUtils.combine_transformation_matrix(R_WB, p_W_baselink)
        T_BW = LinearAlgebraUtils.invert_transformation_matrix(T_WB)
        return T_BW
    
    def _calc_R_BL(self) -> NDArray[np.float64]:
        """
        The Leg frame must follow the following three rules:
        1. The origin is located at the hip joint.
        2. The u-axis is parallel to the x-axis of the Baselink frame.
        3. The uw-plane passes through the foot point.
        """
        foot_to_hip_vec = Vector3(x=self.p_B["hip"].x - self.p_B["foot"].x,
                                  y=self.p_B["hip"].y - self.p_B["foot"].y,
                                  z=self.p_B["hip"].z - self.p_B["foot"].z)
        norm_w = np.linalg.norm(np.array([foot_to_hip_vec.y, foot_to_hip_vec.z])) # length of the original w vector 
        u = np.array([1.0, 0.0, 0.0])
        w = np.array([0.0, foot_to_hip_vec.y/norm_w, foot_to_hip_vec.z/norm_w])
        v = -np.cross(u, w)

        return np.column_stack((u, v, w))