import numpy as np
import warnings
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
        self.p_uw: dict[str, NDArray[np.float64]] = {} # points in the (u,w) coordinates in the Leg frame
        self.joint_phi: dict[str, float] = {} # +u-axis rotation in the Leg frame
        self.joint_theta: dict[str, float] = {} # -v-axis rotation in the Leg frame
        self.joint_targets: dict[str, float] = {} # final joint angle commands

    def calc_joint_targets(self, p_W: Mapping[str, Vector3], q_W_baselink: Quaternion) -> dict[str, float]:
        self.p_W = dict(p_W)
        self.p_W["foot"] = Vector3(x=p_W["target"].x,
                                   y=p_W["target"].y,
                                   z=p_W["target"].z + Config.L_FOOT)
        R_WB, T_BW = self._calc_WB_transforms(q_W_baselink, self.p_W["baselink"])
        self.p_B.update(self._transform_points_World_to_Baselink(T_BW))
        R_BL, T_LB = self._calc_BL_transforms()
        self.p_L.update(self._transform_points_Baselink_to_Leg(T_LB))
        self.p_uw.update(self._transform_points_Leg_to_uw())
        self.joint_phi["leg"] = self._calc_phi_BL()
        e_L_proj = self._project_gravity_to_uw_plane(R_WB.T, R_BL.T)
        self.p_uw["ankle"] = self._calc_p_uw_ankle(e_L_proj)

    def _transform_points_World_to_Baselink(self, T_BW: NDArray[np.float64]) -> dict[str, Vector3]:
        p_B_hip = LinearAlgebraUtils.transform_point(T_BW, self.p_W["hip"])
        p_B_foot = LinearAlgebraUtils.transform_point(T_BW, self.p_W["foot"])
        return {"hip": p_B_hip, "foot": p_B_foot}

    def _calc_WB_transforms(self, 
                            q_W_baselink: Quaternion, 
                            p_W_baselink: Vector3) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        R_WB = LinearAlgebraUtils.quaternion_to_rotation_matrix(q_W_baselink)
        T_WB = LinearAlgebraUtils.combine_transformation_matrix(R_WB, p_W_baselink)
        T_BW = LinearAlgebraUtils.invert_transformation_matrix(T_WB)
        return R_WB, T_BW
    
    def _transform_points_Baselink_to_Leg(self, T_LB: NDArray[np.float64]) -> dict[str, Vector3]:
        p_L_foot = LinearAlgebraUtils.transform_point(T_LB, self.p_B["foot"])
        return {"foot": p_L_foot}

    def _calc_BL_transforms(self) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        R_BL = self._calc_R_BL()
        T_BL = LinearAlgebraUtils.combine_transformation_matrix(R_BL, self.p_B["hip"])
        T_LB = LinearAlgebraUtils.invert_transformation_matrix(T_BL)
        return R_BL, T_LB
    
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
        if norm_w < 1e-12:
            raise ValueError("Cannot normalize zero-length vector")        
        u = np.array([1.0, 0.0, 0.0])
        w = np.array([0.0, foot_to_hip_vec.y/norm_w, foot_to_hip_vec.z/norm_w])
        v = -np.cross(u, w)

        return np.column_stack((u, v, w))

    def _transform_points_Leg_to_uw(self) -> dict[str, NDArray[np.float64]]:
        p_uw_foot = np.array([self.p_L["foot"].x, self.p_L["foot"].z])
        return {"foot": p_uw_foot}
    
    def _calc_phi_BL(self) -> float:
        delta_y = self.p_B["hip"].y - self.p_B["foot"].y
        delta_z = self.p_B["hip"].z - self.p_B["foot"].z
        return np.degrees(np.arctan2(-delta_y, delta_z))

    def _project_gravity_to_uw_plane(self, R_BW: NDArray[np.float64], R_LB: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Returns the projection of World -Z (gravity) onto the Leg's uw-plane,
        expressed as (u, v, w) scalar components.
        """
        e_W = np.array([0.0, 0.0, -1.0]) # gravity vector in World frame
        e_B = R_BW @ e_W
        R_BL = R_LB.T
        nhat_B_uw = -R_BL[:, 1]

        P = (np.eye(3) - np.outer(nhat_B_uw, nhat_B_uw)) @ e_B
        norm_P = np.linalg.norm(P)

        if norm_P < 1e-12:
            warnings.warn(
                "Gravity is parallel to the uw-plane normal; projection is degenerated. Returning -w_L.",
                category=RuntimeWarning,
                stacklevel=2,
            )
            return np.array([0, 0, -1])

        e_B_proj = P / norm_P
        e_L_proj = R_LB @ e_B_proj
        return e_L_proj
    
    def _calc_p_uw_ankle(self, e_L_proj: NDArray[np.float64]) -> NDArray[np.float64]:
        e_uw_proj: NDArray[np.float64] = e_L_proj[[0, 2]]
        e_uw_proj_norm = LinearAlgebraUtils.normalize_vec(e_uw_proj)
        d_uw_ankle = Config.L_ANKLE * e_uw_proj_norm
        return self.p_uw["foot"] - d_uw_ankle