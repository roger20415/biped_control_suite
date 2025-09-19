from typing import Mapping

import numpy as np
import sympy as sp
from geometry_msgs.msg import Quaternion, Vector3
from numpy.typing import NDArray

from .config import Config, LegSide, SupportSide, VALID_LEG_SIDES, VALID_SUPPORT_SIDES
from .linear_algebra_utils import LinearAlgebraUtils

STANCE_LEG_JOINT_ALPHA: float = 5.0  # in degrees
LAMBDA_FOR_SWING_END: float = 0.5  # between 0 and 1. lambda>0.5 baselink closer to support point and further from swing end.
SWING_TRAJECTORY_MID_HEIGHT: float = 0.0015 # in meters
REQUIRED_P_W_KEYS: tuple[str, ...] = ("baselink", "l_foot", "r_foot")
REQUIRED_Q_W_KEYS: tuple[str, ...] = ("baselink", "l_foot", "r_foot")

class SSTODSManager:
    def __init__(self):
        self._stance_side: LegSide = "undefined"
        self._swing_side: LegSide = "undefined"
        self._support_side: SupportSide = "undefined"
        self.validate_parameters()

    def validate_parameters(self) -> None:
        if STANCE_LEG_JOINT_ALPHA <= 0.0 or STANCE_LEG_JOINT_ALPHA >= 90.0:
            raise ValueError("STANCE_LEG_JOINT_ALPHA must be between 0 and 90 degrees.")
        if LAMBDA_FOR_SWING_END <= 0.0 or LAMBDA_FOR_SWING_END >= 1.0:
            raise ValueError("LAMBDA_FOR_SWING_END must be between 0 and 1.")

    def set_stance_side(self, side: LegSide) -> None:
        if side not in VALID_LEG_SIDES:
            raise ValueError("Invalid leg side.")
        self._stance_side = side

    def set_swing_side(self, side: LegSide) -> None:
        if side not in VALID_LEG_SIDES:
            raise ValueError("Invalid leg side.")
        self._swing_side = side
    
    def set_support_side(self, side: SupportSide) -> None:
        if side not in VALID_SUPPORT_SIDES:
            raise ValueError("Invalid support side.")
        self._support_side = side

    def build_stance_of_s(self) -> sp.Expr:
        s = sp.symbols('s', real=True)
        stance_of_s = STANCE_LEG_JOINT_ALPHA * (1 - s)
        return sp.simplify(stance_of_s)

    def build_swing_of_s(self, p_W: Mapping[str, Vector3], q_W: Mapping[str, Quaternion]) -> NDArray[object]:
        if not self._if_subscribe_data_ready(p_W, q_W):
            raise ValueError("Position or orientation data is not yet received.")
        if not self._if_side_defined():
            raise ValueError("Stance or swing or support side is undefined.")
        baselink_move_dist_xB: float = self._calc_baselink_move_dist_xB()
        p_S_baselink_target: NDArray[np.float64] = self._calc_p_S_baselink_target(baselink_move_dist_xB, p_W, q_W)
        p_S_stance: NDArray[np.float64] = self._calc_p_S_stance(p_W, q_W)
        p_S_swing_end: NDArray[np.float64] = self._calc_p_S_swing_end(p_S_baselink_target, p_S_stance)
        p_W_swing_start: NDArray[np.float64] = self._calc_p_W_swing_start(p_W, q_W)
        swing_of_s = self._build_swing_of_s(p_W_swing_start, p_S_swing_end)
        return swing_of_s

    def _if_subscribe_data_ready(self, p_W: Mapping[str, Vector3], q_W: Mapping[str, Quaternion]) -> bool:
        for key in REQUIRED_P_W_KEYS:
            if p_W.get(key) is None:
                return False
        for key in REQUIRED_Q_W_KEYS:
            if q_W.get(key) is None:
                return False
        return True
    
    def _if_side_defined(self) -> bool:
        return self._stance_side in VALID_LEG_SIDES and self._swing_side in VALID_LEG_SIDES and self._support_side in VALID_SUPPORT_SIDES

    def _build_swing_of_s(self, p_W_swing_start: NDArray[np.float64], p_S_swing_end: NDArray[np.float64]) -> NDArray[object]:
        p_S_swing_start = np.array([p_W_swing_start[0], p_W_swing_start[1], 0.0], dtype=np.float64)
        p_S_swing_mid = (p_S_swing_start + p_S_swing_end) / 2.0
        p_W_swing_mid = np.array([p_S_swing_mid[0], p_S_swing_mid[1], SWING_TRAJECTORY_MID_HEIGHT], dtype=np.float64)

        s = sp.symbols('s', real=True)
        raw_swing_of_s = ((1-s)**2)*p_W_swing_start + 2*(1-s)*s*p_W_swing_mid + (s**2)*p_S_swing_end
        swing_of_s = np.empty(3, dtype=object)
        for i in range(3):
            swing_of_s[i] = sp.simplify(raw_swing_of_s[i])
        
        return swing_of_s

    def _calc_baselink_move_dist_xB(self) -> float:
        return (Config.THIGH_LEN + Config.CALF_LEN)*np.sin(np.deg2rad(STANCE_LEG_JOINT_ALPHA))

    def _calc_p_S_baselink_target(self, baselink_move: float, p_W: Mapping[str, Vector3], q_W: Mapping[str, Quaternion]) -> NDArray[np.float64]:
        R_WB = LinearAlgebraUtils.quaternion_to_rotation_matrix(q_W["baselink"])
        xB_W = R_WB[:, 0]
        xB_S = np.array([xB_W[0], xB_W[1], 0.0])
        xB_S_norm = LinearAlgebraUtils.normalize_vec(xB_S)
        p_S_baselink = np.array([
            p_W["baselink"].x,
            p_W["baselink"].y,
            0.0
        ])
        return p_S_baselink + baselink_move * xB_S_norm
    
    def _calc_p_S_stance(self, p_W: Mapping[str, Vector3], q_W: Mapping[str, Quaternion]) -> NDArray[np.float64]:
        if self._stance_side == "left":
            xLFOOT_W_norm = self._calc_xFOOT_W_norm(q_W["l_foot"])
            p_W_l_foot = np.array([p_W["l_foot"].x, p_W["l_foot"].y, p_W["l_foot"].z], dtype=np.float64)
            p_W_l_foot_mid =  p_W_l_foot - Config.FOOT_LINK_X_SEMI_LENGTH * xLFOOT_W_norm
            p_S_stance = np.array([p_W_l_foot_mid[0], p_W_l_foot_mid[1], 0.0], dtype=np.float64)
        elif self._stance_side == "right":
            xRFOOT_W_norm = self._calc_xFOOT_W_norm(q_W["r_foot"])
            p_W_r_foot = np.array([p_W["r_foot"].x, p_W["r_foot"].y, p_W["r_foot"].z], dtype=np.float64)
            p_W_r_foot_mid =  p_W_r_foot - Config.FOOT_LINK_X_SEMI_LENGTH * xRFOOT_W_norm
            p_S_stance = np.array([p_W_r_foot_mid[0], p_W_r_foot_mid[1], 0.0], dtype=np.float64)
        else:
            raise ValueError("Stance side is undefined.")
        return p_S_stance
    

    def _calc_xFOOT_W_norm(self, q_W_foot: Quaternion) -> NDArray[np.float64]:
        R_W_FOOT: NDArray[np.float64] = LinearAlgebraUtils.quaternion_to_rotation_matrix(q_W_foot)
        xFOOT_W = R_W_FOOT[:, 0]
        return LinearAlgebraUtils.normalize_vec(xFOOT_W)

    def _calc_zFOOT_W_norm(self, q_W_foot: Quaternion) -> NDArray[np.float64]:
        R_W_FOOT: NDArray[np.float64] = LinearAlgebraUtils.quaternion_to_rotation_matrix(q_W_foot)
        zFOOT_W = R_W_FOOT[:, 2]
        return LinearAlgebraUtils.normalize_vec(zFOOT_W)

    def _calc_p_S_swing_end(self, p_S_baselink_target: NDArray[np.float64], p_S_stance: NDArray[np.float64]) -> NDArray[np.float64]:
        return (p_S_baselink_target - (1-LAMBDA_FOR_SWING_END) * p_S_stance) / LAMBDA_FOR_SWING_END

    def _calc_p_W_swing_start(self, p_W: Mapping[str, Vector3], q_W: Mapping[str, Quaternion]) -> NDArray[np.float64]:
        if self._swing_side == "left":
            xLFOOT_W_norm = self._calc_xFOOT_W_norm(q_W["l_foot"])
            p_W_l_foot = np.array([p_W["l_foot"].x, p_W["l_foot"].y, p_W["l_foot"].z], dtype=np.float64)
            p_W_l_foot_mid =  p_W_l_foot - Config.FOOT_LINK_X_SEMI_LENGTH * xLFOOT_W_norm
            zLFOOT_W_norm = self._calc_zFOOT_W_norm(q_W["l_foot"])
            p_W_swing_start = p_W_l_foot_mid - Config.FOOT_LEN*zLFOOT_W_norm
        elif self._swing_side == "right":
            xRFOOT_W_norm = self._calc_xFOOT_W_norm(q_W["r_foot"])
            p_W_r_foot = np.array([p_W["r_foot"].x, p_W["r_foot"].y, p_W["r_foot"].z], dtype=np.float64)
            p_W_r_foot_mid =  p_W_r_foot - Config.FOOT_LINK_X_SEMI_LENGTH * xRFOOT_W_norm
            zRFOOT_W_norm = self._calc_zFOOT_W_norm(q_W["r_foot"])
            p_W_swing_start = p_W_r_foot_mid - Config.FOOT_LEN*zRFOOT_W_norm
        else:
            raise ValueError("Swing side is undefined.")
        return p_W_swing_start