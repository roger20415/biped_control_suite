import numpy as np
import rclpy
import sys
from geometry_msgs.msg import Quaternion, Vector3
from numpy.typing import NDArray
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from std_msgs.msg import Float32, Float64MultiArray, String
from typing import Optional
from .config import Config, LegSide
from .linear_algebra_utils import LinearAlgebraUtils

PHASE_UPDATE_PERIOD: float = 0.05 # in seconds
SWING_FOOT_TOUCHDOWN_PERIOD:float = 5 # in seconds # SSP: single support phase
STANCE_LEG_JOINT_ALPHA: float = 5.0 # in degrees
SWING_TRAJECTORY_MID_HEIGHT: float = 0.0015 # in meters
VALID_LEG_SIDES: tuple[str, ...] = ("left", "right")
VALID_SUPPORT_SIDES: tuple[str, ...] = ("left", "right", "mid")
REQUIRED_P_W_KEYS: tuple[str] = ("baselink", "l_foot", "r_foot")
REQUIRED_Q_W_KEYS: tuple[str] = ("baselink", "l_foot", "r_foot")

class BipedMotionPlannerNode(Node):
    def __init__(self):
        super().__init__('biped_motion_planner_node')
        self._swing_leg_side: LegSide = "undefined"
        self._stance_leg_side: LegSide = "undefined"
        self._support_side: str = "undefined"
        self._p_W: dict[str, Optional[Vector3]] = {k: None for k in REQUIRED_P_W_KEYS}
        self._q_W: dict[str, Optional[Quaternion]] = {k: None for k in REQUIRED_Q_W_KEYS}

        self._timer = self.create_timer(PHASE_UPDATE_PERIOD, self._timer_callback)
        self.clock = self.get_clock()
        self.t0 = None

        qos_sensor = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        self._baselink_translate_subscriber_ = self.create_subscription(
            Vector3,
            '/baselink/translate',
            self._baselink_translate_callback,
            qos_sensor
        )
        self._l_foot_translate_subscriber_ = self.create_subscription(
            Vector3,
            '/l_foot/translate',
            self._l_foot_translate_callback,
            qos_sensor
        )
        self._r_foot_translate_subscriber_ = self.create_subscription(
            Vector3,
            '/r_foot/translate',
            self._r_foot_translate_callback,
            qos_sensor
        )
        self._baselink_quat_subscriber_ = self.create_subscription(
            Quaternion,
            '/baselink/quat',
            self._baselink_quat_callback,
            qos_sensor
        )
        self._l_foot_quat_subscriber_ = self.create_subscription(
            Quaternion,
            '/l_foot/quat',
            self._l_foot_quat_callback,
            qos_sensor
        )
        self._r_foot_quat_subscriber_ = self.create_subscription(
            Quaternion,
            '/r_foot/quat',
            self._r_foot_quat_callback,
            qos_sensor
        )

    def _baselink_translate_callback(self, msg: Vector3) -> None:
        self._p_W["baselink"] = msg
    def _l_foot_translate_callback(self, msg: Vector3) -> None:
        self._p_W["l_foot"] = msg
    def _r_foot_translate_callback(self, msg: Vector3) -> None:
        self._p_W["r_foot"] = msg
    def _baselink_quat_callback(self, msg: Quaternion) -> None:
        self._q_W["baselink"] = msg
    def _l_foot_quat_callback(self, msg: Quaternion) -> None:
        self._q_W["l_foot"] = msg
    def _r_foot_quat_callback(self, msg: Quaternion) -> None:
        self._q_W["r_foot"] = msg

    def _timer_callback(self) -> None:
        now = self.clock.now()
        if self.t0 is None:
            self.t0 = now
            return
        t = (now - self.t0).nanoseconds * 1e-9
        s = max(0.0, min(t / self.T, 1.0)) # the phase
        self.on_swing_foot_touchdown(s)

    def on_swing_foot_touchdown(self, s: float) -> None:
        if not self._check_all_data_received():
            self.get_logger().warn("Not all required data received yet.")
            return
        current_stance_angle: float = self._calc_current_stance_angle(s)
        current_swing_target: NDArray[np.float64] = self._calc_current_swing_target(s)

    def _calc_current_stance_angle(self, s: float) -> float:
        return STANCE_LEG_JOINT_ALPHA * (1.0 - s)
    
    def _calc_current_swing_target(self, s: float) -> NDArray[np.float64]:
        baselink_move: float = self._calc_baselink_move()
        p_S_baselink_target: NDArray[np.float64] = self._calc_p_S_baselink_target(baselink_move)
    
    def _calc_baselink_move(self) -> float:
        return (Config.THIGH_LEN + Config.CALF_LEN)*np.sin(np.deg2rad(STANCE_LEG_JOINT_ALPHA))
    
    def _calc_p_S_baselink_target(self, baselink_move: float) -> NDArray[np.float64]:
        R_WB = LinearAlgebraUtils.quaternion_to_rotation_matrix(self._q_W["baselink"])
        xB_W = R_WB[:, 0]
        xB_S = np.array([xB_W[0], xB_W[1], 0.0])
        xB_S_norm = xB_S / np.linalg.norm(xB_S)
        p_S_baselink = np.array([
            self._p_W["baselink"].x,
            self._p_W["baselink"].y,
            0.0
        ])
        return p_S_baselink + baselink_move * xB_S_norm
    
    def _check_all_data_received(self) -> bool:
        for k in REQUIRED_P_W_KEYS:
            if self._p_W[k] is None:
                return False
        for k in REQUIRED_Q_W_KEYS:
            if self._q_W[k] is None:
                return False
        return True

def main(args=None):
    rclpy.init(args=args)
    node = BipedMotionPlannerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        sys.exit(0)

if __name__ == '__main__':
    main()