import numpy as np
import rclpy
import sys
from geometry_msgs.msg import Quaternion
from numpy.typing import NDArray
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from std_msgs.msg import Float64MultiArray
from typing import Optional
from .config import Config, LegSide
from .linear_algebra_utils import LinearAlgebraUtils

_COM_KEYS: tuple[str, ...] = (
    "baselink",
    "back", "sacrum",
    "l_hip", "r_hip",
    "l_thigh", "r_thigh",
    "l_calf", "r_calf",
    "l_ankle", "r_ankle",
    "l_foot", "r_foot",
)
SACRUM_MOVE_THRESHOLD: float = 0.0065/50 # in meters (left to right foot distance: 0.0065)
SACRUM_MOVE_STEP: float = 0.018/50 # joint target command (sacrum joint limits: +-0.009)


class CounterweightControlNode(Node):
    def __init__(self):
        super().__init__('counterweight_control_node')
        self._support_side: LegSide = "left"
        self._p_W_joints_com: dict[str, np.ndarray] = {k: np.zeros(3, dtype=np.float64) for k in _COM_KEYS}
        self._sacrum_target: float = 0.0

        self.declare_parameter('publish_period', 0.05)  # 20 Hz
        self._publish_period: float = float(self.get_parameter('publish_period').value)

        self._q_W_baselink: Optional[Quaternion] = None

        qos_sensor = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        self._com_subscriber_ = self.create_subscription(
            Float64MultiArray,
            '/com',
            self._com_callback,
            qos_sensor
        )
        self._baselink_quat_subscriber_ = self.create_subscription(
            Quaternion,
            '/baselink/quat',
            self._baselink_quat_callback,
            qos_sensor
        )

        self._counterweight_publisher_ = self.create_publisher(
            Float64MultiArray,
            '/counterweight/joint_targets',
            10
        )

        self._timer = self.create_timer(self._publish_period, self._timer_callback)

    def _pub_counterweight_pos(self, counterweight_pos) -> None:
        msg = Float64MultiArray()
        msg.data = [float(i) for i in counterweight_pos]
        self._counterweight_publisher_.publish(msg)

    def _com_callback(self, msg: Float64MultiArray) -> None:
        data = np.asarray(msg.data, dtype=np.float64)
        expected = 3 * len(_COM_KEYS)
        if data.size != expected:
            self.get_logger().error(f"Expected {expected} values (got {data.size}).")
            return

        vecs = data.reshape(len(_COM_KEYS), 3)
        p_W_joints_com = {name: vecs[i] for i, name in enumerate(_COM_KEYS)}
        self._p_W_joints_com = p_W_joints_com

    def _calc_p_W_biped_com(self, joints_com: dict[str, NDArray[np.float64]]) -> NDArray[np.float64]:
        total_mass: float = (
            Config.BASELINK_MASS + Config.BACK_MASS + Config.SACRUM_MASS +
            Config.HIP_MASS * 2 +
            Config.THIGH_MASS * 2 +
            Config.CALF_MASS * 2 +
            Config.ANKLE_MASS * 2 +
            Config.FOOT_MASS * 2)
        weighted_sum: NDArray[np.float64] = (
            joints_com["baselink"] * Config.BASELINK_MASS +
            joints_com["back"] * Config.BACK_MASS +
            joints_com["sacrum"] * Config.SACRUM_MASS +
            (joints_com["l_hip"] + joints_com["r_hip"]) * Config.HIP_MASS +
            (joints_com["l_thigh"] + joints_com["r_thigh"]) * Config.THIGH_MASS +
            (joints_com["l_calf"] + joints_com["r_calf"]) * Config.CALF_MASS +
            (joints_com["l_ankle"] + joints_com["r_ankle"]) * Config.ANKLE_MASS +
            (joints_com["l_foot"] + joints_com["r_foot"]) * Config.FOOT_MASS
        )
        p_W_biped_com: NDArray[np.float64] = weighted_sum / total_mass
        return p_W_biped_com

    def _calc_p_S_support(self) -> NDArray[np.float64]:
        if self._support_side == "left":
            return np.array([self._p_W_joints_com["l_foot"][0], 
                             self._p_W_joints_com["l_foot"][1], 
                             0.0], dtype=np.float64)
        elif self._support_side == "right":
            return np.array([self._p_W_joints_com["r_foot"][0], 
                             self._p_W_joints_com["r_foot"][1], 
                             0.0], dtype=np.float64)
        else:
            raise ValueError("Support side is undefined.")
    
    def _timer_callback(self) -> None:
        vec_S_com_to_support = self._calc_vec_S_com_to_support()
        vec_S_sacrum_proj = self._calc_vec_S_sacrum_proj()

        if np.linalg.norm(vec_S_com_to_support) < SACRUM_MOVE_THRESHOLD:
            pass
        elif np.dot(vec_S_com_to_support, vec_S_sacrum_proj) > 0:
            self._sacrum_target += SACRUM_MOVE_STEP

        else: 
            self._sacrum_target -= SACRUM_MOVE_STEP
        self._pub_counterweight_pos([self._sacrum_target])
        
    def _baselink_quat_callback(self, msg: Quaternion) -> None:
        self._q_W_baselink = msg

    def _calc_vec_S_com_to_support(self) -> NDArray[np.float64]:
        p_W_biped_com: NDArray[np.float64] = self._calc_p_W_biped_com(self._p_W_joints_com)
        p_S_biped_com: NDArray[np.float64] = np.array([p_W_biped_com[0], p_W_biped_com[1], 0.0], dtype=np.float64)
        p_S_support: NDArray[np.float64] = self._calc_p_S_support()
        vec_S_com_to_support: NDArray[np.float64] = p_S_support - p_S_biped_com
        return vec_S_com_to_support

    def _calc_vec_S_sacrum_proj(self) -> NDArray[np.float64]:
        if self._q_W_baselink is None:
            raise ValueError("Baselink quaternion is not yet received.")
        R_WB = LinearAlgebraUtils.quaternion_to_rotation_matrix(self._q_W_baselink)
        vec_W_yB = R_WB[:, 1]
        vec_S_yB = np.array([vec_W_yB[0], vec_W_yB[1], 0.0], dtype=np.float64)
        return vec_S_yB

def main(args=None):   
    rclpy.init(args=args)
    node = CounterweightControlNode()
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