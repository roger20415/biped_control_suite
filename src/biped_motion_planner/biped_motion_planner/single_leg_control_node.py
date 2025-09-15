import rclpy
import sys
from geometry_msgs.msg import Quaternion, Vector3
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from std_msgs.msg import Float64MultiArray, String
from typing import Optional
from .config import Config, LegSide
from .joint_targets_calculator import JointTargetsCalculator

JOINT_NUMS:int = 10 # exclude back, sacrum
FALL_DOWN_Z_THRESHOLD: float = 0.011 # in meters
STEP_SIZE: float = 0.0001 # in meters
REQUIRED_P_W_KEYS: tuple[str] = ("baselink", "hip", "foot", "target")
REQUIRED_P_W_RAW_KEYS: tuple[str] = ("l_hip", "l_foot", "r_hip", "r_foot")

class SingleLegControlNode(Node):
    def __init__(self):
        super().__init__('single_leg_control_node')
        self._leg_side: LegSide = "left"
        self.joint_targets_calculator = JointTargetsCalculator()
        self._q_W_baselink: Optional[Quaternion] = None
        self._p_W_raw: dict[str, Optional[Vector3]] = {k: None for k in REQUIRED_P_W_RAW_KEYS}
        self._p_W: dict[str, Optional[Vector3]] = {k: None for k in REQUIRED_P_W_KEYS}
        self._init_p_W_target()

        qos_sensor = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        self._teleop_key_subscriber_ = self.create_subscription(
            String,
            '/biped/teleop_key',
            self._teleop_key_callback,
            10
        )
        self._baselink_quat_subscriber_ = self.create_subscription(
            Quaternion,
            '/baselink/quat',
            self._baselink_quat_callback,
            qos_sensor
        )
        self._baselink_translate_subscriber_ = self.create_subscription(
            Vector3,
            '/baselink/translate',
            self._baselink_translate_callback,
            qos_sensor
        )
        self._l_hip_translate_subscriber_ = self.create_subscription(
            Vector3,
            '/l_hip/translate',
            self._l_hip_translate_callback,
            qos_sensor
        )
        self._r_hip_translate_subscriber_ = self.create_subscription(
            Vector3,
            '/r_hip/translate',
            self._r_hip_translate_callback,
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
        self._joint_target_publisher_ = self.create_publisher(
            Float64MultiArray,
            '/biped/joint_target',
            10
        )
    
    def _teleop_key_callback(self, msg: String) -> None:
        key = msg.data
        if key not in ['w', 'a', 's', 'd', 'e', 'r']:
            self.get_logger().warn(f"Received invalid teleop key: {key}")
            return
        self._compose_leg_p_W()
        self._p_W["target"] = self._calc_p_W_target(key)
        ready, missing = self._check_state_ready()
        if not ready:
            self.get_logger().warn(f"State not ready, missing: {', '.join(missing)}")
            return
        # TODO: check if state is fresh enough
        self.get_logger().info(f"p_W_target: {self._p_W['target']}")
        hold_prev_pose, joint_targets = self.joint_targets_calculator.calc_joint_targets(self._p_W, self._q_W_baselink, self._leg_side)
        if hold_prev_pose:
            self.get_logger().info("Holding previous pose.")
        else:
            joint_pose = self._compose_joint_pose_for_publish(joint_targets)
            self._pub_joint_pos(joint_pose)
    
    def _baselink_quat_callback(self, msg: Quaternion) -> None:
        self._q_W_baselink = msg
    def _baselink_translate_callback(self, msg: Vector3) -> None:
        self._p_W["baselink"] = msg
        if msg.z < FALL_DOWN_Z_THRESHOLD:
            self.get_logger().warn("Robot has fallen down! Clear and init p_W_target")
            self._init_p_W_target()
    def _l_hip_translate_callback(self, msg: Vector3) -> None:
        self._p_W_raw["l_hip"] = msg
    def _r_hip_translate_callback(self, msg: Vector3) -> None:
        self._p_W_raw["r_hip"] = msg
    def _l_foot_translate_callback(self, msg: Vector3) -> None:
        self._p_W_raw["l_foot"] = msg
    def _r_foot_translate_callback(self, msg: Vector3) -> None:
        self._p_W_raw["r_foot"] = msg
    
    def _pub_joint_pos(self, joint_pos: list[float]) -> None:
        msg = Float64MultiArray()
        msg.data = [float(i) for i in joint_pos]
        self._joint_target_publisher_.publish(msg)

    def _compose_joint_pose_for_publish(self, joint_targets) -> list[float]:
        joint_pose: list[float] = [0.0]*JOINT_NUMS
        if self._leg_side == "left":
            joint_pose[0] = joint_targets['hip'] # l_hip
            joint_pose[2] = joint_targets['thigh'] # l_thigh
            joint_pose[4] = joint_targets['calf'] # l_calf
            joint_pose[6] = joint_targets['ankle'] # l_ankle
            joint_pose[8] = joint_targets['foot'] # l_foot
        elif self._leg_side == "right":
            joint_pose[1] = joint_targets['hip'] # r_hip
            joint_pose[3] = joint_targets['thigh'] # r_thigh
            joint_pose[5] = joint_targets['calf'] # r_calf
            joint_pose[7] = joint_targets['ankle'] # r_ankle
            joint_pose[9] = joint_targets['foot'] # r_foot
        return joint_pose
    
    def _check_state_ready(self) -> tuple[bool, list[str]]:
        missing: list[str] = []
        if self._q_W_baselink is None:
            missing.append("q_W_baselink")
        for k in REQUIRED_P_W_KEYS:
            if self._p_W.get(k) is None:
                missing.append(f"p_W[{k}]")
        return (len(missing) == 0, missing)
    
    def _calc_p_W_target(self, key: str) -> Optional[Vector3]:
        current_target = Vector3(
            x=self._p_W["foot"].x,
            y=self._p_W["foot"].y,
            z=self._p_W["foot"].z-Config.FOOT_LEN
        )
        if current_target is None:
            self.get_logger().error("Current target position is None.")
            return None
        new_target = Vector3(
            x=current_target.x,
            y=current_target.y,
            z=current_target.z
        )
        if key == 'w':
            new_target.z=current_target.z+STEP_SIZE
        elif key == 's':
            new_target.z=current_target.z-STEP_SIZE
        elif key == 'a':
            new_target.x=current_target.x+STEP_SIZE
        elif key == 'd':
            new_target.x=current_target.x-STEP_SIZE
        elif key == 'e':
            new_target.y=current_target.y+STEP_SIZE
        elif key == 'r':
            new_target.y=current_target.y-STEP_SIZE
        return new_target
    
    def _compose_leg_p_W(self) -> None:
        if any(self._p_W_raw.get(k) is None for k in REQUIRED_P_W_RAW_KEYS):
            self.get_logger().error("Raw leg joint positions are incomplete.")
            return
        if self._leg_side == "left":
            self._p_W["hip"] = self._p_W_raw["l_hip"]
            self._p_W["foot"] = self._p_W_raw["l_foot"]
        elif self._leg_side == "right":
            self._p_W["hip"] = self._p_W_raw["r_hip"]
            self._p_W["foot"] = self._p_W_raw["r_foot"]
    
    def _init_p_W_target(self) -> None:
        if self._leg_side == "left":
            self._p_W["target"] = Config.ORIGIN_L_TARGET
        elif self._leg_side == "right":
            self._p_W["target"] = Config.ORIGIN_R_TARGET
        else:
            self.get_logger().error("Leg side is undefined. Exiting.")
            rclpy.shutdown()
            sys.exit(1)
    
def main(args=None):
    rclpy.init(args=args)
    node = SingleLegControlNode()
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