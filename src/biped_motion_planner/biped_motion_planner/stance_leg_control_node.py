import rclpy
import sys
from geometry_msgs.msg import Quaternion, Vector3
from rclpy.node import Node
from std_msgs.msg import Float32, Float64MultiArray, String
from typing import Optional
from .config import Config, LegSide

JOINT_NUMS:int = 5 # exclude back, sacrum


class StanceLegControlNode(Node):
    def __init__(self):
        super().__init__('stance_leg_control_node')
        self._leg_side: LegSide = "undefined"

        self._stance_side_subscriber_ = self.create_subscription(
            String,
            '/biped/stance_side',
            self._stance_side_callback,
            10
        )
        self._stance_leg_target_subscriber_ = self.create_subscription(
            Float32,
            '/biped/stance_leg_target',
            self._stance_leg_target_callback,
            10
        )

        self._left_joint_target_publisher_ = self.create_publisher(
            Float64MultiArray,
            '/biped/left_joint_target',
            10
        )
        self._right_joint_target_publisher_ = self.create_publisher(
            Float64MultiArray,
            '/biped/right_joint_target',
            10
        )
    
    def _stance_leg_target_callback(self, msg: Vector3) -> None:
        joint_pose = self._compose_joint_pose_for_publish(joint_targets)
        self._pub_joint_pos(joint_pose)

    def _stance_side_callback(self, msg: String) -> None:
        if msg.data not in ("left", "right"):
            self.get_logger().error(f"Invalid stance side: {msg.data}. Must be 'left' or 'right'.")
            return
        if msg.data != self._leg_side:
            self.get_logger().info(f"Switching stance side from {self._leg_side} to {msg.data}.")
            self._leg_side = msg.data

    def _pub_joint_pos(self, joint_pos: list[float]) -> None:
        msg = Float64MultiArray()
        msg.data = [float(i) for i in joint_pos]
        if self._leg_side == "left":
            self._left_joint_target_publisher_.publish(msg)
        elif self._leg_side == "right":
            self._right_joint_target_publisher_.publish(msg)
        else:
            self.get_logger().error(f"Invalid leg side: {self._leg_side}. Cannot publish joint targets.")

    def _compose_joint_pose_for_publish(self, joint_targets) -> list[float]:
        joint_pose: list[float] = [
            joint_targets['hip'],
            joint_targets['thigh'],
            joint_targets['calf'],
            joint_targets['ankle'],
            joint_targets['foot']
        ]
        if len(joint_pose) != JOINT_NUMS:
            raise ValueError("Invalid swing leg joint pose length.")
        return joint_pose
    

def main(args=None):
    rclpy.init(args=args)
    node = StanceLegControlNode()
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