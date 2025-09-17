import rclpy
import sys
from rclpy.node import Node
from std_msgs.msg import Float32, Float64MultiArray, String
from .config import LegSide

STANCE_LEG_JOINT_ALPHA: float = 5.0 # in degrees
VALID_LEG_SIDES: tuple[str, ...] = ("left", "right")
VALID_SUPPORT_SIDES: tuple[str, ...] = ("left", "right", "mid")

class BipedMotionPlannerNode(Node):
    def __init__(self):
        super().__init__('biped_motion_planner_node')
        self._swing_leg_side: LegSide = "undefined"
        self._stance_leg_side: LegSide = "undefined"
        self._support_side: str = "undefined"




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