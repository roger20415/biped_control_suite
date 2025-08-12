import curses
import rclpy
import threading
import time
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray

JOINT_NUMS:int = 12


class BipedKeyboardNode(Node):
    def __init__(self, stdscr):
        super().__init__('biped_keyboard_node')

        self._joint_target_publisher_ = self.create_publisher(
            Float64MultiArray,
            '/biped/joint_target',
            10
        )

        self._joint_pos: list[float] = [0.0]*JOINT_NUMS
        self._key_in_count:int = 0

        self._stdscr = stdscr
        self._stdscr.keypad(False)

    def process_key_in(self):
        while rclpy.ok():
            input_key:int = self._stdscr.getch()
            if input_key == curses.ERR:
                self._print_basic_info(ord(' '))
                time.sleep(0.1)
                continue

            self._key_in_count += 1
            self._print_basic_info(input_key)

            if input_key == ord('w'):
                self._handle_key_w()
            elif input_key == ord('q'): # Exit
                break

    def _print_basic_info(self, key):
        self._stdscr.clear()
        self._stdscr.move(0, 0)
        self._stdscr.addstr(f"{self._key_in_count:5d} Key '{chr(key)}' pressed!")

    def _handle_key_w(self):
        self._joint_pos = [1.0]*12
        self._pub_joint_pos()

    def _pub_joint_pos(self):
        msg = Float64MultiArray()
        msg.data = [float(i) for i in self._joint_pos]
        self._joint_target_publisher_.publish(msg)
    
def main(args=None):
    stdscr = curses.initscr()
    curses.noecho()
    curses.raw()

    rclpy.init(args=args)
    biped_keyboard_node = BipedKeyboardNode(stdscr)
    spin_thread = threading.Thread(
        target=rclpy.spin, 
        args=(biped_keyboard_node,), 
        daemon=True)
    spin_thread.start()

    try:
        biped_keyboard_node.process_key_in()
    finally:
        biped_keyboard_node.get_logger().info(f'Quit')
        curses.endwin()
        rclpy.shutdown()
        spin_thread.join()

if __name__ == '__main__':
    main()