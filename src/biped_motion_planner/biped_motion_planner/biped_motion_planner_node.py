from __future__ import annotations

import sys

import numpy as np
import rclpy

from dataclasses import dataclass
from enum import Enum, auto
from numpy.typing import NDArray
from typing import Callable, Dict, Optional

from geometry_msgs.msg import Quaternion, Vector3
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Float32, Float64MultiArray, String

from .config import Config, LegSide
from .linear_algebra_utils import LinearAlgebraUtils

import rclpy
from rclpy.node import Node

TIMER_PERIOD: float = 0.05 # in seconds
INIT_TO_SS_DURATION: float = 5.0 # in seconds
SS_TO_DS_DURATION: float = 5.0 # in seconds
DS_TO_SS_DURATION: float = 5.0 # in seconds


class Phase(Enum):
    INIT_TO_SS = auto()
    SS_TO_DS   = auto()
    DS_TO_SS   = auto()


@dataclass
class PhaseHandlers:
    on_enter: Callable[['BipedMotionPlannerNode'], None]
    on_step:  Callable[['BipedMotionPlannerNode'], Optional[Phase]]


class BipedMotionPlannerNode(Node):
    def __init__(self):
        super().__init__('biped_motion_planner')
        self._timer = self.create_timer(TIMER_PERIOD, self._on_timer)
        self._prev_time = self.get_clock().now()

        self._current_phase: Phase = Phase.INIT_TO_SS
        self._total_step_idx: int = 0
        self._phase_step_idx: int = 0
        self._phase_duration_time: float = 0.0
        self._phase_time_budget: Dict[Phase, float] =  {
            Phase.INIT_TO_SS: INIT_TO_SS_DURATION,
            Phase.SS_TO_DS: SS_TO_DS_DURATION,
            Phase.DS_TO_SS: DS_TO_SS_DURATION,
        }

        self._handlers: Dict[Phase, PhaseHandlers] = {
            Phase.INIT_TO_SS: PhaseHandlers(
                on_enter=self._enter_init_to_ss,
                on_step=self._step_init_to_ss
            ),
            Phase.SS_TO_DS: PhaseHandlers(
                on_enter=self._enter_ss_to_ds,
                on_step=self._step_ss_to_ds
            ),
            Phase.DS_TO_SS: PhaseHandlers(
                on_enter=self._enter_ds_to_ss,
                on_step=self._step_ds_to_ss
            ),
        }

        self._handlers[self._current_phase].on_enter(self)

    def _on_timer(self) -> None:
        next_phase = self._handlers[self._current_phase].on_step(self)
        self._total_step_idx += 1
        self._phase_step_idx += 1
        if next_phase is not None or self._phase_budget_reached():
            self._transition_to(next_phase or self._default_next_phase())

    def _phase_budget_reached(self) -> bool:
        budget = self._phase_time_budget.get(self._current_phase, None)
        return budget is not None and self._phase_duration_time >= budget

    def _default_next_phase(self) -> Phase:
        if self._current_phase == Phase.INIT_TO_SS:
            return Phase.SS_TO_DS
        if self._current_phase == Phase.SS_TO_DS:
            return Phase.DS_TO_SS
        if self._current_phase == Phase.DS_TO_SS:
            return Phase.SS_TO_DS

    def _transition_to(self, next_phase: Phase) -> None:
        if next_phase == self._current_phase:
            return
        self.get_logger().info(
            f'Transition: {self._current_phase.name} -> {next_phase.name} '
            f'(total_step={self._total_step_idx})'
        )
        self._current_phase = next_phase
        self._phase_step_idx = 0
        self._handlers[self._current_phase].on_enter(self)

    def _enter_init_to_ss(self) -> None:
        self.get_logger().info('[ENTER] INIT_TO_SS')

    def _step_init_to_ss(self) -> Optional[Phase]:
        return None

    def _enter_ss_to_ds(self) -> None:
        self.get_logger().info('[ENTER] SS_TO_DS')

    def _step_ss_to_ds(self) -> Optional[Phase]:
        return None

    def _enter_ds_to_ss(self) -> None:
        self.get_logger().info('[ENTER] DS_TO_SS')

    def _step_ds_to_ss(self) -> Optional[Phase]:
        return None

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

if __name__ == '__main__':
    main()
