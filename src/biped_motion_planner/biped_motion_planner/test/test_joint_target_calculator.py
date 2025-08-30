import numpy as np
import pytest
from geometry_msgs.msg import Quaternion, Vector3
from biped_motion_planner.joint_targets_calculator import Config
from biped_motion_planner.joint_targets_calculator import JointTargetsCalculator


@pytest.fixture
def joint_targets_calculator():
    return JointTargetsCalculator()

def test_calc_joint_targets_adds_L_FOOT_to_target_z(joint_targets_calculator):
    p_W = {
        "target": Vector3(x=1.5, y=2.5, z=3.5),
        "baselink": Vector3(x=0.5, y=1.5, z=2.5),
        "hip": Vector3(x=0.5, y=2.0, z=3.0)
    }
    q_W_baselink = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0) # no rotation
    joint_targets_calculator.calc_joint_targets(p_W, q_W_baselink)
    assert np.isclose(
        joint_targets_calculator.p_W["foot"].z - joint_targets_calculator.p_W["target"].z, 
        Config.L_FOOT)