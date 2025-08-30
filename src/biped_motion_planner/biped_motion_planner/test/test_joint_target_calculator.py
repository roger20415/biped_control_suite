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
    
def test_transform_points_world_to_baselink(joint_targets_calculator):
    q_W_baselink = Quaternion(x=0.6532814824381882, 
                              y=0.27059805007309845, 
                              z=-0.6532814824381883, 
                              w=0.27059805007309856)
    
    joint_targets_calculator.p_W = {
        "foot": Vector3(x=1.5, y=2.5, z=3.5),
        "baselink": Vector3(x=0.5, y=1.5, z=2.5),
        "hip": Vector3(x=0.5, y=2.0, z=3.0)
    }

    joint_targets_calculator._transform_points_world_to_baselink(q_W_baselink)
    assert np.allclose(
        [joint_targets_calculator.p_B["hip"].x, joint_targets_calculator.p_B["hip"].y, joint_targets_calculator.p_B["hip"].z],
        [-0.5, -0.35355339, -0.35355339],
        atol=1e-12
    )
    assert np.allclose(
        [joint_targets_calculator.p_B["foot"].x, joint_targets_calculator.p_B["foot"].y, joint_targets_calculator.p_B["foot"].z],
        [-1.0, 0.0, -1.41421356],
        atol=1e-12
    )
