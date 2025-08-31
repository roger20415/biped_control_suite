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
    
def test_transform_points_World_to_Baselink(joint_targets_calculator):
    T_BW = np.array([
        [ 0.0,                0.0,               -1.0,                2.5],
        [ 0.7071067811865476, -0.7071067811865476, 0.0,               0.7071067811865476],
        [-0.7071067811865476, -0.7071067811865476, 0.0,               1.4142135623730951],
        [ 0.0,                0.0,                0.0,                1.0],
    ], dtype=np.float64)
    
    joint_targets_calculator.p_W = {
        "foot": Vector3(x=1.5, y=2.5, z=3.5),
        "baselink": Vector3(x=0.5, y=1.5, z=2.5),
        "hip": Vector3(x=0.5, y=2.0, z=3.0)
    }

    joint_targets_calculator.p_B.update(joint_targets_calculator._transform_points_World_to_Baselink(T_BW))
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

def test_calc_WB_transform(joint_targets_calculator):
    q_W_baselink = Quaternion(x=0.6532814824381882, 
                            y=0.27059805007309845, 
                            z=-0.6532814824381883, 
                            w=0.27059805007309856)
    p_W_baselink = Vector3(x=0.5, y=1.5, z=2.5)
    R_WB, T_BW = joint_targets_calculator._calc_WB_transforms(q_W_baselink, p_W_baselink)
    R_WB_expected = np.array([
        [ 0.0,  0.7071067811865476, -0.7071067811865476],
        [ 0.0, -0.7071067811865476, -0.7071067811865476],
        [-1.0,  0.0,                0.0],
    ], dtype=np.float64)
    assert np.allclose(R_WB, R_WB_expected, atol=1e-12)

    T_BW_expected = np.array([
        [ 0.0,                0.0,               -1.0,                2.5],
        [ 0.7071067811865476, -0.7071067811865476, 0.0,               0.7071067811865476],
        [-0.7071067811865476, -0.7071067811865476, 0.0,               1.4142135623730951],
        [ 0.0,                0.0,                0.0,                1.0],
    ], dtype=np.float64)
    assert np.allclose(T_BW, T_BW_expected, atol=1e-12)

def test_calc_BL_transforms(joint_targets_calculator):
    joint_targets_calculator.p_B = {
        "foot": Vector3(x=-1.0, y=0.0, z=-1.41421356),
        "hip": Vector3(x=-0.5, y=-0.35355339, z=-0.35355339)
    }

    R_BL, R_LB, T_LB = joint_targets_calculator._calc_BL_transforms()
    R_BL_expected = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 0.948683298050514, -0.316227766016838],
        [0.0, 0.316227766016838,  0.948683298050514],
    ], dtype=np.float64)

    R_LB_expected = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 0.948683298050514,  0.316227766016838],
        [0.0, -0.316227766016838, 0.948683298050514],
    ], dtype=np.float64)

    T_LB_expected = np.array([
        [1.0, 0.0, 0.0, 0.5],
        [0.0, 0.948683298050514,  0.316227766016838, 0.447213595499958],
        [0.0, -0.316227766016838, 0.948683298050514, 0.223606797749979],
        [0.0, 0.0, 0.0, 1.0],
    ], dtype=np.float64)

    assert np.allclose(R_BL, R_BL_expected, atol=1e-12)
    assert np.allclose(R_LB, R_LB_expected, atol=1e-12)
    assert np.allclose(T_LB, T_LB_expected, atol=1e-12)


def test_calc_R_BL(joint_targets_calculator):
    joint_targets_calculator.p_B = {
        "foot": Vector3(x=-1.0, y=0.0, z=-1.41421356),
        "hip": Vector3(x=-0.5, y=-0.35355339, z=-0.35355339)
    }
    R_BL = joint_targets_calculator._calc_R_BL()
    R_BL_expected = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 0.948683298050514, -0.316227766016838],
        [0.0, 0.316227766016838,  0.948683298050514],
    ], dtype=np.float64)
    assert np.allclose(R_BL, R_BL_expected, atol=1e-12)