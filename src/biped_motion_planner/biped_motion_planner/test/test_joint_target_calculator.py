import numpy as np
import pytest
from geometry_msgs.msg import Quaternion, Vector3
from biped_motion_planner.joint_targets_calculator import Config
from biped_motion_planner.joint_targets_calculator import JointTargetsCalculator


@pytest.fixture
def joint_targets_calculator():
    return JointTargetsCalculator()

def test_calc_joint_targets_adds_FOOT_LEN_to_target_z(joint_targets_calculator):
    p_W = {
        "target": Vector3(x=1.5, y=2.5, z=3.5),
        "baselink": Vector3(x=0.5, y=1.5, z=2.5),
        "hip": Vector3(x=0.5, y=2.0, z=3.0)
    }
    q_W_baselink = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0) # no rotation
    joint_targets_calculator.calc_joint_targets(p_W, q_W_baselink)
    assert np.isclose(
        joint_targets_calculator.p_W["foot"].z - joint_targets_calculator.p_W["target"].z, 
        Config.FOOT_LEN)
    
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

def test_transform_points_Baselink_to_Leg(joint_targets_calculator):
    T_LB = np.array([
        [1.0, 0.0, 0.0, 0.5],
        [0.0, 0.948683298050514,  0.316227766016838, 0.447213595499958],
        [0.0, -0.316227766016838, 0.948683298050514, 0.223606797749979],
        [0.0, 0.0, 0.0, 1.0],
    ], dtype=np.float64)

    joint_targets_calculator.p_B = {
        "foot": Vector3(x=-1.0, y=0.0, z=-1.41421356)
    }
    joint_targets_calculator.p_L.update(joint_targets_calculator._transform_points_Baselink_to_Leg(T_LB))

    assert np.allclose(
        [joint_targets_calculator.p_L["foot"].x, joint_targets_calculator.p_L["foot"].y, joint_targets_calculator.p_L["foot"].z],
        [-0.5, 0.0, -1.118033989],
        atol=1e-9
    )

def test_calc_BL_transforms(joint_targets_calculator):
    joint_targets_calculator.p_B = {
        "foot": Vector3(x=-1.0, y=0.0, z=-1.41421356),
        "hip": Vector3(x=-0.5, y=-0.35355339, z=-0.35355339)
    }

    R_BL, T_LB = joint_targets_calculator._calc_BL_transforms()
    R_BL_expected = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 0.948683298050514, -0.316227766016838],
        [0.0, 0.316227766016838,  0.948683298050514],
    ], dtype=np.float64)

    T_LB_expected = np.array([
        [1.0, 0.0, 0.0, 0.5],
        [0.0, 0.948683298050514,  0.316227766016838, 0.447213595499958],
        [0.0, -0.316227766016838, 0.948683298050514, 0.223606797749979],
        [0.0, 0.0, 0.0, 1.0],
    ], dtype=np.float64)

    assert np.allclose(R_BL, R_BL_expected, atol=1e-12)
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

def test_calc_R_BL_norm_zero(joint_targets_calculator):
    joint_targets_calculator.p_B = {
        "foot": Vector3(x=0.0, y=0.0, z=0.0),
        "hip": Vector3(x=5.0, y=0.0, z=0.0)
    }
    with pytest.raises(ValueError, match="zero-length"):
        joint_targets_calculator._calc_R_BL()

def test_transform_points_Leg_to_uw(joint_targets_calculator):
    joint_targets_calculator.p_L = {
        "foot": Vector3(x=-0.5, y=0.0, z=-1.118033989)
    }
    joint_targets_calculator.p_uw.update(joint_targets_calculator._transform_points_Leg_to_uw())

    assert np.allclose(
        [joint_targets_calculator.p_uw["foot"][0], joint_targets_calculator.p_uw["foot"][1]],
        [-0.5, -1.118033989],
        atol=1e-12
    )

def test_calc_phi_BL(joint_targets_calculator):

    # foot at 3rd quadrant
    joint_targets_calculator.p_B = {
        "foot": Vector3(x=5.0, y=-1.73205081, z=-1.0),
        "hip": Vector3(x=0.0, y=0.0, z=-0.0)
    }
    phi_BL = joint_targets_calculator._calc_phi_BL()
    assert np.allclose(phi_BL, -60, atol=1e-12)

    # foot at 4th quadrant
    joint_targets_calculator.p_B = {
        "foot": Vector3(x=5.0, y=1.73205081, z=-1.0),
        "hip": Vector3(x=0.0, y=0.0, z=-0.0)
    }
    phi_BL = joint_targets_calculator._calc_phi_BL()
    assert np.isclose(phi_BL, 60, atol=1e-12)


def test_project_gravity_to_uw_plane(joint_targets_calculator):
    R_BW = np.eye(3, dtype=np.float64)
    R_LB = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 0.948683298050514,  0.316227766016838],
        [0.0, -0.316227766016838, 0.948683298050514],
    ], dtype=np.float64)
    e_L_proj = joint_targets_calculator._project_gravity_to_uw_plane(R_BW, R_LB)
    e_L_proj_expected = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    assert np.allclose(e_L_proj, e_L_proj_expected, atol=1e-12)

    R_BW = np.array([
        [ 0.0, 0.0, 1.0],
        [ 0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0],
    ], dtype=np.float64)

    R_LB = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 0.948683298050514,  0.316227766016838],
        [0.0, -0.316227766016838, 0.948683298050514],
    ], dtype=np.float64)

    e_L_proj = joint_targets_calculator._project_gravity_to_uw_plane(R_BW, R_LB)
    e_L_proj_expected = np.array([-1.0, 0.0, 0.0], dtype=np.float64)
    assert np.allclose(e_L_proj, e_L_proj_expected, atol=1e-12)

def test_project_gravity_to_uw_plane_degenerate_projection_warns(joint_targets_calculator):
    R_BW = np.eye(3, dtype=np.float64)
    R_LB = np.array([[1., 0., 0.],
                     [0., 0., -1.],
                     [0., 1.,  0.]], dtype=np.float64)
    with pytest.warns(RuntimeWarning, match="Gravity is parallel to the uw-plane normal; projection is degenerated. Returning -w_L."):
        e_L_proj = joint_targets_calculator._project_gravity_to_uw_plane(R_BW, R_LB)
    e_L_proj_expected = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    assert np.allclose(e_L_proj, e_L_proj_expected, atol=1e-12)

def test_calc_p_uw_ankle(joint_targets_calculator):
    joint_targets_calculator.p_uw = {
        "foot": np.array([-0.5, -1.118033989], dtype=np.float64)
    }
    e_L_proj = np.array([0.6, 0.0, 0.8], dtype=np.float64)
    p_uw_ankle = joint_targets_calculator._calc_p_uw_ankle(e_L_proj)

    # The vector of ankle to foot
    delta = joint_targets_calculator.p_uw["foot"] - p_uw_ankle
    assert np.isclose(np.linalg.norm(delta), Config.ANKLE_LEN, atol=1e-12)

    e_uw_proj = np.array([e_L_proj[0], e_L_proj[2]], dtype=np.float64)
    e_uw_proj_norm = e_uw_proj / np.linalg.norm(e_uw_proj)
    delta_norm = delta / np.linalg.norm(delta)
    assert np.allclose(delta_norm, e_uw_proj_norm, atol=1e-12)

def test_calc_theta_calf(monkeypatch, joint_targets_calculator):
    # test1: ankle at 4th quadrant
    monkeypatch.setattr(Config, "THIGH_LEN", 1.0)
    monkeypatch.setattr(Config, "CALF_LEN", 1.0)
    joint_targets_calculator.p_uw = {
        "thigh": np.array([0.0, -1.0], dtype=np.float64),
        "ankle": np.array([1.0, -2.2], dtype=np.float64)
    }

    theta_calf, thigh_to_ankle_vec_uw, p_uw_ankle_new, hold_prev_pose = joint_targets_calculator._calc_theta_calf()
    assert np.isclose(theta_calf, -77.291, atol=1e-3)
    assert np.allclose(thigh_to_ankle_vec_uw, np.array([1.0, -1.2]), atol=1e-12)
    assert np.allclose(p_uw_ankle_new, joint_targets_calculator.p_uw["ankle"], atol=1e-12)
    assert hold_prev_pose is False

    # test2: ankle at 3rd quadrant
    monkeypatch.setattr(Config, "THIGH_LEN", 1.0)
    monkeypatch.setattr(Config, "CALF_LEN", 3.0)
    joint_targets_calculator.p_uw = {
        "thigh": np.array([0.0, -1.0], dtype=np.float64),
        "ankle": np.array([0.4639, -4.4691], dtype=np.float64)
    }

    theta_calf, thigh_to_ankle_vec_uw, p_uw_ankle_new, hold_prev_pose = joint_targets_calculator._calc_theta_calf()
    assert np.isclose(theta_calf, -67.976, atol=1e-3)
    assert np.allclose(thigh_to_ankle_vec_uw, np.array([0.4639, -3.4691]), atol=1e-12)
    assert np.allclose(p_uw_ankle_new, joint_targets_calculator.p_uw["ankle"], atol=1e-12)
    assert hold_prev_pose is False

    # test3: theta_calf equals -90
    monkeypatch.setattr(Config, "THIGH_LEN", 1.0)
    monkeypatch.setattr(Config, "CALF_LEN", 1.0)
    joint_targets_calculator.p_uw = {
        "thigh": np.array([0.0, -1.0], dtype=np.float64),
        "ankle": np.array([1.0, -2.0], dtype=np.float64)
    }

    theta_calf, thigh_to_ankle_vec_uw, p_uw_ankle_new, hold_prev_pose = joint_targets_calculator._calc_theta_calf()
    assert np.isclose(theta_calf, -90.0, atol=1e-3)
    assert np.allclose(thigh_to_ankle_vec_uw, np.array([1.0, -1.0]), atol=1e-12)
    assert np.allclose(p_uw_ankle_new, joint_targets_calculator.p_uw["ankle"], atol=1e-12)
    assert hold_prev_pose is False

def test_calc_theta_calf_ankle_too_far(monkeypatch, joint_targets_calculator):
    monkeypatch.setattr(Config, "THIGH_LEN", 1.0)
    monkeypatch.setattr(Config, "CALF_LEN", 1.0)
    joint_targets_calculator.p_uw = {
        "thigh": np.array([0.0, -1.0], dtype=np.float64),
        "ankle": np.array([5.0, -6.0], dtype=np.float64)
    }
    p_uw_ankle_new_expect = np.array([2/np.sqrt(2), -1-(2/np.sqrt(2))])
    with pytest.warns(RuntimeWarning, match="Ankle is too far from hip"):
        theta_calf, thigh_to_ankle_vec_uw, p_uw_ankle_new, hold_prev_pose = joint_targets_calculator._calc_theta_calf()
    assert np.isclose(theta_calf, 0.0, atol=1e-12)
    assert np.allclose(thigh_to_ankle_vec_uw, np.array([2/np.sqrt(2), -(2/np.sqrt(2))]), atol=1e-12)
    assert np.allclose(p_uw_ankle_new, p_uw_ankle_new_expect, atol=1e-12)
    assert hold_prev_pose is False

def test_calc_theta_calf_ankle_too_close(monkeypatch, joint_targets_calculator):
    # Hold the previous pose if the ankle is too close to the thigh and unreachable.
    monkeypatch.setattr(Config, "THIGH_LEN", 1.0)
    monkeypatch.setattr(Config, "CALF_LEN", 3.0)
    joint_targets_calculator.p_uw = {
        "thigh": np.array([0.0, -1.0], dtype=np.float64),
        "ankle": np.array([0.0, -2.0], dtype=np.float64)
    }

    with pytest.warns(RuntimeWarning, match="Ankle is too close to hip"):
        theta_calf, thigh_to_ankle_vec_uw, p_uw_ankle_new, hold_prev_pose = joint_targets_calculator._calc_theta_calf()
    assert hold_prev_pose is True
    assert thigh_to_ankle_vec_uw is None
    assert theta_calf is None
    assert p_uw_ankle_new is None

def test_calc_theta_calf_exceed(monkeypatch, joint_targets_calculator):
    # Hold the previous pose if the calf joint angle (theta_calf) exceeds ±90°.
    monkeypatch.setattr(Config, "THIGH_LEN", 1.0)
    monkeypatch.setattr(Config, "CALF_LEN", 1.0)
    joint_targets_calculator.p_uw = {
        "thigh": np.array([0.0, -1.0], dtype=np.float64),
        "ankle": np.array([0.0, -2.0], dtype=np.float64)
    }

    with pytest.warns(RuntimeWarning, match="exceeds"):
        theta_calf, thigh_to_ankle_vec_uw, p_uw_ankle_new, hold_prev_pose = joint_targets_calculator._calc_theta_calf()
    assert hold_prev_pose is True
    assert thigh_to_ankle_vec_uw is None
    assert theta_calf is None
    assert p_uw_ankle_new is None