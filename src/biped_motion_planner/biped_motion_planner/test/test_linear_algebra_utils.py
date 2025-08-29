from geometry_msgs.msg import Quaternion, Vector3
import numpy as np
import pytest
from biped_motion_planner.linear_algebra_utils import LinearAlgebraUtils

@pytest.fixture
def linear_algebra_utils():
    return LinearAlgebraUtils()

def test_quaternion_to_rotation_matrix(linear_algebra_utils):
    q_1 = Quaternion(
        x=0.3,
        y=-0.5,
        z=0.1,
        w=0.8
    )
    R_expected_1 = np.array([
        [0.474747474747475, -0.464646464646465, -0.747474747474747],
        [-0.141414141414141, 0.797979797979798, -0.585858585858586],
        [0.868686868686869, 0.383838383838384, 0.313131313131313]
    ], dtype=np.float64)
    R_computed_1 = linear_algebra_utils.quaternion_to_rotation_matrix(q_1)

    q_2 = Quaternion(
        x=1.0,
        y=0.0,
        z=0.0,
        w=1.0
    )
    R_expected_2 = np.array([
        [1.0,  0.0,  0.0],
        [0.0,  0.0, -1.0],
        [0.0,  1.0,  0.0]
    ], dtype=np.float64)
    R_computed_2 = linear_algebra_utils.quaternion_to_rotation_matrix(q_2)

    assert np.allclose(R_computed_1, R_expected_1, atol=1e-12)    
    assert np.allclose(R_computed_2, R_expected_2, atol=1e-12)

def test_combine_transformation_matrix(linear_algebra_utils):
    R = np.array([
        [0.474747474747475, -0.464646464646465, -0.747474747474747],
        [-0.141414141414141, 0.797979797979798, -0.585858585858586],
        [0.868686868686869, 0.383838383838384, 0.313131313131313]
    ], dtype=np.float64)
    t = Vector3(x=1.0, y=2.0, z=3.0)
    T_expected = np.array([
        [0.474747474747475, -0.464646464646465, -0.747474747474747, 1.0],
        [-0.141414141414141, 0.797979797979798, -0.585858585858586, 2.0],
        [0.868686868686869, 0.383838383838384, 0.313131313131313, 3.0],
        [0.0, 0.0, 0.0, 1.0]
    ], dtype=np.float64)
    T_computed = linear_algebra_utils.combine_transformation_matrix(R, t)
    assert np.allclose(T_computed, T_expected, atol=1e-12)

def test_normalize_vec(linear_algebra_utils):
    vec = [3.0, 4.0, 0.0]
    vec_expected = np.array([0.6, 0.8, 0.0], dtype=np.float64)
    vec_computed = linear_algebra_utils.normalize_vec(vec)
    assert np.allclose(vec_computed, vec_expected, atol=1e-12)