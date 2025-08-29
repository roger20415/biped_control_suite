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

def test_invert_transformation_matrix(linear_algebra_utils):
    T_1 = np.array([
        [0.474747474747475, -0.464646464646465, -0.747474747474747, 1.0],
        [-0.141414141414141, 0.797979797979798, -0.585858585858586, 2.0],
        [0.868686868686869, 0.383838383838384, 0.313131313131313, 3.0],
        [0.0, 0.0, 0.0, 1.0]
    ], dtype=np.float64)
    T_expected_1 = np.array([
        [ 0.474747474748, -0.141414141414,  0.868686868687, -2.79797979798 ],
        [-0.464646464646,  0.797979797980,  0.383838383838, -2.28282828283 ],
        [-0.747474747475, -0.585858585859,  0.313131313131,  0.97979797998 ],
        [ 0.0,             0.0,             0.0,             1.0           ]
    ], dtype=np.float64)
    T_computed_1 = linear_algebra_utils._invert_transformation_matrix(T_1)

    T_2 = np.array([
        [ 0.353553, -0.612372,  0.707107,  1.5 ],
        [ 0.926777,  0.126826, -0.353553, -2.0 ],
        [ 0.126826,  0.780330,  0.612372,  0.5 ],
        [ 0.0,       0.0,       0.0,       1.0 ]
    ], dtype=np.float64)
    T_expected_2 = np.array([
        [ 0.353553,  0.926777,  0.126826,  1.259812 ],
        [-0.612372,  0.126826,  0.780330,  0.782045 ],
        [ 0.707107, -0.353553,  0.612372, -2.073953 ],
        [ 0.0,       0.0,       0.0,       1.0      ]
    ], dtype=np.float64)
    T_computed_2 = linear_algebra_utils._invert_transformation_matrix(T_2)

    assert np.allclose(T_computed_1, T_expected_1, atol=1e-12)
    assert np.allclose(T_computed_2, T_expected_2, atol=1e-12)