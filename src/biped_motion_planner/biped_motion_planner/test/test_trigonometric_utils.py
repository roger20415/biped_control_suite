import numpy as np
import pytest
from biped_motion_planner.trigonometric_utils import TrigonometricUtils

@pytest.fixture
def trigonometric_utils():
    return TrigonometricUtils()

def test_clamp_cos(trigonometric_utils):
    assert np.isclose(trigonometric_utils.clamp_cos(0.5), 0.5)
    assert np.isclose(trigonometric_utils.clamp_cos(1.0), 1.0)
    assert np.isclose(trigonometric_utils.clamp_cos(-1.0), -1.0)

def test_clamp_cos_slightly_outside(trigonometric_utils):
    with pytest.warns(RuntimeWarning, match="clamped"):
        c = trigonometric_utils.clamp_cos(1.0 + 1e-10)
        assert np.isclose(c, 1.0)
    
    with pytest.warns(RuntimeWarning, match="clamped"):
        c = trigonometric_utils.clamp_cos(-1.0 - 1e-10)
        assert np.isclose(c, -1.0)

def test_clamp_cos_outside(trigonometric_utils):
    with pytest.raises(ValueError, match="cosine out of"):
        trigonometric_utils.clamp_cos(1.0 + 1e-8)
    with pytest.raises(ValueError, match="cosine out of"):
        trigonometric_utils.clamp_cos(-1.0 - 1e-8)

def test_normalize_angle_to_180(trigonometric_utils):
    assert np.isclose(trigonometric_utils.normalize_angle_to_180(0), 0)
    assert np.isclose(trigonometric_utils.normalize_angle_to_180(180), -180)
    assert np.isclose(trigonometric_utils.normalize_angle_to_180(-180), -180)
    assert np.isclose(trigonometric_utils.normalize_angle_to_180(190), -170)
    assert np.isclose(trigonometric_utils.normalize_angle_to_180(-700), 20)