class Config:
    # foot length (in meters)
    FOOT_LEN: float = 1.0
    ANKLE_LEN: float = 1.0
    HIP_LEN: float = 1.0
    THIGH_LEN: float = 1.0
    CALF_LEN: float = 1.0

    # joint angle limits (in degrees)
    L_HIP_MAX_DEG: float = 80.0
    L_HIP_MIN_DEG: float = -50.0
    R_HIP_MAX_DEG: float = 50.0
    R_HIP_MIN_DEG: float = -80.0
    THIGH_MAX_DEG: float = 90.0
    THIGH_MIN_DEG: float = -90.0
    CALF_MAX_DEG: float = 90.0
    CALF_MIN_DEG: float = -90.0
    ANKLE_MAX_DEG: float = 90.0
    ANKLE_MIN_DEG: float = -90.0
    FOOT_MAX_DEG: float = 90.0
    FOOT_MIN_DEG: float = -90.0

    # default joint angles (in degrees)
    HIP_THETA_UW: float = 270.0