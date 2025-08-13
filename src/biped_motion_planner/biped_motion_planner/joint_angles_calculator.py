from collections import namedtuple

Point3D = namedtuple("Point3D", ["x", "y", "z"])
Point2D = namedtuple("Point2D", ["u", "v"])

L_FOOT: float = 1.0

class JointAnglesCalculator():
    def calculate_joint_angles(self, p_target_xyz: Point3D, p_hip_xyz: Point3D, phi_base: float) -> tuple[float]:
        p_foot_xyz = Point3D(p_target_xyz.x, p_target_xyz.y, L_FOOT)
        pass
        
        
        