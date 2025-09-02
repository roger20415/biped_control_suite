import numpy as np
import warnings

class TrigonometricUtils():
    @staticmethod
    def clamp_cos(c: float, tol: float = 1e-9) -> float:
        if c > 1.0 + tol or c < -1.0 - tol:
            raise ValueError("cosine out of [-1, 1] by more than tol")
        if c > 1.0 or c < -1.0:
            warnings.warn("cosine slightly outside [-1,1]; clamped", RuntimeWarning)
            c = np.clip(c, -1.0, 1.0)
        return c