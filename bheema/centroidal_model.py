import numpy as np
from scipy.linalg import expm

# ============================================================
# Utility
# ============================================================

def skew(v: np.ndarray) -> np.ndarray:
    """
    Returns skew-symmetric matrix such that:
        skew(v) @ f = v x f
    """
    return np.array([
        [0.0,     -v[2],  v[1]],
        [v[2],     0.0,  -v[0]],
        [-v[1],    v[0],  0.0]
    ])

# ============================================================
# Centroidal Continuous Model
# ============================================================

class CentroidalDynamics:
    """
    Humanoid centroidal model

    State:
        x = [ p_com(3),
              rpy(3),
              v_com(3),
              omega_world(3) ]   (12x1)

    Input:
        u = [ f_left(3),
              f_right(3) ]      (6x1)
    """

    def __init__(self, mass: float, inertia_body: np.ndarray):
        self.m = float(mass)
        # Store the body-frame inertia (Constant)
        self.I_body = inertia_body.reshape(3, 3)
        self.I_inv_body = np.linalg.inv(self.I_body)

        self.g_vec = np.array([0.0, 0.0, -9.81])

    # ------------------------------------------------------------
    # Continuous-time dynamics
    # ------------------------------------------------------------
    def continuous_matrices(self, r_left: np.ndarray, r_right: np.ndarray, yaw: float):
        """
        Build continuous model:
            x_dot = A x + B u + g
        """

        I3 = np.eye(3)
        Z3 = np.zeros((3, 3))

        # --- Rotate Inertia to World Frame ---
        cy = np.cos(yaw)
        sy = np.sin(yaw)
        R_yaw = np.array([
            [ cy, -sy,  0.0],
            [ sy,  cy,  0.0],
            [0.0,  0.0,  1.0]
        ])

        # I_world_inv = R_z * I_body_inv * R_z^T
        I_inv_world = R_yaw @ self.I_inv_body @ R_yaw.T

        # --- A Matrix ---
        A = np.block([
            [Z3, Z3, I3, Z3],  # p_dot = v
            [Z3, Z3, Z3, I3],  # rpy_dot ≈ omega_world (Valid small-angle approx for world frame)
            [Z3, Z3, Z3, Z3],  # v_dot independent of x
            [Z3, Z3, Z3, Z3],  # omega_dot independent of x
        ])

        # --- B Matrix (Using rotated Inertia) ---
        B_left = np.vstack([
            Z3,
            Z3,
            (1.0 / self.m) * I3,
            I_inv_world @ skew(r_left)
        ])

        B_right = np.vstack([
            Z3,
            Z3,
            (1.0 / self.m) * I3,
            I_inv_world @ skew(r_right)
        ])

        B = np.hstack([B_left, B_right])

        # --- Gravity ---
        g = np.zeros((12, 1))
        g[6:9, 0] = self.g_vec

        return A, B, g


# ============================================================
# Discrete Model (ZOH)
# ============================================================
class CentroidalDiscrete:
    """
    Exact Analytical ZOH discretization (Lightning Fast & 100% Accurate).
    """
    def __init__(self, dt: float):
        self.dt = float(dt)

    def discretize(self, A: np.ndarray, B: np.ndarray, g: np.ndarray):
        nx = A.shape[0]
        
        # Exact Ad because A^2 = 0
        A_d = np.eye(nx) + A * self.dt
        
        # Exact Bd = B*dt + 0.5*A*B*dt^2
        B_d = B * self.dt + 0.5 * (A @ B) * (self.dt ** 2)
        
        # Exact gd = g*dt + 0.5*A*g*dt^2
        g_d = g * self.dt + 0.5 * (A @ g) * (self.dt ** 2)

        return A_d, B_d, g_d