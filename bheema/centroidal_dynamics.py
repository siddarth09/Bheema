import numpy as np
from scipy.linalg import expm

# =====================================================
# Utility
# =====================================================

def skew(r: np.ndarray) -> np.ndarray:
    """
    Skew-symmetric matrix such that:
        skew(r) @ f = r × f
    """
    r = r.reshape(3,)
    return np.array([
        [0.0,   -r[2],  r[1]],
        [r[2],   0.0,  -r[0]],
        [-r[1],  r[0],  0.0],
    ])


# =====================================================
# Centroidal Dynamics Model
# =====================================================

class CentroidalDynamics:
    """
    Centroidal dynamics for BHEEMA.

    State (12):
        x = [ p_com(3),
              rpy(3),
              v_com(3),
              omega(3) ]

    Input:
        u = [ f_L(3), f_R(3) ]   (foot forces in WORLD frame)
    """

    def __init__(self, mass: float, inertia_com: np.ndarray):
        self.m = float(mass)
        self.I = inertia_com.reshape(3, 3)
        self.I_inv = np.linalg.inv(self.I)

        self.g = np.array([0.0, 0.0, -9.81])

    # -------------------------------------------------
    # Continuous-time model
    # -------------------------------------------------
    def continuous_matrices(self, r_L: np.ndarray, r_R: np.ndarray):
        """
        Build continuous-time model:

            ẋ = A_c x + B_c u + g

        r_L, r_R:
            Foot lever arms from COM (WORLD frame)
        """

        I3 = np.eye(3)
        Z3 = np.zeros((3, 3))

        # State order: [p, rpy, v, omega]
        A_c = np.block([
            [Z3, Z3, I3, Z3],   # ṗ = v
            [Z3, Z3, Z3, I3],   # rpẏ ≈ ω
            [Z3, Z3, Z3, Z3],   # v̇ independent of x
            [Z3, Z3, Z3, Z3],   # ω̇ independent of x
        ])

        # Input matrix
        B_L = np.vstack([
            Z3,
            Z3,
            (1.0 / self.m) * I3,
            self.I_inv @ skew(r_L),
        ])

        B_R = np.vstack([
            Z3,
            Z3,
            (1.0 / self.m) * I3,
            self.I_inv @ skew(r_R),
        ])

        B_c = np.hstack([B_L, B_R])

        # Gravity
        g_c = np.zeros((12, 1))
        g_c[6:9, 0] = self.g

        return A_c, B_c, g_c


# =====================================================
# Discrete-Time Model (ZOH)
# =====================================================

class CentroidalDiscreteModel:
    """
    Zero-Order Hold discretization of centroidal dynamics.
    """

    def __init__(self, dt: float):
        self.dt = float(dt)

    def discretize(self, A_c: np.ndarray, B_c: np.ndarray, g_c: np.ndarray):
        """
        ZOH discretization using matrix exponential.

        Returns:
            A_d, B_d, g_d
        """

        nx = A_c.shape[0]
        nu = B_c.shape[1]

        # Augmented system:
        # | A  B  g |
        # | 0  0  0 |
        # | 0  0  0 |
        aug = np.zeros((nx + nu + 1, nx + nu + 1))
        aug[:nx, :nx] = A_c
        aug[:nx, nx:nx+nu] = B_c
        aug[:nx, -1:] = g_c

        exp_aug = expm(aug * self.dt)

        A_d = exp_aug[:nx, :nx]
        B_d = exp_aug[:nx, nx:nx+nu]
        g_d = exp_aug[:nx, -1:].reshape(nx, 1)

        return A_d, B_d, g_d

    # -------------------------------------------------
    # Single-step simulation (debug / testing)
    # -------------------------------------------------
    def step(self, x: np.ndarray, u: np.ndarray, A_d, B_d, g_d):
        """
        x_{k+1} = A_d x_k + B_d u_k + g_d
        """
        return A_d @ x + B_d @ u + g_d



def almost_zero(x, tol=1e-8):
    return np.linalg.norm(x) < tol

def test_gravity_compensation():
    """Testing the change in velocity to be zero when equal forces are applied"""
    print("\n=== Test 1: Gravity compensation ===")

    m = 40.0
    I = np.diag([2.0, 2.5, 1.2])
    dt = 0.02

    dyn = CentroidalDynamics(mass=m, inertia_com=I)
    disc = CentroidalDiscreteModel(dt)


    # Feet symmetric under COM
    rL = np.array([0.0,  0.1, -0.8])
    rR = np.array([0.0, -0.1, -0.8])

    A, B, g = dyn.continuous_matrices(rL, rR)
    Ad, Bd, gd = disc.discretize(A, B, g)

    x0 = np.zeros((12, 1))

    fz = 0.5 * m * 9.81
    u = np.array([
        0, 0, fz,
        0, 0, fz
    ]).reshape(6, 1)

    x1 = disc.step(x0, u, Ad, Bd, gd)

    dv = x1[6:9] - x0[6:9]

    print("Δv:", dv.flatten())
    assert almost_zero(dv), "Gravity compensation failed!"


def test_upward_acceleration():

    """When added vertical force, the COM should go upwards"""
    print("\n=== Test 2: Pure upward acceleration ===")

    m = 40.0
    I = np.diag([2.0, 2.5, 1.2])
    dt = 0.02

    dyn = CentroidalDynamics(m, I)
    disc = CentroidalDiscreteModel(dt)
    rL = np.array([0.0,  0.1, -0.8])
    rR = np.array([0.0, -0.1, -0.8])

    A, B, g = dyn.continuous_matrices(rL, rR)
    Ad, Bd, gd = disc.discretize(A, B, g)

    x0 = np.zeros((12, 1))

    fz = 0.5 * m * 9.81 + 200
    u = np.array([0,0,fz, 0,0,fz]).reshape(6,1)

    x1 = disc.step(x0, u, Ad, Bd, gd)

    v = x1[6:9]
    omega = x1[9:12]

    print("v_com:", v.flatten())
    print("omega:", omega.flatten())

    assert v[2] > 0, "COM should accelerate upward"
    assert almost_zero(omega), "No rotation expected!"

def test_pure_rotation():
    """Robot foot tipping effect,basically give more force on the left foot, to generate more torque"""
    print("\n=== Test 3: Asymmetric forces → rotation ===")

    m = 40.0
    I = np.diag([2.0, 2.5, 1.2])
    dt = 0.02

    dyn = CentroidalDynamics(m, I)
    disc = CentroidalDiscreteModel(dt)
    rL = np.array([0.0,  0.2, -0.8])
    rR = np.array([0.0, -0.2, -0.8])

    A, B, g = dyn.continuous_matrices(rL, rR)
    Ad, Bd, gd = disc.discretize(A, B, g)

    x0 = np.zeros((12, 1))

    fz_L = 0.6 * m * 9.81
    fz_R = 0.4 * m * 9.81

    u = np.array([0,0,fz_L, 0,0,fz_R]).reshape(6,1)

    x1 = disc.step(x0, u, Ad, Bd, gd)

    omega = x1[9:12]

    print("omega:", omega.flatten())

    assert not almost_zero(omega), "Rotation expected but got none!"

def test_free_fall():
    """To test if gravity is applied correctly"""
    print("\n=== Test 4: Free fall ===")

    m = 40.0
    I = np.diag([2.0, 2.5, 1.2])
    dt = 0.02

    dyn = CentroidalDynamics(m, I)
    disc = CentroidalDiscreteModel(dt)
    rL = np.array([0.0,  0.1, -0.8])
    rR = np.array([0.0, -0.1, -0.8])

    A, B, g = dyn.continuous_matrices(rL, rR)
    Ad, Bd, gd = disc.discretize(A, B, g)

    x0 = np.zeros((12,1))
    u = np.zeros((6,1))

    x1 = disc.step(x0, u, Ad, Bd, gd)

    v = x1[6:9]

    print("v_com:", v.flatten())
    assert v[2] < 0, "COM should fall downward"

if __name__ == "__main__":
    test_gravity_compensation()
    test_upward_acceleration()
    test_pure_rotation()
    test_free_fall()

    print("\n All centroidal dynamics tests passed.")
