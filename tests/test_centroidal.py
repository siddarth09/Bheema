import numpy as np
from bheema.centroidal_model import CentroidalDiscrete,CentroidalDynamics


def almost_zero(x, tol=1e-8):
    return np.linalg.norm(x) < tol


def test_gravity_compensation():
    print("\n=== Test 1: Gravity Compensation ===")

    m = 40.0
    I = np.diag([2.0, 2.5, 1.2])
    dt = 0.02

    dyn = CentroidalDynamics(m, I)
    disc = CentroidalDiscrete(dt)

    # Symmetric feet
    rL = np.array([0.0,  0.1, -0.8])
    rR = np.array([0.0, -0.1, -0.8])

    A, B, g = dyn.continuous_matrices(rL, rR)
    Ad, Bd, gd = disc.discretize(A, B, g)

    x0 = np.zeros((12, 1))

    # Each foot supports half weight
    fz = 0.5 * m * 9.81
    u = np.array([
        0, 0, fz,
        0, 0, fz
    ]).reshape(6, 1)

    x1 = Ad @ x0 + Bd @ u + gd

    dv = x1[6:9] - x0[6:9]

    print("Δv:", dv.flatten())
    assert almost_zero(dv), "Gravity compensation failed!"


def test_upward_acceleration():
    print("\n=== Test 2: Upward Acceleration ===")

    m = 40.0
    I = np.diag([2.0, 2.5, 1.2])
    dt = 0.02

    dyn = CentroidalDynamics(m, I)
    disc = CentroidalDiscrete(dt)

    rL = np.array([0.0,  0.1, -0.8])
    rR = np.array([0.0, -0.1, -0.8])

    A, B, g = dyn.continuous_matrices(rL, rR)
    Ad, Bd, gd = disc.discretize(A, B, g)

    x0 = np.zeros((12, 1))

    # Add extra upward force
    fz = 0.5 * m * 9.81 + 200
    u = np.array([
        0, 0, fz,
        0, 0, fz
    ]).reshape(6, 1)

    x1 = Ad @ x0 + Bd @ u + gd

    v = x1[6:9]
    omega = x1[9:12]

    print("v_com:", v.flatten())
    print("omega:", omega.flatten())

    assert v[2] > 0, "COM should accelerate upward"
    assert almost_zero(omega), "No rotation expected!"


def test_pure_rotation():
    print("\n=== Test 3: Asymmetric Forces → Rotation ===")

    m = 40.0
    I = np.diag([2.0, 2.5, 1.2])
    dt = 0.02

    dyn = CentroidalDynamics(m, I)
    disc = CentroidalDiscrete(dt)

    rL = np.array([0.0,  0.2, -0.8])
    rR = np.array([0.0, -0.2, -0.8])

    A, B, g = dyn.continuous_matrices(rL, rR)
    Ad, Bd, gd = disc.discretize(A, B, g)

    x0 = np.zeros((12, 1))

    fz_L = 0.6 * m * 9.81
    fz_R = 0.4 * m * 9.81

    u = np.array([
        0, 0, fz_L,
        0, 0, fz_R
    ]).reshape(6, 1)

    x1 = Ad @ x0 + Bd @ u + gd

    omega = x1[9:12]

    print("omega:", omega.flatten())

    assert not almost_zero(omega), "Rotation expected but none detected!"


def test_free_fall():
    print("\n=== Test 4: Free Fall ===")

    m = 40.0
    I = np.diag([2.0, 2.5, 1.2])
    dt = 0.02

    dyn = CentroidalDynamics(m, I)
    disc = CentroidalDiscrete(dt)

    rL = np.array([0.0,  0.1, -0.8])
    rR = np.array([0.0, -0.1, -0.8])

    A, B, g = dyn.continuous_matrices(rL, rR)
    Ad, Bd, gd = disc.discretize(A, B, g)

    x0 = np.zeros((12, 1))
    u = np.zeros((6, 1))

    x1 = Ad @ x0 + Bd @ u + gd

    v = x1[6:9]

    print("v_com:", v.flatten())

    assert v[2] < 0, "COM should fall downward"


if __name__ == "__main__":
    test_gravity_compensation()
    test_upward_acceleration()
    test_pure_rotation()
    test_free_fall()

    print("\nAll centroidal dynamics tests passed.")
