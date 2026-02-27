import numpy as np

from bheema.mpc import CentroidalMPC
from bheema.centroidal_model import CentroidalDynamics, CentroidalDiscrete


# ===============================================================
# Helper
# ===============================================================

def build_model(N=6, dt=0.02, mass=40.0):

    I = np.diag([2.0, 2.5, 1.2])

    dyn = CentroidalDynamics(mass, I)
    disc = CentroidalDiscrete(dt)

    rL = np.array([0.0,  0.1, -0.8])
    rR = np.array([0.0, -0.1, -0.8])

    A_c, B_c, g_c = dyn.continuous_matrices(rL, rR)
    A_d, B_d, g_d = disc.discretize(A_c, B_c, g_c)

    Ad_seq = [A_d for _ in range(N)]
    Bd_seq = [B_d for _ in range(N)]
    gd_seq = [g_d for _ in range(N)]

    return Ad_seq, Bd_seq, gd_seq, mass


# ===============================================================
# 1️⃣ Gravity Compensation
# ===============================================================

def test_mpc_gravity():

    print("\n=== TEST 1: Gravity Compensation ===")

    N = 6
    Ad_seq, Bd_seq, gd_seq, mass = build_model(N)

    x0 = np.zeros((12, 1))
    x_ref = np.tile(x0, (1, N))
    contact_table = np.ones((2, N), dtype=int)

    mpc = CentroidalMPC(N=N, dt=0.02)
    u0 = mpc.solve(Ad_seq, Bd_seq, gd_seq, x0, x_ref, contact_table)

    fL = u0[0:3]
    fR = u0[3:6]

    total_fz = fL[2] + fR[2]

    print("Total Fz:", total_fz)
    print("Expected:", mass * 9.81)

    assert abs(total_fz - mass * 9.81) < 10.0
    print("✔ Passed")


# ===============================================================
# 2️⃣ Forward Velocity
# ===============================================================

def test_mpc_forward():

    print("\n=== TEST 2: Forward Velocity Tracking ===")

    N = 6
    Ad_seq, Bd_seq, gd_seq, mass = build_model(N)

    x0 = np.zeros((12, 1))
    x_ref = np.zeros((12, N))
    x_ref[6, :] = 0.3  # desired vx

    contact_table = np.ones((2, N), dtype=int)

    mpc = CentroidalMPC(N=N, dt=0.02)
    u0 = mpc.solve(Ad_seq, Bd_seq, gd_seq, x0, x_ref, contact_table)

    fL = u0[0:3]
    fR = u0[3:6]

    total_fx = fL[0] + fR[0]

    print("Total Fx:", total_fx)

    assert total_fx > 0.0
    print("✔ Passed")


# ===============================================================
# 3️⃣ Single Support
# ===============================================================

def test_mpc_single_support():

    print("\n=== TEST 3: Single Support ===")

    N = 6
    Ad_seq, Bd_seq, gd_seq, mass = build_model(N)

    x0 = np.zeros((12, 1))
    x_ref = np.tile(x0, (1, N))

    contact_table = np.ones((2, N), dtype=int)
    contact_table[1, :] = 0  # right foot swing

    mpc = CentroidalMPC(N=N, dt=0.02)
    u0 = mpc.solve(Ad_seq, Bd_seq, gd_seq, x0, x_ref, contact_table)

    fL = u0[0:3]
    fR = u0[3:6]

    print("Right foot force:", fR)

    assert np.allclose(fR, np.zeros(3), atol=1e-3)
    print("✔ Passed")


# ===============================================================
# 4️⃣ Yaw Rate Tracking
# ===============================================================

def test_mpc_yaw():

    print("\n=== TEST 4: Yaw Rate Tracking ===")

    N = 6
    Ad_seq, Bd_seq, gd_seq, mass = build_model(N)

    x0 = np.zeros((12, 1))
    x_ref = np.zeros((12, N))
    x_ref[11, :] = 0.5  # desired yaw rate

    contact_table = np.ones((2, N), dtype=int)

    mpc = CentroidalMPC(N=N, dt=0.02)
    u0 = mpc.solve(Ad_seq, Bd_seq, gd_seq, x0, x_ref, contact_table)

    fL = u0[0:3]
    fR = u0[3:6]

    rL = np.array([0.0,  0.1, -0.8])
    rR = np.array([0.0, -0.1, -0.8])

    tauL = np.cross(rL, fL)
    tauR = np.cross(rR, fR)

    tau_total = tauL + tauR

    print("Yaw torque:", tau_total[2])

    assert tau_total[2] != 0



# ===============================================================
# 5️⃣ Asymmetric Lever Arm Test
# ===============================================================

def test_mpc_asymmetric():

    print("\n=== TEST 5: Asymmetric Lever Arm ===")

    dt = 0.02
    N = 6
    mass = 40.0
    I = np.diag([2.0, 2.5, 1.2])

    dyn = CentroidalDynamics(mass, I)
    disc = CentroidalDiscrete(dt)

    # Asymmetric feet
    rL = np.array([0.2,  0.1, -0.8])
    rR = np.array([-0.2, -0.1, -0.8])

    A_c, B_c, g_c = dyn.continuous_matrices(rL, rR)
    A_d, B_d, g_d = disc.discretize(A_c, B_c, g_c)

    Ad_seq = [A_d for _ in range(N)]
    Bd_seq = [B_d for _ in range(N)]
    gd_seq = [g_d for _ in range(N)]

    x0 = np.zeros((12, 1))
    x_ref = np.zeros((12, N))
    x_ref[11, :] = 0.5

    contact_table = np.ones((2, N), dtype=int)

    mpc = CentroidalMPC(N=N, dt=dt)
    u0 = mpc.solve(Ad_seq, Bd_seq, gd_seq, x0, x_ref, contact_table)

    fL = u0[0:3]
    fR = u0[3:6]

    print("Left force:", fL)
    print("Right force:", fR)

    assert not np.allclose(fL, fR)
    print("✔ Passed")


if __name__ == "__main__":

    test_mpc_gravity()
    test_mpc_forward()
    test_mpc_single_support()
    test_mpc_yaw()
    test_mpc_asymmetric()

    print("\n🎯 ALL MPC TESTS COMPLETED")
