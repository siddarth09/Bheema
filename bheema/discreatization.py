import numpy as np
import pinocchio as pin

from state import G1State


def skew(r: np.ndarray) -> np.ndarray:
    """Return [r]_x such that [r]_x f = r × f."""
    r = r.reshape(3,)
    return np.array([
        [0.0,   -r[2],  r[1]],
        [r[2],   0.0,  -r[0]],
        [-r[1],  r[0],  0.0],
    ], dtype=float)


class CentroidalDiscreteModel:
    """
    x = [ p_com(3),
          theta_rpy(3),
          v_com(3),
          omega_world(3) ]   (12x1)

    u = [ f_L(3), f_R(3) ]  (6x1)
    """

    def __init__(self, dt: float):
        self.dt = float(dt)

    # ------------------------------------------------------------------
    # Discrete model construction
    # ------------------------------------------------------------------
    def build_Ad_Bd_gd(self, m: float, I: np.ndarray,
                       r_L: np.ndarray, r_R: np.ndarray):

        dt = self.dt
        I = np.asarray(I, dtype=float).reshape(3, 3)
        I_inv = np.linalg.inv(I)

        I3 = np.eye(3)
        Z3 = np.zeros((3, 3))

        # Continuous A
        A = np.block([
            [Z3, Z3, I3, Z3],   # p_dot = v
            [Z3, Z3, Z3, I3],   # theta_dot = omega
            [Z3, Z3, Z3, Z3],   # v_dot independent
            [Z3, Z3, Z3, Z3],   # omega_dot independent
        ])

        # Continuous B
        B_L = np.vstack([
            Z3,
            Z3,
            (1.0 / m) * I3,
            I_inv @ skew(r_L),
        ])

        B_R = np.vstack([
            Z3,
            Z3,
            (1.0 / m) * I3,
            I_inv @ skew(r_R),
        ])

        B = np.hstack([B_L, B_R])

        # Gravity
        g_vec = np.zeros((12, 1))
        g_vec[6:9, 0] = np.array([0.0, 0.0, -9.81])

        # Euler discretization
        Ad = np.eye(12) + dt * A
        Bd = dt * B
        gd = dt * g_vec

        return Ad, Bd, gd

    # ------------------------------------------------------------------
    def step(self, x, u, Ad, Bd, gd):
        x = np.asarray(x).reshape(12, 1)
        u = np.asarray(u).reshape(6, 1)
        return Ad @ x + Bd @ u + gd


# ----------------------------------------------------------------------
# Helper: build x from G1State
# ----------------------------------------------------------------------
def build_centroidal_x(robot: G1State) -> np.ndarray:
    """
    x = [p_com, rpy, v_com, omega]
    """
    rpy = robot.get_base_orientation_rpy()
    p_com, v_com = robot.get_com_state_world()
    omega = robot.get_base_angular_velocity_world()

    x = np.concatenate([
        p_com,
        rpy,
        v_com,
        omega
    ])

    return x.reshape(12, 1)


# ----------------------------------------------------------------------
# Test with REAL G1 model
# ----------------------------------------------------------------------
def test_centroidal_step_with_g1():

    dt = 0.02
    model = CentroidalDiscreteModel(dt)
    robot = G1State()

    q = pin.neutral(robot.model)
    dq = np.zeros(robot.model.nv)
    q[2] = 0.75

    robot.update(q, dq)

    x0 = build_centroidal_x(robot)

    m = robot.model.mass
    I = robot.data.Ig.inertia.copy()
    r_L, r_R = robot.get_foot_levers_world()

    Ad, Bd, gd = model.build_Ad_Bd_gd(m, I, r_L, r_R)

    fz = 0.5 * m * 9.81
    u = np.array([0, 0, fz, 0, 0, fz]).reshape(6, 1)

    x1 = model.step(x0, u, Ad, Bd, gd)
    print("\n[REAL G1] Gravity test Δv:", x1[6:9].flatten())

    u[2] += 200
    u[5] += 200
    x2 = model.step(x0, u, Ad, Bd, gd)
    print("[REAL G1] Upward v:", x2[6:9].flatten())
    print("[REAL G1] omega:", x2[9:12].flatten())


# ----------------------------------------------------------------------
# Toy sanity tests (no Pinocchio)
# ----------------------------------------------------------------------
def main():

    dt = 0.02
    model = CentroidalDiscreteModel(dt)

    m = 45.0
    I = np.diag([2.0, 2.5, 0.8])
    r_L = np.array([-0.03,  0.12, -0.67])
    r_R = np.array([-0.03, -0.12, -0.67])

    Ad, Bd, gd = model.build_Ad_Bd_gd(m, I, r_L, r_R)

    x0 = np.zeros((12, 1))
    x0[2, 0] = 1.0

    fz = 0.5 * m * 9.81
    u = np.array([0, 0, fz, 0, 0, fz]).reshape(6, 1)

    x1 = model.step(x0, u, Ad, Bd, gd)
    print("\n[TOY] Gravity Δv:", x1[6:9].flatten())

    u[2] += 300
    x2 = model.step(x0, u, Ad, Bd, gd)
    print("[TOY] omega:", x2[9:12].flatten())


if __name__ == "__main__":
    main()
