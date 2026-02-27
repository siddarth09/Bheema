import numpy as np


class ComReference:
    """
    Generates centroidal reference trajectory for humanoid.

    State order:
      x = [ p_com(3),
            rpy(3),
            v_com(3),
            omega(3) ]
    """

    def __init__(self, horizon_N: int, dt: float):
        self.N = horizon_N
        self.dt = dt

    def generate(
        self,
        x0: np.ndarray,
        v_des_world: np.ndarray,
        z_des: float,
        yaw_rate_des: float = 0.0,
    ):
        """
        Returns x_ref (12 x N)

        x0: current centroidal state (12x1)
        v_des_world: desired linear velocity (3,)
        z_des: desired COM height
        yaw_rate_des: desired yaw rate
        """

        x_ref = np.zeros((12, self.N))

        p0 = x0[0:3, 0]
        yaw0 = x0[5, 0]

        for k in range(self.N):

            t = (k + 1) * self.dt

            # --- Position ---
            p_des = p0 + v_des_world * t
            p_des[2] = z_des  # lock height

            # --- Orientation ---
            roll_des = 0.0
            pitch_des = 0.0
            yaw_des = yaw0 + yaw_rate_des * t

            # --- Velocity ---
            v_des = v_des_world

            # --- Angular velocity ---
            omega_des = np.array([0.0, 0.0, yaw_rate_des])

            xk = np.concatenate([
                p_des,
                [roll_des, pitch_des, yaw_des],
                v_des,
                omega_des
            ])

            x_ref[:, k] = xk

        return x_ref
