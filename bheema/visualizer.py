import matplotlib.pyplot as plt
import numpy as np

class Bheemaplotter:
    def __init__(self, joint_names):
        self.joint_names = joint_names
        # Data buffers
        self.t_history = []
        self.x_history = []
        self.u_history = []
        self.tau_history = []

    def record(self, t, x, u, tau):
        """Buffer data during the control loop."""
        self.t_history.append(t)
        self.x_history.append(x.flatten().copy())
        self.u_history.append(u.flatten().copy())
        self.tau_history.append(tau.copy())

    def plot(self):
        """Generate static plots."""
        t = np.array(self.t_history) - self.t_history[0]
        x = np.array(self.x_history)
        u = np.array(self.u_history)
        tau = np.array(self.tau_history)

        fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

        # 1. COM Height (z) and Forward Velocity (vx)
        axes[0].plot(t, x[:, 2], label='Height [m]')
        axes[0].plot(t, x[:, 6], label='Forward Vel [m/s]')
        axes[0].set_title("Centroidal State")
        axes[0].legend()
        axes[0].grid(True)

        # 2. MPC Ground Reaction Forces (Fz)
        axes[1].plot(t, u[:, 2], label='Left Fz [N]')
        axes[1].plot(t, u[:, 5], label='Right Fz [N]')
        axes[1].set_title("MPC Contact Forces")
        axes[1].set_ylabel("Force [N]")
        axes[1].legend()
        axes[1].grid(True)

        # 3. WBC Joint Torques
        # Plotting only the first 6 (Left Leg) for clarity; adjust as needed
        for i in range(6):
            axes[2].plot(t, tau[:, i], label=self.joint_names[i])
        axes[2].set_title("WBC Joint Torques (Left Leg)")
        axes[2].set_ylabel("Torque [Nm]")
        axes[2].set_xlabel("Time [s]")
        axes[2].legend(loc='upper right', ncol=2)
        axes[2].grid(True)

        plt.tight_layout()
        plt.show()