import numpy as np
from bheema.g1_config import PinG1Model 
from bheema.gait import Gait
from numpy import cos, sin

class ComTraj:
    def __init__(self, g1: PinG1Model):
        self.dummy_g1 = PinG1Model()
        
        # Initialize trajectory endpoints
        self.pos_des_world = np.zeros(3)
        
        # Seed first trajectory from current state
        x_vec = g1.compute_com_x_vec().reshape(-1)
        self.pos_des_world[0:2] = x_vec[0:2]
        self.pos_des_world[2] = 0.60 # Standard G1 nominal height
        
        self.N = 16 
        self.nx = 12
        self.nu = 12

    def compute_x_ref_vec(self):
        refs = [
            self.pos_traj_world,
            self.rpy_traj_world,
            self.vel_traj_world,
            self.omega_traj_world,
        ]
        # stack into shape (12, N)
        N = min(r.shape[1] for r in refs)
        ref_traj = np.vstack([r[:, :N] for r in refs])
        return ref_traj 

    def generate_traj(self,
                        g1: PinG1Model,
                        gait: Gait,
                        time_now: float,
                        x_vel_des_body: float,
                        y_vel_des_body: float,
                        z_pos_des_body: float,
                        yaw_rate_des_body: float,
                        time_step: float):
        
        self.initial_x_vec = g1.compute_com_x_vec()
        initial_pos = self.initial_x_vec[0:3].flatten()
        self.m = g1.data.Ig.mass
        self.I_com_world = g1.data.Ig.inertia
        
        x0, y0, z0 = initial_pos
        yaw = self.initial_x_vec[5]
        
        time_horizon = gait.gait_period * 1.5

        # 1) Clamp desired world COM to stay near current position (Go2 style)
        max_pos_error = 0.15   # slightly looser for biped sway
        
        if self.pos_des_world[0] - x0 > max_pos_error:
            self.pos_des_world[0] = x0 + max_pos_error
        if x0 - self.pos_des_world[0] > max_pos_error:
            self.pos_des_world[0] = x0 - max_pos_error

        if self.pos_des_world[1] - y0 > max_pos_error:
            self.pos_des_world[1] = y0 + max_pos_error
        if y0 - self.pos_des_world[1] > max_pos_error:
            self.pos_des_world[1] = y0 - max_pos_error

        self.pos_des_world[2] = z_pos_des_body

        # 2) Time horizon
        self.N = int(time_horizon / time_step)
        N = self.N
        t_vec = (np.arange(N) + 1) * time_step      
        self.time = t_vec

        # Rotation
        R_z = g1.R_z
        vel_desired_world = R_z @ np.array([x_vel_des_body, y_vel_des_body, 0.0])

        # 3) Allocate trajectory arrays
        self.pos_traj_world     = np.zeros((3, N))
        self.vel_traj_world     = np.zeros((3, N))
        self.rpy_traj_world     = np.zeros((3, N))
        self.omega_traj_world   = np.zeros((3, N))

        current_sway_y = y0 
        
        # ZMP Sway Logic 
        for i in range(N):
            mask = gait.compute_current_mask(time_now + i * time_step)
            
            # 1. Determine the raw target
            if mask[0] == 1 and mask[1] == 1: 
                zmp_target_y = 0.0 
            elif mask[0] == 1: 
                zmp_target_y = (gait.NOMINAL_STANCE_WIDTH / 2.0) * 0.6 # Left Stance
            else: 
                zmp_target_y = -(gait.NOMINAL_STANCE_WIDTH / 2.0) * 0.6 # Right Stance
            
            # 2. Smooth the Sway (Low-Pass Filter) to prevent teleporting
            filter_alpha = 0.85  # 0.0 = instant jump, 1.0 = never moves
            current_sway_y = filter_alpha * current_sway_y + (1.0 - filter_alpha) * zmp_target_y
            
            # 3. Assign Position
            self.pos_traj_world[0, i] = x0 + vel_desired_world[0] * t_vec[i]
            self.pos_traj_world[1, i] = current_sway_y 
            self.pos_traj_world[2, i] = z_pos_des_body

            # 4. Mathematically tie Velocity to the smoothed Position
            if i == 0:
                vy_sway = (current_sway_y - y0) / time_step
            else:
                vy_sway = (current_sway_y - self.pos_traj_world[1, i-1]) / time_step
                
            self.vel_traj_world[0, i] = vel_desired_world[0]
            self.vel_traj_world[1, i] = vel_desired_world[1] + vy_sway # Add sway vel to commanded vel
            self.vel_traj_world[2, i] = 0.0

        # Linear velocity in world
        self.vel_traj_world[:, :] = vel_desired_world.reshape(3, 1)

        # RPY in world
        self.rpy_traj_world[0, :] = 0.0
        self.rpy_traj_world[1, :] = 0.0
        self.rpy_traj_world[2, :] = yaw + yaw_rate_des_body * t_vec

        # RPY rates in BODY frame
        self.omega_traj_world[0, :] = 0.0
        self.omega_traj_world[1, :] = 0.0
        self.omega_traj_world[2, :] = yaw_rate_des_body

        self.contact_table = gait.compute_contact_table(time_now, time_step, N)

        r_l_traj_world = np.zeros((3, N))
        r_r_traj_world = np.zeros((3, N))

        [r_l_next_td_world, r_r_next_td_world] = g1.get_foot_lever_world()

        # Biped uses 2 element mask [Left, Right]
        mask_previous = np.array([2, 2])
        
        # 50-state configuration vector for G1 dummy update
        q_dummy = self.dummy_g1.current_config.get_q()
        dq_dummy = self.dummy_g1.current_config.get_dq()

        for i in range(N):
            current_mask = gait.compute_current_mask(time_now + i * time_step)

            # Dummy Kinematics Update (Go2 Style adapted to 50-DOF G1)
            q_dummy[0:3] = self.pos_traj_world[:, i]
            
            cy, sy = cos(self.rpy_traj_world[2, i]/2.0), sin(self.rpy_traj_world[2, i]/2.0)
            q_dummy[3:7] = np.array([0.0, 0.0, sy, cy]) # Roll/Pitch 0, Yaw mapped to Quat
            
            R = g1.R_world_to_body
            dq_dummy[0:3] = R @ self.vel_traj_world[:, i]
            dq_dummy[3:6] = R @ self.omega_traj_world[:, i]
            
            self.dummy_g1.update_model(q_dummy, dq_dummy)
            p_base_traj_world = self.dummy_g1.current_config.base_pos

            ## Left Foot Logic
            if current_mask[0] != mask_previous[0] and current_mask[0] == 0:
                # Takes off
                pos_l_next_td_world = gait.compute_touchdown_world_for_traj_purpose_only(self.dummy_g1, "LEFT")
                r_l_next_td_world = pos_l_next_td_world - p_base_traj_world
                r_l_traj_world[:, i] = np.array([0, 0, 0])

            elif current_mask[0] != mask_previous[0] and current_mask[0] == 1:
                # Touch down
                r_l_traj_world[:, i] = r_l_next_td_world 

            elif current_mask[0] == mask_previous[0]:
                # No change from last time step
                r_l_traj_world[:, i] = r_l_traj_world[:, i-1] 

            ## Right Foot Logic
            if current_mask[1] != mask_previous[1] and current_mask[1] == 0:
                # Takes off
                pos_r_next_td_world = gait.compute_touchdown_world_for_traj_purpose_only(self.dummy_g1, "RIGHT")
                r_r_next_td_world = pos_r_next_td_world - p_base_traj_world
                r_r_traj_world[:, i] = np.array([0, 0, 0])

            elif current_mask[1] != mask_previous[1] and current_mask[1] == 1:
                # Touch down
                r_r_traj_world[:, i] = r_r_next_td_world 

            elif current_mask[1] == mask_previous[1]:
                # No change from last time step
                r_r_traj_world[:, i] = r_r_traj_world[:, i-1] 

            mask_previous = current_mask

        # Save
        self.r_l_foot_world = r_l_traj_world
        self.r_r_foot_world = r_r_traj_world

        # Update the traj dynamics
        self._continuousDynamics(g1)
        self._discreteDynamics(time_step)
        self.x_ref = self.compute_x_ref_vec()

    def _skew(self, vector):
        return np.array([
            [0, -vector[2], vector[1]],
            [vector[2], 0, -vector[0]],
            [-vector[1], vector[0], 0]
        ])
    
    def _continuousDynamics(self, g1):
        m = self.m
        I_inv = np.linalg.inv(self.I_com_world)
        yaw_avg = np.average(self.rpy_traj_world[2, :])
        
        R_z_T = np.array([
            [cos(yaw_avg),  sin(yaw_avg), 0],
            [-sin(yaw_avg), cos(yaw_avg), 0],
            [0,             0,            1]
        ])

        self.Ac = np.block([
            [np.zeros((3, 3)), np.zeros((3, 3)), np.eye(3),        np.zeros((3, 3))],
            [np.zeros((3, 3)), np.zeros((3, 3)), np.zeros((3, 3)), R_z_T           ],
            [np.zeros((3, 3)), np.zeros((3, 3)), np.zeros((3, 3)), np.zeros((3, 3))],
            [np.zeros((3, 3)), np.zeros((3, 3)), np.zeros((3, 3)), np.zeros((3, 3))],
        ])

        self.Bc = np.zeros((self.N, 12, 12))
        for i in range(self.N):
            r_L = self.r_l_foot_world[:, i]
            r_R = self.r_r_foot_world[:, i]

            skew_L = self._skew(r_L)
            skew_R = self._skew(r_R)

            # Columns: [F_left(3), Tau_left(3), F_right(3), Tau_right(3)]
            self.Bc[i] = np.block([
                [np.zeros((3,3)),   np.zeros((3,3)),   np.zeros((3,3)),   np.zeros((3,3))],
                [np.zeros((3,3)),   np.zeros((3,3)),   np.zeros((3,3)),   np.zeros((3,3))],
                [(1/m)*np.eye(3),   np.zeros((3,3)),   (1/m)*np.eye(3),   np.zeros((3,3))],
                [I_inv @ skew_L,    I_inv,             I_inv @ skew_R,    I_inv],
            ])

        # Gravity Vector
        self.gc = np.array([0, 0, 0,  0, 0, 0,  0, 0, -9.81,  0, 0, 0])

    def _discreteDynamics(self, dt: float):
        N = self.N
        m = self.m
        I_inv = np.linalg.inv(self.I_com_world)

        yaw_avg = float(np.average(self.rpy_traj_world[2, :]))
        cy, sy = np.cos(yaw_avg), np.sin(yaw_avg)
        RzT = np.array([[cy, sy, 0.0],
                        [-sy, cy, 0.0],
                        [0.0, 0.0, 1.0]], dtype=float)

        # ---- Ad ----
        Ad = np.eye(12, dtype=float)
        Ad[0:3, 6:9] = dt * np.eye(3)        # p <- v
        Ad[3:6, 9:12] = dt * RzT             # rpy <- omega 
        self.Ad = Ad

        # ---- gd ---- 
        g = np.array([0.0, 0.0, -9.81], dtype=float)
        gd = np.zeros((12, 1), dtype=float)
        gd[0:3, 0] = 0.5 * g * dt * dt       # p += 1/2 g dt^2
        gd[6:9, 0] = g * dt                  # v += g dt
        self.gd = gd

        # ---- Bd ----
        Bd = np.zeros((N, 12, 12), dtype=float)

        Bp_F = (0.5 * dt * dt / m) * np.eye(3) # p from forces
        Bv_F = (dt / m) * np.eye(3)            # v from forces
        
        # Explicit torques do not translate CoM
        Bp_Tau = np.zeros((3,3))
        Bv_Tau = np.zeros((3,3))

        for i in range(N):
            r_L = self.r_l_foot_world[:, i]
            r_R = self.r_r_foot_world[:, i]

            # Angular influence from forces (lever arm)
            W_L_F = I_inv @ self._skew(r_L)
            W_R_F = I_inv @ self._skew(r_R)
            
            # Angular influence from explicit torques (direct injection)
            W_L_Tau = I_inv
            W_R_Tau = I_inv

            Bi = Bd[i]

            # 1. Position block (p)
            Bi[0:3, 0:3]   = Bp_F
            Bi[0:3, 3:6]   = Bp_Tau
            Bi[0:3, 6:9]   = Bp_F
            Bi[0:3, 9:12]  = Bp_Tau

            # 2. Velocity block (v)
            Bi[6:9, 0:3]   = Bv_F
            Bi[6:9, 3:6]   = Bv_Tau
            Bi[6:9, 6:9]   = Bv_F
            Bi[6:9, 9:12]  = Bv_Tau

            # 3. Angular Velocity block (omega)
            Bi[9:12, 0:3]  = dt * W_L_F
            Bi[9:12, 3:6]  = dt * W_L_Tau
            Bi[9:12, 6:9]  = dt * W_R_F
            Bi[9:12, 9:12] = dt * W_R_Tau

            # 4. Orientation block (rpy) - 2nd order term
            Bi[3:6, 0:3]   = 0.5 * dt * dt * (RzT @ W_L_F)
            Bi[3:6, 3:6]   = 0.5 * dt * dt * (RzT @ W_L_Tau)
            Bi[3:6, 6:9]   = 0.5 * dt * dt * (RzT @ W_R_F)
            Bi[3:6, 9:12]  = 0.5 * dt * dt * (RzT @ W_R_Tau)

        self.Bd = Bd

# ==============================================================================
# DEBUG: Plot the Generated Trajectory
# ==============================================================================
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    print("Running ComTraj Debugger...")

    # Initialize Dummy Robot and Gait
    g1 = PinG1Model()
    gait = Gait(frequency_hz=1.0, duty=0.68)
    traj = ComTraj(g1)

    # Dummy walking inputs
    time_now = 0.0
    time_step = 0.02
    x_vel_des = 0.4
    y_vel_des = 0.0
    z_pos_des = 0.60
    yaw_rate_des = 0.0

    # Generate
    traj.generate_traj(
        g1, gait, time_now,
        x_vel_des, y_vel_des, z_pos_des, yaw_rate_des,
        time_step
    )

    t = traj.time

    # Create the Plot Layout
    fig, axs = plt.subplots(4, 1, figsize=(10, 12), sharex=True)

    # 1. CoM Position
    axs[0].plot(t, traj.pos_traj_world[0, :], label='CoM X', color='blue')
    axs[0].plot(t, traj.pos_traj_world[1, :], label='CoM Y (Sway)', color='orange')
    axs[0].plot(t, traj.pos_traj_world[2, :], label='CoM Z', color='green')
    axs[0].set_ylabel('Position (m)')
    axs[0].legend(loc="upper left")
    axs[0].grid(True)
    axs[0].set_title('Reference CoM Position (Check the orange sway line for smoothness)')

    # 2. CoM Velocity
    axs[1].plot(t, traj.vel_traj_world[0, :], label='Vel X', color='blue')
    axs[1].plot(t, traj.vel_traj_world[1, :], label='Vel Y', color='orange')
    axs[1].set_ylabel('Velocity (m/s)')
    axs[1].legend(loc="upper left")
    axs[1].grid(True)
    axs[1].set_title('Reference CoM Velocity')

    # 3. Relative Foot Placement (Lever Arms)
    axs[2].plot(t, traj.r_l_foot_world[1, :], label='Left Foot relative Y', color='blue')
    axs[2].plot(t, traj.r_r_foot_world[1, :], label='Right Foot relative Y', color='orange')
    axs[2].set_ylabel('Distance (m)')
    axs[2].legend(loc="upper left")
    axs[2].grid(True)
    axs[2].set_title('Foot placement relative to CoM (If these jump sharply, MPC will crash)')

    # 4. Contact Schedule
    axs[3].plot(t, traj.contact_table[0, :], label='Left Contact', drawstyle='steps-post', color='blue')
    axs[3].plot(t, traj.contact_table[1, :], label='Right Contact', drawstyle='steps-post', linestyle='--', color='orange')
    axs[3].set_ylabel('Contact (1/0)')
    axs[3].set_xlabel('Time Horizon (s)')
    axs[3].legend(loc="upper left")
    axs[3].grid(True)
    axs[3].set_title('Gait Contact Schedule')

    plt.tight_layout()
    plt.show()