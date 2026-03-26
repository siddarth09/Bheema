import numpy as np
from bheema.g1_config import PinG1Model 
from .gait import Gait
from numpy import cos, sin

class ComTraj:
    def __init__(self, g1: PinG1Model):
        self.dummy_g1 = PinG1Model()
        
        # 1. Properly initialize the array before assignment
        self.pos_des_world = np.zeros(3)
        
        # 2. Get current state to seed the first trajectory
        x_vec = g1.compute_com_x_vec().reshape(-1)
        self.pos_des_world[0:2] = x_vec[0:2]
        
        # 3. Use the nominal height as a baseline
        self.pos_des_world[2] = 0.68 
        
        self.N = 30 # This will be updated in generate_traj
        self.nx = 12
        self.nu = 12

    def compute_x_ref_vec(self):
        refs = [
            self.pos_traj_world,
            self.rpy_traj_world,
            self.vel_traj_world,
            self.omega_traj_world,
        ]
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
        
        # Increase lookahead to 1.5 periods for better "anticipation"
        time_horizon = gait.gait_period * 1.5

        # --- FIX 1: LOOSEN POSITION CLAMP ---
        # If this is too small (0.1), the robot "stutters" because it can't 
        # move as fast as the velocity command suggests.
        max_pos_error = 0.3   
        
        if self.pos_des_world[0] - x0 > max_pos_error:
            self.pos_des_world[0] = x0 + max_pos_error
        elif x0 - self.pos_des_world[0] > max_pos_error:
            self.pos_des_world[0] = x0 - max_pos_error

        if self.pos_des_world[1] - y0 > max_pos_error:
            self.pos_des_world[1] = y0 + max_pos_error
        elif y0 - self.pos_des_world[1] > max_pos_error:
            self.pos_des_world[1] = y0 - max_pos_error

        self.pos_des_world[2] = z_pos_des_body

        self.N = int(time_horizon / time_step)
        N = self.N
        t_vec = (np.arange(N) + 1) * time_step      
        self.time = t_vec

        R_z = g1.R_z
        vel_desired_world = R_z @ np.array([x_vel_des_body, y_vel_des_body, 0.0])

        self.pos_traj_world     = np.zeros((3, N))
        self.vel_traj_world     = np.zeros((3, N))
        self.rpy_traj_world     = np.zeros((3, N))
        self.omega_traj_world   = np.zeros((3, N))

        self.pos_traj_world[:, :] = (
            self.pos_des_world.reshape(3, 1) + (vel_desired_world.reshape(3, 1) * t_vec.reshape(1, N))
        )

        self.vel_traj_world[:, :] = vel_desired_world.reshape(3, 1)

        self.rpy_traj_world[0, :] = 0.0
        self.rpy_traj_world[1, :] = 0.0
        self.rpy_traj_world[2, :] = yaw + yaw_rate_des_body * t_vec

        self.omega_traj_world[0, :] = 0.0
        self.omega_traj_world[1, :] = 0.0
        self.omega_traj_world[2, :] = yaw_rate_des_body

        self.contact_table = gait.compute_contact_table(time_now, time_step, N)

        # Biped Foot trajectories
        r_l_traj_world = np.zeros((3,N))
        r_r_traj_world = np.zeros((3,N))

        [r_l_next_td_world, r_r_next_td_world] = g1.get_foot_lever_world()
        mask_previous = np.array([2, 2])

        for i in range(N):
            current_mask = gait.compute_current_mask(time_now + i * time_step)
            self.dummy_g1.current_config.base_pos = self.pos_traj_world[:, i]
            
            cy = cos(self.rpy_traj_world[2, i]/2.0)
            sy = sin(self.rpy_traj_world[2, i]/2.0)
            self.dummy_g1.current_config.base_quad = np.array([0.0, 0.0, sy, cy]) 
            
            R = g1.R_world_to_body
            self.dummy_g1.current_config.base_vel = R @ self.vel_traj_world[:, i]
            self.dummy_g1.current_config.base_ang_vel = R @ self.omega_traj_world[:, i]
            self.dummy_g1.update_model()

            p_base_traj_world = self.dummy_g1.pos_com_world

            ## Left foot logic (Simplified for stability)
            if current_mask[0] == 0: # Swing
                pos_l_next_td_world = gait.compute_touchdown_world_for_traj_purpose_only(self.dummy_g1, "LEFT") 
                r_l_next_td_world = pos_l_next_td_world - p_base_traj_world
                r_l_traj_world[:,i] = r_l_next_td_world # Preview the next step
            else: # Stance
                r_l_traj_world[:,i] = r_l_next_td_world 

            ## Right foot logic
            if current_mask[1] == 0: # Swing
                pos_r_next_td_world = gait.compute_touchdown_world_for_traj_purpose_only(self.dummy_g1, "RIGHT") 
                r_r_next_td_world = pos_r_next_td_world - p_base_traj_world
                r_r_traj_world[:,i] = r_r_next_td_world
            else: # Stance
                r_r_traj_world[:,i] = r_r_next_td_world 

        self.r_l_foot_world = r_l_traj_world
        self.r_r_foot_world = r_r_traj_world

        self._continuousDynamics(g1)
        self._discreteDynamics(time_step)
       
        self.x_ref = self.compute_x_ref_vec()

    def _skew(self,vector):
        return np.array([
            [0, -vector[2], vector[1]],
            [vector[2], 0, -vector[0]],
            [-vector[1], vector[0], 0]
        ])
    
    def _continuousDynamics(self, g1):
        m = self.m
        I_com_world = self.I_com_world

        yaw_avg = np.average(self.rpy_traj_world[2, :])
        R_z = np.array([
            [cos(yaw_avg), -sin(yaw_avg), 0],
            [sin(yaw_avg),  cos(yaw_avg), 0],
            [0,             0,            1]
        ])

        self.Ac = np.block([
            [np.zeros((3, 3)), np.zeros((3, 3)), np.eye(3),        np.zeros((3, 3))],
            [np.zeros((3, 3)), np.zeros((3, 3)), np.zeros((3, 3)), R_z.T           ],
            [np.zeros((3, 3)), np.zeros((3, 3)), np.zeros((3, 3)), np.zeros((3, 3))],
            [np.zeros((3, 3)), np.zeros((3, 3)), np.zeros((3, 3)), np.zeros((3, 3))],
        ])

        # 12x12 Matrix for 6D Wrenches (Biped)
        self.Bc = np.zeros((self.N, 12, 12))
        for i in range(self.N):

            r_L = self.r_l_foot_world[:, i]
            r_R = self.r_r_foot_world[:, i]

            skew_L = self._skew(r_L)
            skew_R = self._skew(r_R)

            I_inv = np.linalg.inv(I_com_world)

            # Columns: [F_left(3), Tau_left(3), F_right(3), Tau_right(3)]
            self.Bc[i] = np.block([
                [np.zeros((3,3)),   np.zeros((3,3)),   np.zeros((3,3)),   np.zeros((3,3))],
                [np.zeros((3,3)),   np.zeros((3,3)),   np.zeros((3,3)),   np.zeros((3,3))],
                # Linear Acceleration: Influenced ONLY by Forces
                [(1/m)*np.eye(3),   np.zeros((3,3)),   (1/m)*np.eye(3),   np.zeros((3,3))],
                # Angular Acceleration: Influenced by [r x F] AND explicit Torques
                [I_inv @ skew_L,    I_inv,             I_inv @ skew_R,    I_inv],
            ])

        # Gravity Vector
        self.gc = np.array([0,0,0, 0,0,0, 0,0,-9.81, 0,0,0])

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
        Ad[0:3, 6:9] = dt * np.eye(3)        
        Ad[3:6, 9:12] = dt * RzT             
        self.Ad = Ad

        # ---- gd ----
        g = np.array([0.0, 0.0, -9.81], dtype=float)
        gd = np.zeros((12, 1), dtype=float)
        gd[0:3, 0] = 0.5 * g * dt * dt       
        gd[6:9, 0] = g * dt                  
        self.gd = gd

        # ---- Bd ----
        Bd = np.zeros((N, 12, 12), dtype=float)

        B_F = (dt / m) * np.eye(3)        # Velocity change from Force
        B_Tau = np.zeros((3,3))           # Velocity change from Torque (Zero)

        Bp_F = 0.5 * dt * B_F             # Position change from Force
        Bp_Tau = np.zeros((3,3))          # Position change from Torque (Zero)

        for i in range(N):
            r_L = self.r_l_foot_world[:, i]
            r_R = self.r_r_foot_world[:, i]

            # Angular Velocity change from Forces (Lever arm)
            W_L_F = I_inv @ self._skew(r_L)
            W_R_F = I_inv @ self._skew(r_R)
            
            # Angular Velocity change from explicit Torques (Direct injection)
            W_L_Tau = I_inv
            W_R_Tau = I_inv

            Bi = Bd[i]

            # 1. Position block
            Bi[0:3, 0:3]   = Bp_F
            Bi[0:3, 3:6]   = Bp_Tau
            Bi[0:3, 6:9]   = Bp_F
            Bi[0:3, 9:12]  = Bp_Tau

            # 2. Velocity block
            Bi[6:9, 0:3]   = B_F
            Bi[6:9, 3:6]   = B_Tau
            Bi[6:9, 6:9]   = B_F
            Bi[6:9, 9:12]  = B_Tau

            # 3. Angular Velocity (Omega) block
            Bi[9:12, 0:3]  = dt * W_L_F
            Bi[9:12, 3:6]  = dt * W_L_Tau
            Bi[9:12, 6:9]  = dt * W_R_F
            Bi[9:12, 9:12] = dt * W_R_Tau

            # 4. RPY Orientation block
            Bi[3:6, 0:3]   = 0.5 * dt * dt * (RzT @ W_L_F)
            Bi[3:6, 3:6]   = 0.5 * dt * dt * (RzT @ W_L_Tau)
            Bi[3:6, 6:9]   = 0.5 * dt * dt * (RzT @ W_R_F)
            Bi[3:6, 9:12]  = 0.5 * dt * dt * (RzT @ W_R_Tau)

        self.Bd = Bd