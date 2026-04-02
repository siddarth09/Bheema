import numpy as np
from bheema.g1_config import PinG1Model
from numpy import cos, sin

# --------------------------------------------------------------------------------
# Gait Setting (BIPED)
# --------------------------------------------------------------------------------

PHASE_OFFSET = np.array([0.0, 0.5]).reshape(2)    
HEIGHT_SWING = 0.18
NOMINAL_STANCE_WIDTH = 0.25
MIN_FOOT_GAP = 0.15

class Gait():
    def __init__(self, frequency_hz, duty):
        self.gait_duty = duty
        self.gait_hz = frequency_hz
        self.NOMINAL_STANCE_WIDTH = NOMINAL_STANCE_WIDTH
        self.gait_period = 1.0 / frequency_hz 
        self.stance_time = self.gait_duty * self.gait_period
        self.swing_time = (1.0 - self.gait_duty) * self.gait_period

    def compute_current_mask(self, time):
        mask = self.compute_contact_table(time, 0, 1)
        return mask.flatten() # Returns [Left, Right]
    
    def compute_contact_table(self, t0: float, dt: float, N: int) -> np.ndarray:
        t = t0 + np.arange(N) * dt
        t = t + dt/2.0

        phases = np.mod(PHASE_OFFSET[:, None] + t[None, :] / self.gait_period, 1.0)
        contact_table = (phases < self.gait_duty).astype(np.int32)
        return contact_table        
    
    def compute_touchdown_world_for_traj_purpose_only(self, g1: PinG1Model, leg: str):
        base_pos = g1.current_config.base_pos
        base_vel = g1.current_config.base_vel
        R_z = g1.R_z
        yaw_rate = getattr(g1, 'yaw_rate_des_world', 0.0) 

        # Enforce Minimum Gap mathematically
        half_gap = MIN_FOOT_GAP / 2.0
        if leg.lower() == "left":
            lateral_offset = max(NOMINAL_STANCE_WIDTH / 2.0, half_gap)
        else:
            lateral_offset = min(-NOMINAL_STANCE_WIDTH / 2.0, -half_gap)
            
        hip_offset = np.array([0.0, lateral_offset, 0.0])
        body_pos = np.array([base_pos[0], base_pos[1], 0.0])
        hip_pos_world = body_pos + R_z @ hip_offset

        t_swing = self.swing_time
        t_stance = self.stance_time

        T = t_swing + 0.5 * t_stance
        pred_time = T / 2.0

        pos_norminal_term = [hip_pos_world[0], hip_pos_world[1], 0.0]
        pos_drift_term = [base_vel[0] * pred_time, base_vel[1] * pred_time, 0.0]

        dtheta = yaw_rate * pred_time
        center_xy = np.array([base_pos[0], base_pos[1]])  
        r_xy = np.array([pos_norminal_term[0], pos_norminal_term[1]]) - center_xy

        rotation_correction_term = np.array([
            -dtheta * r_xy[1],
             dtheta * r_xy[0],
             0.0
        ])
        
        pos_touchdown_world = (np.array(pos_norminal_term)
                                + np.array(pos_drift_term)
                                + np.array(rotation_correction_term))

        return pos_touchdown_world
    
    def compute_swing_traj_and_touchdown(self, g1: PinG1Model, leg: str):
        base_pos = g1.current_config.base_pos
        pos_com_world = g1.pos_com_world
        vel_com_world = g1.vel_com_world
        R_z = g1.R_z
        yaw_rate = getattr(g1, 'yaw_rate_des_world', 0.0)

        # Use the new unified function from g1_config.py
        pos_foot_L, pos_foot_R = g1.get_foot_placement_in_world()
        foot_pos = pos_foot_L if leg.lower() == "left" else pos_foot_R

        # FIXED: Order of Operations. Calculate offset first, then hip_pos_world
        half_gap = MIN_FOOT_GAP / 2.0
        if leg.lower() == "left":
            lateral_offset = max(NOMINAL_STANCE_WIDTH / 2.0, half_gap)
        else:
            lateral_offset = min(-NOMINAL_STANCE_WIDTH / 2.0, -half_gap)
            
        hip_offset = np.array([0.0, lateral_offset, 0.0])
        body_pos = np.array([base_pos[0], base_pos[1], 0.0])
        hip_pos_world = body_pos + R_z @ hip_offset

        x_vel_des = getattr(g1, 'x_vel_des_world', 0.0)
        y_vel_des = getattr(g1, 'y_vel_des_world', 0.0)
        x_pos_des = getattr(g1, 'x_pos_des_world', pos_com_world[0])
        y_pos_des = getattr(g1, 'y_pos_des_world', pos_com_world[1])

        t_swing = self.swing_time
        t_stance = self.stance_time
        T = t_swing + 0.5 * t_stance
        pred_time = T / 2.0

        # FIXED: Go2-style Raibert Heuristic Gains (Position + Velocity)
        k_v_x = 1.1* T          
        k_p_x = 0.3             
        k_v_y = 0.5 * T          
        k_p_y = 0.05

        pos_norminal_term = [hip_pos_world[0], hip_pos_world[1], 0.0]
        pos_drift_term = [x_vel_des * pred_time, y_vel_des * pred_time, 0.0]
        
        pos_correction_term = [
            k_p_x * (pos_com_world[0] - x_pos_des), 
            k_p_y * (pos_com_world[1] - y_pos_des), 
            0.0
        ]
        
        vel_correction_term = [
            k_v_x * (vel_com_world[0] - x_vel_des), 
            k_v_y * (vel_com_world[1] - y_vel_des), 
            0.0
        ]
                    
        dtheta = yaw_rate * pred_time
        center_xy = np.array([base_pos[0], base_pos[1]])  
        r_xy = np.array([pos_norminal_term[0], pos_norminal_term[1]]) - center_xy

        rotation_correction_term = np.array([
            -dtheta * r_xy[1],
             dtheta * r_xy[0],
             0.0
        ])
    
        pos_touchdown_world = (np.array(pos_norminal_term)
                                + np.array(pos_drift_term)
                                + np.array(pos_correction_term)
                                + np.array(vel_correction_term)
                                + np.array(rotation_correction_term))
        
        foot_pos[2] = 0.0 
        
        pos_foot_traj_eval_at_world = self.make_swing_trajectory(foot_pos, pos_touchdown_world, t_swing, h_sw=HEIGHT_SWING)
        return pos_foot_traj_eval_at_world, pos_touchdown_world

    def make_swing_trajectory(self, p0, pf, t_swing, h_sw):
        p0 = np.asarray(p0, dtype=float)
        pf = np.asarray(pf, dtype=float)
        T = float(t_swing)
        dp = pf - p0

        def eval_at(t):
            s = np.clip(t / T, 0.0, 1.0)

            mj   = 10*s**3 - 15*s**4 + 6*s**5
            dmj  = 30*s**2 - 60*s**3 + 30*s**4           
            d2mj = 60*s    - 180*s**2 + 120*s**3         

            p = p0 + dp * mj
            v = (dp * dmj) / T
            a = (dp * d2mj) / (T**2)

            if h_sw != 0.0:
                b    = 64 * s**3 * (1 - s)**3
                db   = 192 * s**2 * (1 - s)**2 * (1 - 2*s)            
                d2b  = 192 * ( 2*s*(1 - s)**2*(1 - 2*s)
                            - 2*s**2*(1 - s)*(1 - 2*s)
                            - 2*s**2*(1 - s)**2 )                  

                p[2] += h_sw * b
                v[2] += h_sw * db / T
                a[2] += h_sw * d2b / (T**2)

            return p, v, a

        return eval_at
    

    # ==============================================================================
# DEBUG: Plot the Gait & Raibert Foot Trajectory
# ==============================================================================
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    print("Running Gait & Raibert Heuristic Debugger...")

    # 1. Initialize Dummy Robot and Gait
    g1 = PinG1Model()
    gait = Gait(frequency_hz=1.0, duty=0.68)

    # 2. Inject a dummy velocity state to trigger the Raibert Heuristic
    # Let's pretend the robot is moving forward at 0.5 m/s, but getting pushed slightly right (-0.1 m/s)
    g1.current_config.base_vel = np.array([0.5, -0.1, 0.0]) 
    g1.vel_com_world = np.array([0.5, -0.1, 0.0])
    
    # We WANT to be moving straight forward at 0.5 m/s
    g1.x_vel_des_world = 0.5
    g1.y_vel_des_world = 0.0
    
    # 3. Generate the Swing Trajectory for the LEFT leg
    print(f"Swing Time: {gait.swing_time:.3f} s")
    traj_func, td_pos = gait.compute_swing_traj_and_touchdown(g1, "LEFT")
    print(f"Calculated Touchdown Position (World): X={td_pos[0]:.3f}, Y={td_pos[1]:.3f}, Z={td_pos[2]:.3f}")

    # 4. Sample the trajectory over the swing duration
    dt = 0.01
    t_vec = np.arange(0, gait.swing_time + dt, dt)
    pos_log = np.zeros((3, len(t_vec)))
    vel_log = np.zeros((3, len(t_vec)))
    acc_log = np.zeros((3, len(t_vec)))

    for i, t in enumerate(t_vec):
        p, v, a = traj_func(t)
        pos_log[:, i] = p
        vel_log[:, i] = v
        acc_log[:, i] = a

    # 5. Create the Plot Layout
    fig = plt.figure(figsize=(14, 10))

    # Plot A: 3D Foot Arc
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    ax1.plot(pos_log[0, :], pos_log[1, :], pos_log[2, :], label='Foot Swing Arc', color='blue', linewidth=2)
    ax1.scatter(pos_log[0, 0], pos_log[1, 0], pos_log[2, 0], color='green', s=100, label='Takeoff (Start)')
    ax1.scatter(pos_log[0, -1], pos_log[1, -1], pos_log[2, -1], color='red', s=100, label='Touchdown (Target)')
    
    # Draw a line for the ground plane
    ax1.plot([pos_log[0, 0], pos_log[0, -1]], [pos_log[1, 0], pos_log[1, -1]], [0, 0], color='gray', linestyle='--')
    
    ax1.set_xlabel('X (m) - Forward')
    ax1.set_ylabel('Y (m) - Lateral')
    ax1.set_zlabel('Z (m) - Height')
    ax1.set_title('3D Raibert Swing Trajectory')
    ax1.legend()

    # Plot B: Height Clearance (Z vs Time)
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(t_vec, pos_log[2, :], color='green', linewidth=2)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Foot Height Z (m)')
    ax2.set_title(f'Swing Clearance (Max = {np.max(pos_log[2, :]):.3f}m)')
    ax2.grid(True)

    # Plot C: Ground Progression (X/Y vs Time)
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.plot(t_vec, pos_log[0, :], label='X (Forward Step)', color='blue', linewidth=2)
    ax3.plot(t_vec, pos_log[1, :], label='Y (Lateral Step)', color='orange', linewidth=2)
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Position (m)')
    ax3.set_title('Ground Plane Progression')
    ax3.grid(True)
    ax3.legend()

    # Plot D: Minimum Jerk Velocity Profile
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.plot(t_vec, vel_log[0, :], label='Vel X (Forward)', color='blue')
    ax4.plot(t_vec, vel_log[1, :], label='Vel Y (Lateral)', color='orange')
    ax4.plot(t_vec, vel_log[2, :], label='Vel Z (Vertical)', color='green')
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Velocity (m/s)')
    ax4.set_title('Cartesian Velocity Profile (Should start/end at 0)')
    ax4.grid(True)
    ax4.legend()

    plt.tight_layout()
    plt.show()