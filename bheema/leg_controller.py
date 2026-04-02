import numpy as np
import pinocchio as pin
from .g1_config import PinG1Model
from .gait import Gait
from dataclasses import dataclass

# --------------------------------------------------------------------------------
# Leg Controller Setting (BIPED)
# --------------------------------------------------------------------------------

KP_SWING = np.diag([2500, 3500, 1000, 400, 400, 400]) 
KD_SWING = np.diag([120, 120, 120, 10, 10, 10]) 

# Mapping from leg name to index in the 2-element biped mask
LEG_INDEX = {
    "LEFT": 0,
    "RIGHT": 1,
}

# Mapping from leg name to the joint torque slice in the 49-DOF (C*dq + g) vector
# Base = 0:6. Left Leg = 6:12. Right Leg = 12:18.
JOINT_SLICES = {
    "LEFT": slice(6, 12),
    "RIGHT": slice(12, 18),
}

@dataclass
class LegOutput:
    tau: np.ndarray       # shape (6,) - 6 joints per leg
    pos_des: np.ndarray   # shape (3,)
    pos_now: np.ndarray   # shape (3,)
    vel_des: np.ndarray   # shape (3,)
    vel_now: np.ndarray   # shape (3,)

class LegController():
        
    def __init__(self):
        # Biped uses a 2-element mask
        self.last_mask = np.array([2, 2])

    def compute_leg_torque(
        self,
        leg: str,
        g1: PinG1Model,
        gait: Gait,
        contact_wrench: np.ndarray, # (6,) [Fx, Fy, Fz, Tx, Ty, Tz]
        current_time: float,
    ):
        # 1. Extract Parameters
        leg_idx = LEG_INDEX[leg.upper()]
        joint_slice = JOINT_SLICES[leg.upper()]
        foot_id = g1.left_foot_id if leg_idx == 0 else g1.right_foot_id

        # Jacobians
        J_foot_world = g1.compute_leg_Jacobian_world(leg) # (6x6)
        
        # Pinocchio returns the full 6x49 Spatial Jacobian natively
        J_full_foot_world = pin.getFrameJacobian(
            g1.model, g1.data, foot_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
        ) # (6x49)

        # Dynamics Terms (g: 49x1, C: 49x49, M: 49x49)
        g, C, M = g1.compute_dynamics_terms()
        dq = g1.current_config.get_dq()

        current_mask = gait.compute_current_mask(current_time)
        tau_cmd = np.zeros(6)

        # Initialize desired to current (using 6D extraction but storing 3D for logging)
        foot_pos_now, foot_vel_now = g1.get_single_foot_state_in_world(leg)
        foot_pos_des = foot_pos_now.copy()
        foot_vel_des = foot_vel_now.copy()

        oMf_now = g1.data.oMf[foot_id]
        spatial_vel_now = pin.getFrameVelocity(g1.model, g1.data, foot_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED).vector

        # 2. Detect takeoff transition
        if self.last_mask[leg_idx] != current_mask[leg_idx] and current_mask[leg_idx] == 0:
            setattr(self, f"{leg}_takeoff_time", current_time)
            traj, td_pos = gait.compute_swing_traj_and_touchdown(g1, leg)
            setattr(self, f"{leg}_traj", traj)
            setattr(self, f"{leg}_td_pos", td_pos)

        # 3. Swing vs stance
        if current_mask[leg_idx] == 0:  # Swing phase
            takeoff_time = getattr(self, f"{leg}_takeoff_time")
            traj = getattr(self, f"{leg}_traj")

            time_since_takeoff = current_time - takeoff_time
            foot_pos_des, foot_vel_des_3d, foot_acl_des_3d = traj(time_since_takeoff)
            
            # --- 6D ERROR COMPUTATION ---
            # Position Error (3D)
            pos_error = foot_pos_des - foot_pos_now
            
            # Orientation Error (3D) -> Keep foot flat with ground
            R_now = oMf_now.rotation
            R_des = g1.R_z 
            ori_error_local = pin.log3(R_now.T @ R_des)
            ori_error_world = R_now @ ori_error_local
            
            # Combine to 6D Spatial Error
            spatial_error = np.concatenate([pos_error, ori_error_world])
            
            # Velocity Error (6D) -> Desired angular velocity is 0
            spatial_vel_des = np.concatenate([foot_vel_des_3d, np.zeros(3)]) 
            spatial_vel_error = spatial_vel_des - spatial_vel_now

            # Acceleration Desired (6D) -> Desired angular accel is 0
            spatial_acl_des = np.concatenate([foot_acl_des_3d, np.zeros(3)])

            # --- OPERATIONAL SPACE DYNAMICS (Go2 Exact Emulation) ---
            # Lambda: Cartesian Mass Matrix (6x6) computed from full 49x49 M matrix
            Lambda = np.linalg.pinv(
                J_full_foot_world @ np.linalg.inv(M) @ J_full_foot_world.T
            )
            
            # Jdot_dq (6x1)
            Jdot_dq = g1.compute_Jdot_dq_world(leg)

            # Feedforward term (6x1)
            f_ff = Lambda @ (spatial_acl_des - Jdot_dq)

            # PD + feedforward in 6D Cartesian space
            force_6d = KP_SWING @ spatial_error + KD_SWING @ spatial_vel_error + f_ff 

            # Map to joint torques + add (C*dq + g) leg segment slice
            tau_cmd = J_foot_world.T @ force_6d + (C @ dq + g)[joint_slice]

        else:  # Stance phase
            # Feedforward from MPC
            tau_ff = J_foot_world.T @ -contact_wrench + (C @ dq + g)[joint_slice]
            
            # Joint PD feedback to resist drift
            q_leg = g1.current_config.left_leg_angle if leg_idx == 0 else g1.current_config.right_leg_angle
            dq_leg = g1.current_config.left_leg_vel if leg_idx == 0 else g1.current_config.right_leg_vel
            q_des = np.array([-0.3, 0.0, 0.0, 0.6, -0.3, 0.0])
            
            Kp_stance = 150.0
            Kd_stance = 20.0
            tau_pd = Kp_stance * (q_des - q_leg) - Kd_stance * dq_leg
            
            tau_cmd = tau_ff + tau_pd
        # Update mask memory
        self.last_mask[leg_idx] = current_mask[leg_idx]

        return LegOutput(
            tau=tau_cmd,
            pos_des=foot_pos_des,
            pos_now=foot_pos_now,
            vel_des=foot_vel_des_3d if 'foot_vel_des_3d' in locals() else foot_vel_now,
            vel_now=foot_vel_now,
        )