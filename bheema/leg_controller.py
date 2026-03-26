import numpy as np
import pinocchio as pin
from .g1_config import PinG1Model
from .gait import Gait
from dataclasses import dataclass

# --------------------------------------------------------------------------------
# Leg Controller Setting (BIPED)
# --------------------------------------------------------------------------------

# 6D PD Gains for Swing Phase: [x, y, z, roll, pitch, yaw]
KP_SWING = np.diag([2500, 2500, 2000, 400, 400, 400]) 
KD_SWING = np.diag([60, 60, 40, 10, 10, 10]) 
# Mapping from leg name to index in the mask
LEG_INDEX = {
    "LEFT": 0,
    "RIGHT": 1,
}

@dataclass
class LegOutput:
    tau: np.ndarray       # shape (6,) - 6 joints per leg
    pos_des: np.ndarray   # shape (3,) - Just storing translations for logging
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
        contact_wrench: np.ndarray, # (6,) [Fx, Fy, Fz, Tx, Ty, Tz] from MPC
        current_time: float,
    ):
        # 1. Extract Identifiers
        leg_idx = LEG_INDEX[leg.upper()]
        foot_id = g1.left_foot_id if leg_idx == 0 else g1.right_foot_id
        vcols = g1.idx_v_L if leg_idx == 0 else g1.idx_v_R

        # 2. Get Kinematics & Dynamics from Pinocchio
        _, dq = g1.get_full_q_dq()
        
        # 6xnv Full Spatial Jacobian (LOCAL_WORLD_ALIGNED)
        J_full = pin.getFrameJacobian(g1.model, g1.data, foot_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        J_leg = J_full[:, vcols] # 6x6 matrix mapping leg joint vels to foot spatial vel

        # Mass Matrix and Non-Linear Effects (Coriolis + Gravity)
        M = g1.data.M
        nle = g1.data.nle # This is exactly (C*dq + g)

        current_mask = gait.compute_current_mask(current_time)
        tau_cmd = np.zeros(6)

        # Current Foot State
        oMf_now = g1.data.oMf[foot_id]
        foot_pos_now = oMf_now.translation.copy()
        
        spatial_vel_now = pin.getFrameVelocity(g1.model, g1.data, foot_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED).vector
        foot_vel_now = spatial_vel_now[0:3]

        foot_pos_des = foot_pos_now.copy()
        foot_vel_des = foot_vel_now.copy()

        # ---------------------------------------------------------
        # Detect Takeoff Transition
        # ---------------------------------------------------------
        if self.last_mask[leg_idx] != current_mask[leg_idx] and current_mask[leg_idx] == 0:
            setattr(self, f"{leg}_takeoff_time", current_time)
            traj, td_pos = gait.compute_swing_traj_and_touchdown(g1, leg)
            setattr(self, f"{leg}_traj", traj)
            setattr(self, f"{leg}_td_pos", td_pos)

        # ---------------------------------------------------------
        # SWING PHASE 
        # ---------------------------------------------------------
        if current_mask[leg_idx] == 0:  
            takeoff_time = getattr(self, f"{leg}_takeoff_time")
            traj = getattr(self, f"{leg}_traj")
            time_since_takeoff = current_time - takeoff_time
            
            # Get 3D Trajectory targets
            foot_pos_des, foot_vel_des, foot_acl_des = traj(time_since_takeoff)

            # 1. Positional Error (3D)
            pos_error = foot_pos_des - foot_pos_now

            # 2. Orientation Error (3D)
            # We want the foot to land flat, aligned with the robot's yaw
            R_now = oMf_now.rotation
            R_des = g1.R_z 
            # Log3 maps the rotation matrix difference to a 3D angular error vector
            ori_error_local = pin.log3(R_now.T @ R_des)
            ori_error_world = R_now @ ori_error_local

            # Combine into 6D Spatial Error
            spatial_error = np.concatenate([pos_error, ori_error_world])
            
            # 3. Velocity Error (6D)
            spatial_vel_des = np.concatenate([foot_vel_des, np.zeros(3)]) # Zero angular velocity desired
            spatial_vel_error = spatial_vel_des - spatial_vel_now

            # 4. Feedforward Acceleration (6D)
            spatial_acl_des = np.concatenate([foot_acl_des, np.zeros(3)])

            # 5. Operational Space Mass Matrix (Lambda) -> (6x6)
            Lambda = np.linalg.pinv(J_full @ np.linalg.inv(M) @ J_full.T)
            
            # Bias Acceleration (Jdot * dq) -> (6,)
            Jdot = pin.getFrameJacobianTimeVariation(g1.model, g1.data, foot_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
            Jdot_dq = Jdot @ dq

            # 6. Compute 6D Virtual Force
            f_ff = Lambda @ (spatial_acl_des - Jdot_dq)
            force_6d = KP_SWING @ spatial_error + KD_SWING @ spatial_vel_error + f_ff

            # 7. Map to Joint Torques + Add Coriolis/Gravity
            tau_cmd = J_leg.T @ force_6d + nle[vcols]

        # ---------------------------------------------------------
        # STANCE PHASE (Jacobian Transpose Control)
        # ---------------------------------------------------------
        else:  
            # tau = J^T * (-Wrench_from_environment)
            # We add nle (gravity compensation) so the MPC only needs to worry about external forces
            tau_cmd = J_leg.T @ -contact_wrench

        # Update mask memory
        self.last_mask[leg_idx] = current_mask[leg_idx]

        return LegOutput(
            tau=tau_cmd,
            pos_des=foot_pos_des,
            pos_now=foot_pos_now,
            vel_des=foot_vel_des,
            vel_now=foot_vel_now,
        )