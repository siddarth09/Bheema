import numpy as np
import pinocchio as pin

class WholeBodyController:
    """
    Computes joint torques to track MPC forces and swing foot trajectories.
    Utilizes Hierarchical Null-Space Projection to prevent task conflicts.
    """

    def __init__(self, g1_state_obj):
        self.state = g1_state_obj
        self.model = g1_state_obj.model
        self.data = g1_state_obj.data
        
        # Frame IDs
        self.lf_id = self.state.left_foot_frame
        self.rf_id = self.state.right_foot_frame

        # Gains
        self.Kp_swing = np.array([500.0, 500.0, 500.0]) 
        self.Kd_swing = np.array([100.0, 100.0, 100.0])
        self.Kp_joint = 15.0 
        self.Kd_joint = 1.0
        
        # --- Posture Setup ---
        # Get the default URDF posture (usually all zeros)
        self.q_nom = pin.neutral(self.model)[7:].copy()

        # Define the target bent-knee angles (in radians)
        # Note: Depending on your URDF's joint axes, you might need to flip the signs (+/-)
        bent_knee_angles = {
            'left_hip_pitch_joint': -0.3,   # Bend torso forward
            'left_knee_joint': 0.6,         # Bend knee backward
            'left_ankle_pitch_joint': -0.3, # Keep foot flat on ground
            
            'right_hip_pitch_joint': -0.3,
            'right_knee_joint': 0.6,
            'right_ankle_pitch_joint': -0.3
        }

        # Safely map these angles into q_nom using Pinocchio's joint IDs
        for joint_name, target_angle in bent_knee_angles.items():
            if self.model.existJointName(joint_name):
                # Get the joint ID
                joint_id = self.model.getJointId(joint_name)
                # Get the starting index of this joint in the FULL configuration vector (q)
                idx_q = self.model.joints[joint_id].idx_q
                
                # Because q_nom strips out the 7 floating base coordinates (pos + quat),
                # we must subtract 7 to find the correct index in q_nom.
                q_nom_idx = idx_q - 7 
                self.q_nom[q_nom_idx] = target_angle
            else:
                print(f"[WBC Warning] Joint '{joint_name}' not found in URDF. Check spelling.")


    def compute_torques(self, q, dq, f_mpc, swing_target=None):
        # --- 1. SAFETY CASTS ---
        q_safe = np.ascontiguousarray(q, dtype=np.float64)
        dq_safe = np.ascontiguousarray(dq, dtype=np.float64)
        
        # --- 2. Update Kinematics ---
        pin.computeJointJacobians(self.model, self.data, q_safe)
        pin.framesForwardKinematics(self.model, self.data, q_safe)
        
        # --- 3. PROPER Gravity Compensation ---
        # Computes the exact torques needed to hold the robot against gravity
        tau_g = pin.computeGeneralizedGravity(self.model, self.data, q_safe)

        # --- 4. Stance Forces & Jacobians ---
        J_L = pin.computeFrameJacobian(self.model, self.data, q_safe, self.lf_id, pin.LOCAL_WORLD_ALIGNED)
        J_R = pin.computeFrameJacobian(self.model, self.data, q_safe, self.rf_id, pin.LOCAL_WORLD_ALIGNED)
        
        J_L_lin = J_L[:3]
        J_R_lin = J_R[:3]
        
        f_L = np.ascontiguousarray(f_mpc[0:3], dtype=np.float64)
        f_R = np.ascontiguousarray(f_mpc[3:6], dtype=np.float64)
        
        # Task 1: Stance Control
        tau_stance = -(J_L_lin.T @ f_L + J_R_lin.T @ f_R)

        # --- 5. Compute Contact Null-Space Projector (N_c) ---
        # Determine which feet are actually on the ground based on MPC forces
        J_c_list = []
        if f_L[2] > 10.0: # If left Fz > 10N, it's in contact
            J_c_list.append(J_L_lin)
        if f_R[2] > 10.0: # If right Fz > 10N, it's in contact
            J_c_list.append(J_R_lin)
            
        if len(J_c_list) > 0:
            J_c = np.vstack(J_c_list)
            # Damped pseudo-inverse for stability near singularities
            J_c_pinv = J_c.T @ np.linalg.inv(J_c @ J_c.T + 1e-4 * np.eye(J_c.shape[0]))
            N_c = np.eye(self.model.nv) - J_c_pinv @ J_c
        else:
            N_c = np.eye(self.model.nv)

        # --- 6. Swing Leg Control (Projected) ---
        tau_swing = np.zeros(self.model.nv)
        if swing_target is not None:
            is_left = swing_target['is_left']
            pos_des = swing_target['pos']
            vel_des = swing_target['vel']
            
            fid = self.lf_id if is_left else self.rf_id
            J_sw = J_L_lin if is_left else J_R_lin
            
            curr_pos = self.data.oMf[fid].translation
            v_frame = pin.getFrameVelocity(self.model, self.data, fid, pin.LOCAL_WORLD_ALIGNED)
            curr_vel = v_frame.linear
            
            # Virtual force for the swing foot
            f_virt = self.Kp_swing * (pos_des - curr_pos) + self.Kd_swing * (vel_des - curr_vel)
            
            # Project swing torques so swinging the leg doesn't disrupt the stance foot
            raw_swing_tau = J_sw.T @ f_virt
            tau_swing = N_c @ raw_swing_tau

        # --- 7. Posture Control (Projected) ---
        q_joints = q_safe[7:]
        dq_joints = dq_safe[6:]
        
        raw_posture_tau = np.zeros(self.model.nv)
        raw_posture_tau[6:] = self.Kp_joint * (self.q_nom - q_joints) - self.Kd_joint * dq_joints

        # Project posture into the null space so it ONLY acts on free joints
        tau_posture = N_c @ raw_posture_tau

        # --- 8. Final Command ---
        return tau_g + tau_stance + tau_swing + tau_posture