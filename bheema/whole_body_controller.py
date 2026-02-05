# whole_body_controller.py
import numpy as np
import pinocchio as pin

class WholeBodyController:
    """
    Whole Body Controller (position-output) for BHEEMA.
    """

    def __init__(self, state):
        self.model = state.model
        self.data = state.data
        
        # Nominal data for reference generation
        self.nom_data = self.model.createData()
        
        self.nq = self.model.nq
        self.nv = self.model.nv

        # Frame IDs
        self.pelvis_frame = self.model.getFrameId("pelvis")
        self.left_foot_frame = self.model.getFrameId("left_foot")
        self.right_foot_frame = self.model.getFrameId("right_foot")

        # Weights
        self.w_foot = 10.0
        self.w_pelvis_ori = 5.0
        self.w_pelvis_height = 5.0
        self.w_posture = 0.1

        self.kp = 1.0  # Gain

    def damped_pinv(self, J, damping=1e-4):
        """
        Damped Pseudo-Inverse (Levenberg-Marquardt)
        """
        rows, cols = J.shape
        if rows <= cols: # Fat matrix
            return J.T @ np.linalg.inv(J @ J.T + damping * np.eye(rows))
        else: # Tall matrix
            return np.linalg.inv(J.T @ J + damping * np.eye(cols)) @ J.T

    def compute_joint_targets(self, state, f_mpc, contact_table, swing_trajs, q_nominal):
        """
        Hierarchical Least Squares (Task Stacking)
        """
        q, _ = state.get_generalized_state()

        # 1. Update Kinematics
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)

        pin.forwardKinematics(self.model, self.nom_data, q_nominal)
        pin.updateFramePlacements(self.model, self.nom_data)

        # 2. Define Tasks Lists
        tasks_p1 = [] # Priority 1 (Hard)
        tasks_p2 = [] # Priority 2 (Soft)
        tasks_p3 = [] # Priority 3 (Posture)

        # --- P1: Swing Foot ---
        for foot, frame_id, idx in [("left", self.left_foot_frame, 0), ("right", self.right_foot_frame, 1)]:
            if contact_table[idx, 0] == 0 and foot in swing_trajs:
                p_des = swing_trajs[foot]["pos"]
                p_cur = self.data.oMf[frame_id].translation
                
                J6 = pin.computeFrameJacobian(self.model, self.data, q, frame_id, pin.ReferenceFrame.WORLD)
                J = J6[:3, :]
                e = p_des - p_cur
                tasks_p1.append((np.sqrt(self.w_foot) * J, np.sqrt(self.w_foot) * e))

        # --- P1: Pelvis Orientation ---
        R_des = self.nom_data.oMf[self.pelvis_frame].rotation
        R_cur = self.data.oMf[self.pelvis_frame].rotation
        e_ori = pin.log3(R_des @ R_cur.T)
        
        J6 = pin.computeFrameJacobian(self.model, self.data, q, self.pelvis_frame, pin.ReferenceFrame.WORLD)
        J_ori = J6[3:6, :]
        tasks_p1.append((np.sqrt(self.w_pelvis_ori) * J_ori, np.sqrt(self.w_pelvis_ori) * e_ori))

        # --- P2: Pelvis Height ---
        z_des = self.nom_data.oMf[self.pelvis_frame].translation[2]
        z_cur = self.data.oMf[self.pelvis_frame].translation[2]
        e_z = np.array([z_des - z_cur])
        J_z = J6[2:3, :]
        tasks_p2.append((np.sqrt(self.w_pelvis_height) * J_z, np.sqrt(self.w_pelvis_height) * e_z))

        # --- P3: Posture ---
        J_posture = np.eye(self.nv)
        e_posture = pin.difference(self.model, q, q_nominal)
        tasks_p3.append((np.sqrt(self.w_posture) * J_posture, np.sqrt(self.w_posture) * e_posture))

        # 3. Solve Hierarchy
        dq_cmd = np.zeros(self.nv)
        N = np.eye(self.nv)

        # Loop through priority levels
        for priority_tasks in [tasks_p1, tasks_p2, tasks_p3]:
            if not priority_tasks:
                continue
                
            
            Js, es = zip(*priority_tasks)
            J_stack = np.vstack(Js)
            e_stack = np.concatenate(es)

            # Project into Nullspace of previous priorities
            J_eff = J_stack @ N
            
            # Solve
            J_pinv = self.damped_pinv(J_eff, damping=1e-4)
            dq_task = J_pinv @ (e_stack - J_stack @ dq_cmd)
            
            # Accumulate
            dq_cmd += dq_task
            
            # Update Nullspace
            N = N @ (np.eye(self.nv) - J_pinv @ J_eff)

        # 4. Integrate
        # dt = 0.01 (WBC rate 100hz approx)
        dt = 0.01 
        q_des = pin.integrate(self.model, q, self.kp * dq_cmd * dt)
        
        return q_des