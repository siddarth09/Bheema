import pinocchio as pin
import numpy as np
from pathlib import Path
from numpy import cos, sin

# --------------------------------------------------------------------------------
# Model Setting
# --------------------------------------------------------------------------------

# Update this path to where your URDF is stored
URDF_PATH = "/home/sid/projects25/src/bheema/unitree_g1/g1_with_hands.urdf"

class ConfigurationState:
    def __init__(self):
        # 1. Lower the initial base height to match the bent knees
        self.base_pos = np.array([0.0, 0.0, 0.68]) 
        self.base_quad = np.array([0.0, 0.0, 0.0, 1.0])

        # 2. Leg joint angles [hip_p, hip_r, hip_y, knee, ankle_p, ankle_r]
        # Hip pitches back, knee bends forward, ankle pitches back to keep foot flat
        bent_leg = np.array([-0.2, 0.0, 0.0, 0.4, -0.2, 0.0])
        
        self.left_leg_angle = bent_leg.copy()
        self.right_leg_angle = bent_leg.copy()

        # Initial generalized velocities (Keep these as zeros)
        self.base_vel = np.zeros(3)
        self.base_ang_vel = np.zeros(3)
        self.left_leg_vel = np.zeros(6)
        self.right_leg_vel = np.zeros(6)

    def compute_euler_angle_world(self):
        # 1) raw roll, pitch, yaw in [-pi, pi]
        qx, qy, qz, qw = self.base_quad
        q_eig = pin.Quaternion(qw, qx, qy, qz)
        R = q_eig.toRotationMatrix()
        rpy = pin.rpy.matrixToRpy(R) # Returns Euler ZYX
        roll, pitch, yaw_meas = np.array(rpy).reshape(3,)

        # 2) initialize unwrap state on first call
        if not hasattr(self, "_yaw_unwrap_initialized"):
            self._yaw_unwrap_initialized = True
            self._yaw_prev_meas = yaw_meas
            self._yaw_cont = yaw_meas
        else:
            # 3) unwrap: keep smallest change between steps
            yaw_delta = (yaw_meas - self._yaw_prev_meas + np.pi) % (2 * np.pi) - np.pi
            self._yaw_cont += yaw_delta
            self._yaw_prev_meas = yaw_meas

        # 4) return roll, pitch, continuous yaw
        return np.array([roll, pitch, self._yaw_cont])


class PinG1Model:
    def __init__(self, urdf_path=URDF_PATH):
        # Build robot (free-flyer)
        self.model = pin.buildModelFromUrdf(str(urdf_path), pin.JointModelFreeFlyer())
        self.data = self.model.createData()

        self.current_config = ConfigurationState()

        # Cache Frame IDs
        self.base_id = self.model.getFrameId("pelvis")
        self.left_foot_id = self.model.getFrameId("left_ankle_roll_link")
        self.right_foot_id = self.model.getFrameId("right_ankle_roll_link")

        # Get Joint Indices to map our 12 leg variables to the 36-DOF robot
        self.left_joint_names = [
            "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
            "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint"
        ]
        self.right_joint_names = [
            "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
            "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint"
        ]

        self.idx_q_L = [self.model.joints[self.model.getJointId(n)].idx_q for n in self.left_joint_names]
        self.idx_v_L = [self.model.joints[self.model.getJointId(n)].idx_v for n in self.left_joint_names]
        self.idx_q_R = [self.model.joints[self.model.getJointId(n)].idx_q for n in self.right_joint_names]
        self.idx_v_R = [self.model.joints[self.model.getJointId(n)].idx_v for n in self.right_joint_names]

        # Initial forward kinematics
        self.update_model()

    def get_full_q_dq(self):
        """Constructs the full nv/nq vectors, keeping arms at 0 (neutral)"""
        q = pin.neutral(self.model)
        dq = np.zeros(self.model.nv)

        # Base
        q[0:3] = self.current_config.base_pos
        q[3:7] = self.current_config.base_quad
        dq[0:3] = self.current_config.base_vel
        dq[3:6] = self.current_config.base_ang_vel

        # Legs
        for i in range(6):
            q[self.idx_q_L[i]] = self.current_config.left_leg_angle[i]
            q[self.idx_q_R[i]] = self.current_config.right_leg_angle[i]
            dq[self.idx_v_L[i]] = self.current_config.left_leg_vel[i]
            dq[self.idx_v_R[i]] = self.current_config.right_leg_vel[i]

        return q, dq

    def update_model(self):
        q, dq = self.get_full_q_dq()
        
        pin.forwardKinematics(self.model, self.data, q, dq)
        pin.updateFramePlacements(self.model, self.data) 
        pin.computeAllTerms(self.model, self.data, q, dq)
        pin.computeJointJacobians(self.model, self.data, q)
        pin.computeJointJacobiansTimeVariation(self.model, self.data, q, dq)
        pin.ccrba(self.model, self.data, q, dq)
        pin.centerOfMass(self.model, self.data, q, dq)

        self.pos_com_world = self.data.com[0]
        self.vel_com_world = self.data.vcom[0]

        # Base Rotation Matrix
        oMb = self.data.oMf[self.base_id]
        yaw = self.current_config.compute_euler_angle_world()[2]
        self.R_body_to_world = np.array(oMb.rotation)
        self.R_world_to_body = self.R_body_to_world.T

        self.R_z = np.array([
            [cos(yaw), -sin(yaw), 0],
            [sin(yaw),  cos(yaw), 0],
            [0,         0,        1]
        ])

    def compute_com_x_vec(self):
        """Returns the 6-DOF 12 states centroidal x-vector"""
        pos_com_world = self.pos_com_world
        rpy_com_world = self.current_config.compute_euler_angle_world()
        vel_com_world = self.vel_com_world
        rpy_rate_body = self.current_config.base_ang_vel
        
        omega_world = self.R_body_to_world @ rpy_rate_body

        x_vec = np.concatenate([pos_com_world, rpy_com_world, vel_com_world, omega_world])
        return x_vec.reshape(-1, 1)

    def get_foot_placement_in_world(self):
        oMf_L = self.data.oMf[self.left_foot_id]
        oMf_R = self.data.oMf[self.right_foot_id]
        return oMf_L.translation.copy(), oMf_R.translation.copy()
    
    def get_foot_lever_world(self):
        pos_L, pos_R = self.get_foot_placement_in_world()
        return pos_L - self.pos_com_world, pos_R - self.pos_com_world
    
    def compute_leg_Jacobian_world(self, leg: str):
        """Returns a 3x6 translational Jacobian for the 6 leg joints"""
        foot_id = self.left_foot_id if leg.lower() == "left" else self.right_foot_id
        
        # 6xnv full spatial Jacobian
        J_world = pin.getFrameJacobian(self.model, self.data, foot_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        J_pos_world = J_world[0:3,:] # 3xnv translational

        vcols = self.idx_v_L if leg.lower() == "left" else self.idx_v_R
        J_leg_pos_world = J_pos_world[:, vcols] # Extract the 3x6 block

        return J_leg_pos_world

    def compute_Jdot_dq_world(self, leg: str):
        foot_id = self.left_foot_id if leg.lower() == "left" else self.right_foot_id

        Jdot = pin.getFrameJacobianTimeVariation(
            self.model, self.data, foot_id,
            pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
        )
        
        _, dq = self.get_full_q_dq()
        Jdot_dq = Jdot[0:3, :] @ dq
        return np.asarray(Jdot_dq).reshape(3,)

    def compute_dynamics_terms(self):
        return self.data.g, self.data.C, self.data.M