import pinocchio as pin
from pinocchio.robot_wrapper import RobotWrapper
import numpy as np
from pathlib import Path
from numpy import cos, sin

# --------------------------------------------------------------------------------
# Model Setting
# --------------------------------------------------------------------------------

XML_PATH = "/home/sid/projects25/src/bheema/unitree_g1/g1_with_hands.xml"

class ConfigurationState:
    def __init__(self):
        # 1. Base Pose (Lowered to 0.60m for knee leverage)
        self.base_pos = np.array([0.0, 0.0, 0.74])
        self.base_quad = np.array([0.0, 0.0, 0.0, 1.0])

        # 2. Leg joint angles [hip_p, hip_r, hip_y, knee, ankle_p, ankle_r]
        bent_leg = np.array([-0.3, 0.0, 0.0, 0.6, -0.3, 0.0])
        self.left_leg_angle = bent_leg.copy()
        self.right_leg_angle = bent_leg.copy()

        # 3. Waist and Arms 
        self.waist_angle = np.zeros(3)
        self.left_arm_angle = np.zeros(7)
        self.right_arm_angle = np.zeros(7)
        
        # 4. Hands (7 DOFs per hand: thumb x3, middle x2, index x2)
        self.left_hand_angle = np.zeros(7)
        self.right_hand_angle = np.zeros(7)

        # 5. Velocities
        self.base_vel = np.zeros(3)
        self.base_ang_vel = np.zeros(3)
        self.left_leg_vel = np.zeros(6)
        self.right_leg_vel = np.zeros(6)
        self.waist_vel = np.zeros(3)
        self.left_arm_vel = np.zeros(7)
        self.right_arm_vel = np.zeros(7)
        self.left_hand_vel = np.zeros(7)
        self.right_hand_vel = np.zeros(7)

    def get_q(self):
        # Generalized position: (50x1)
        q = np.concatenate([
            self.base_pos, self.base_quad,
            self.left_leg_angle, self.right_leg_angle,
            self.waist_angle, 
            self.left_arm_angle, self.right_arm_angle,
            self.left_hand_angle, self.right_hand_angle
        ])
        return q
    
    def get_dq(self):
        # Generalized velocity: (49x1)
        dq = np.concatenate([
            self.base_vel, self.base_ang_vel,
            self.left_leg_vel, self.right_leg_vel,
            self.waist_vel,
            self.left_arm_vel, self.right_arm_vel,
            self.left_hand_vel, self.right_hand_vel
        ])
        return dq
    
    def update_q(self, q):
        # base pose
        self.base_pos  = q[0:3]  # [x, y, z]
        self.base_quad = q[3:7]  # quaternion [x, y, z, w]

        # joints (Total 43 internal DOFs)
        self.left_leg_angle = q[7:13]
        self.right_leg_angle = q[13:19]
        self.waist_angle = q[19:22]
        self.left_arm_angle = q[22:29]
        self.right_arm_angle = q[29:36]
        self.left_hand_angle = q[36:43]
        self.right_hand_angle = q[43:50]

    def update_dq(self, v):
        # base twist
        self.base_vel     = v[0:3]      # [vx, vy, vz]
        self.base_ang_vel = v[3:6]      # [wx, wy, wz]

        # joint velocities
        self.left_leg_vel = v[6:12]
        self.right_leg_vel = v[12:18]
        self.waist_vel = v[18:21]
        self.left_arm_vel = v[21:28]
        self.right_arm_vel = v[28:35]
        self.left_hand_vel = v[35:42]
        self.right_hand_vel = v[42:49]
    
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
    
    def update_with_euler_angle(self, roll, pitch, yaw):
        cr,sr = np.cos(roll/2), np.sin(roll/2)
        cp,sp = np.cos(pitch/2), np.sin(pitch/2)
        cy,sy = np.cos(yaw/2), np.sin(yaw/2)
        
        qx = sr*cp*cy - cr*sp*sy
        qy = cr*sp*cy + sr*cp*sy
        qz = cr*cp*sy - sr*sp*cy
        qw = cr*cp*cy + sr*sp*sy

        self.base_quad = np.array([qx, qy, qz, qw])


class PinG1Model:
    def __init__(self, xml_path=XML_PATH):
        
        # 1. LOAD MJCF WITH ROBOT WRAPPER
        # Note: Do not pass root_joint=pin.JointModelFreeFlyer() because the MJCF 
        # <freejoint name="floating_base_joint"/> already tells Pinocchio to add one.
        self.robot = RobotWrapper.BuildFromMJCF(str(xml_path))
        self.model = self.robot.model
        self.data = self.model.createData()

        # Initial configuration
        self.current_config = ConfigurationState()
        self.q_init = self.current_config.get_q()
        self.dq_init = self.current_config.get_dq()

        # Frame IDs based on g1_with_hands.xml
        self.base_id = self.model.getFrameId("pelvis") 
        self.left_foot_id = self.model.getFrameId("left_ankle_roll_link")
        self.right_foot_id = self.model.getFrameId("right_ankle_roll_link")

        # Cache leg joint velocity indices for the Jacobian extraction
        left_names = ["left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
                      "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint"]
        right_names = ["right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
                       "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint"]
        
        self.vcols_L = [self.model.joints[self.model.getJointId(n)].idx_v for n in left_names]
        self.vcols_R = [self.model.joints[self.model.getJointId(n)].idx_v for n in right_names]

        # Initialize kinematics
        self.update_model(self.q_init, self.dq_init)

    def get_full_q_dq(self):
        """Returns the full 50-DOF position and 49-DOF velocity vectors"""
        return self.current_config.get_q(), self.current_config.get_dq()

    def compute_com_x_vec(self):
        # This function return the 6-DOF 12 states centroidal x-vector
        pos_com_world = self.pos_com_world
        rpy_com_world = self.current_config.compute_euler_angle_world()
        vel_com_world = self.vel_com_world
        rpy_rate_body = self.current_config.base_ang_vel
        
        R = self.R_body_to_world
        omega_world = R @ rpy_rate_body

        x_vec = np.concatenate([pos_com_world, rpy_com_world, 
                                vel_com_world, omega_world])
        
        return x_vec.reshape(-1, 1)

    def update_model(self, q, dq):
        self.current_config.update_q(q)
        self.current_config.update_dq(dq)

        pin.forwardKinematics(self.model, self.data, q, dq)
        pin.updateFramePlacements(self.model, self.data) 
        pin.computeAllTerms(self.model, self.data, q, dq)
        pin.computeJointJacobians(self.model, self.data, q)
        pin.computeJointJacobiansTimeVariation(self.model, self.data, q, dq)
        pin.ccrba(self.model, self.data, q, dq)
        pin.centerOfMass(self.model, self.data, q, dq)

        self.oMb = self.data.oMf[self.base_id]
        self.oMf_L = self.data.oMf[self.left_foot_id]
        self.oMf_R = self.data.oMf[self.right_foot_id]

        self.pos_com_world = self.data.com[0]
        self.vel_com_world = self.data.vcom[0]

        yaw = self.current_config.compute_euler_angle_world()[2]
        R_bw = np.array(self.oMb.rotation)

        self.R_body_to_world = R_bw
        self.R_world_to_body = R_bw.T

        self.R_z = np.array([
            [cos(yaw), -sin(yaw), 0],
            [sin(yaw),  cos(yaw), 0],
            [0,             0,            1]
        ])

    def get_foot_placement_in_world(self):
        pos_L = self.oMf_L.translation.copy()
        pos_R = self.oMf_R.translation.copy()
        return pos_L, pos_R
    
    def get_foot_lever_world(self):
        pos_com_world = self.pos_com_world    
        pos_L = self.oMf_L.translation - pos_com_world
        pos_R = self.oMf_R.translation - pos_com_world
        return pos_L, pos_R
    
    def get_single_foot_state_in_world(self, leg: str):
        foot_id = self.left_foot_id if leg.lower() == "left" else self.right_foot_id

        # Position in world 
        oMf = self.data.oMf[foot_id]
        foot_pos_world = oMf.translation.copy()  # (3,)

        # 6D spatial velocity in LOCAL_WORLD_ALIGNED (axes = world)
        v6 = pin.getFrameVelocity(self.model, self.data, foot_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        foot_vel_world = np.array(v6.linear).copy()  # (3,)

        return foot_pos_world, foot_vel_world
    
    def compute_leg_Jacobian_world(self, leg: str):
        """Returns the FULL 6x6 Spatial Jacobian for the leg to allow Ankle Wrench Control"""
        foot_id = self.left_foot_id if leg.lower() == "left" else self.right_foot_id
        vcols = self.vcols_L if leg.lower() == "left" else self.vcols_R

        # 6xnv full spatial Jacobian in world frame
        J_world = pin.getFrameJacobian(self.model, self.data, foot_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        
        # EXTRACT FULL 6 ROWS (Needed for flat feet)
        J_leg_pos_world = J_world[:, vcols] 
        return J_leg_pos_world

    def compute_Jdot_dq_world(self, leg: str):
        foot_id = self.left_foot_id if leg.lower() == "left" else self.right_foot_id

        Jdot = pin.getFrameJacobianTimeVariation(
            self.model, self.data, foot_id,
            pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
        )
        
        # Extract full 6D acceleration drift
        Jdot_dq = Jdot[:, :] @ self.current_config.get_dq()
        return np.asarray(Jdot_dq).reshape(6,)
    
    def compute_dynamics_terms(self):
        g = self.data.g            # gravity torque term (49 x 1)
        C = self.data.C            # Coriolis matrix (49 x 49)
        M = self.data.M            # joint-space inertia matrix (49 x 49)
        return g, C, M