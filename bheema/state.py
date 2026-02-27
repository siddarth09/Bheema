import pinocchio as pin
import numpy as np
from pinocchio.robot_wrapper import RobotWrapper
from pathlib import Path

# Adjust this path if necessary
G1_URDF_PATH = Path("/home/sid/projects25/src/bheema/unitree_g1/g1_with_hands.urdf")

class G1State:
    """
    State estimator and Kinematics wrapper using Pinocchio (via URDF).
    """
    def __init__(self):
        
        if not G1_URDF_PATH.exists():
            raise FileNotFoundError(f"URDF not found at {G1_URDF_PATH}")

        print(f"[G1State] Loading URDF: {G1_URDF_PATH}")
        
        # 1. Attempt to Load Model
        try:
            # Pinocchio needs to know where 'package://bheema' is located.
            # Assuming path is .../src/bheema/unitree_g1/g1.urdf
            # .parent = unitree_g1, .parent.parent = bheema, .parent.parent.parent = src
            pkg_root = str(G1_URDF_PATH.parent.parent.parent)
            
            self.robot = RobotWrapper.BuildFromURDF(
                str(G1_URDF_PATH),
                package_dirs=[pkg_root],
                root_joint=pin.JointModelFreeFlyer()
            )
        except ValueError:
            # 2. Fallback: Load Kinematics ONLY (Ignore Meshes)
            # This is perfect for Controllers/WBC which don't need visuals
            print("[Warn] Mesh loading failed. Loading kinematic model only (Safe for WBC).")
            self.model = pin.buildModelFromUrdf(str(G1_URDF_PATH), pin.JointModelFreeFlyer())
            self.robot = RobotWrapper(model=self.model)

        self.model = self.robot.model
        self.data = self.model.createData()

        self.nq = self.model.nq
        self.nv = self.model.nv

        # 3. Initialize State
        self.q = pin.neutral(self.model)
        self.dq = np.zeros(self.nv)

        # 4. Resolve Frame IDs
        self.base_frame = self.model.getFrameId("pelvis")
        
        # Robust frame search
        l_foot_candidates = ["left_ankle_roll_link", "left_foot_link", "left_foot"]
        r_foot_candidates = ["right_ankle_roll_link", "right_foot_link", "right_foot"]
        
        self.left_foot_frame = self._find_frame(l_foot_candidates)
        self.right_foot_frame = self._find_frame(r_foot_candidates)

        print(f"[G1State] Frames Found -> Base: {self.base_frame}, Left: {self.left_foot_frame}, Right: {self.right_foot_frame}")

    def _find_frame(self, candidates):
        for name in candidates:
            if self.model.existFrame(name):
                return self.model.getFrameId(name)
        
        print(f"\n[ERROR] Frame not found. Searched for: {candidates}")
        # Print available frames to help debug
        print("Available frames:", [f.name for f in self.model.frames if "foot" in f.name or "ankle" in f.name])
        raise ValueError(f"Could not find critical frame from candidates: {candidates}")

    # =====================================================
    # Setters
    # =====================================================

    def set_base_pose(self, pos, quat):
        self.q[0:3] = pos
        self.q[3:7] = quat

    def set_base_velocity(self, v_lin, v_ang):
        # Convert World Frame velocity to Body Frame orientation
        # (Standard Pinocchio FreeFlyer convention)
        R = pin.Quaternion(self.q[3:7]).toRotationMatrix()
        v_body = R.T @ v_lin
        w_body = R.T @ v_ang
        
        self.dq[0:3] = v_body
        self.dq[3:6] = w_body

    def set_joint_state(self, qj, dqj):
        self.q[7:] = qj
        self.dq[6:] = dqj

    # =====================================================
    # Updates & Getters
    # =====================================================

    def update(self):
        # Compute Kinematics & Jacobians
        pin.forwardKinematics(self.model, self.data, self.q, self.dq)
        pin.updateFramePlacements(self.model, self.data)
        pin.computeJointJacobians(self.model, self.data, self.q)
        pin.centerOfMass(self.model, self.data, self.q, self.dq)
        pin.ccrba(self.model, self.data, self.q, self.dq)

    def get_mass(self):
       
        return pin.computeTotalMass(self.model)

    def get_centroidal_state(self):
        p = self.data.com[0]
        v = self.data.vcom[0]
        
        R = pin.Quaternion(self.q[3:7]).toRotationMatrix()
        rpy = pin.rpy.matrixToRpy(R)
        
        w_body = self.dq[3:6]
        w_world = R @ w_body

        return np.concatenate([p, rpy, v, w_world]).reshape(12, 1)