"""
Bheema RL Bridge — Fixed for G1 with Hands MJCF
=================================================

The policy was trained on 29 joints (no hands).
Your MJCF has 43 joints (29 + 14 hand joints).

This bridge correctly maps between the two by using
explicit qpos/qvel addresses instead of naive slicing.

Key differences from mjlab's training model:
  - Your MJCF: 50 qpos, 49 qvel, 43 actuators (includes 14 hand joints)
  - Training model: 36 qpos, 35 qvel, 29 joints (no hands)
  - Legs use motor actuators (direct torque control)
  - Upper body uses position actuators (built-in PD)
"""

import torch
import numpy as np
import mujoco as mj


# ═══════════════════════════════════════════════════════════
# The 29 joints the policy controls (in policy output order)
# These match mjlab's training MJCF joint ordering
# ═══════════════════════════════════════════════════════════

POLICY_JOINTS = [
    "left_hip_pitch_joint",       # policy idx 0
    "left_hip_roll_joint",        # 1
    "left_hip_yaw_joint",         # 2
    "left_knee_joint",            # 3
    "left_ankle_pitch_joint",     # 4
    "left_ankle_roll_joint",      # 5
    "right_hip_pitch_joint",      # 6
    "right_hip_roll_joint",       # 7
    "right_hip_yaw_joint",        # 8
    "right_knee_joint",           # 9
    "right_ankle_pitch_joint",    # 10
    "right_ankle_roll_joint",     # 11
    "waist_yaw_joint",            # 12
    "waist_roll_joint",           # 13
    "waist_pitch_joint",          # 14
    "left_shoulder_pitch_joint",  # 15
    "left_shoulder_roll_joint",   # 16
    "left_shoulder_yaw_joint",    # 17
    "left_elbow_joint",           # 18
    "left_wrist_roll_joint",      # 19
    "left_wrist_pitch_joint",     # 20
    "left_wrist_yaw_joint",       # 21
    "right_shoulder_pitch_joint", # 22
    "right_shoulder_roll_joint",  # 23
    "right_shoulder_yaw_joint",   # 24
    "right_elbow_joint",          # 25
    "right_wrist_roll_joint",     # 26
    "right_wrist_pitch_joint",    # 27
    "right_wrist_yaw_joint",      # 28
]

# Default standing pose (from mjlab training config)
DEFAULT_QPOS = np.zeros(29, dtype=np.float32)
_DEFAULTS = {
    "left_hip_pitch_joint": -0.312,
    "right_hip_pitch_joint": -0.312,
    "left_knee_joint": 0.669,
    "right_knee_joint": 0.669,
    "left_ankle_pitch_joint": -0.363,
    "right_ankle_pitch_joint": -0.363,
    "left_elbow_joint": 0.6,
    "right_elbow_joint": 0.6,
    "left_shoulder_roll_joint": 0.2,
    "left_shoulder_pitch_joint": 0.2,
    "right_shoulder_roll_joint": -0.2,
    "right_shoulder_pitch_joint": 0.2,
}
for name, val in _DEFAULTS.items():
    DEFAULT_QPOS[POLICY_JOINTS.index(name)] = val

# Per-joint action scale from mjlab
ACTION_SCALE = np.zeros(29, dtype=np.float32)
_SCALES = {
    "hip_pitch": 0.5475, "hip_roll": 0.3507, "hip_yaw": 0.5475,
    "knee": 0.3507, "ankle_pitch": 0.4386, "ankle_roll": 0.4386,
    "waist_yaw": 0.5475, "waist_roll": 0.4386, "waist_pitch": 0.4386,
    "shoulder_pitch": 0.4386, "shoulder_roll": 0.4386, "shoulder_yaw": 0.4386,
    "elbow": 0.4386, "wrist_roll": 0.4386, "wrist_pitch": 0.0745, "wrist_yaw": 0.0745,
}
for i, name in enumerate(POLICY_JOINTS):
    for pat, s in _SCALES.items():
        if pat in name:
            ACTION_SCALE[i] = s
            break

# Per-joint PD gains from mjlab actuator config
KP = np.zeros(29, dtype=np.float32)
KD = np.zeros(29, dtype=np.float32)
_PD = {
    "hip_pitch": (40.18, 2.56), "hip_roll": (99.10, 6.31), "hip_yaw": (40.18, 2.56),
    "knee": (99.10, 6.31), "ankle_pitch": (28.50, 1.81), "ankle_roll": (28.50, 1.81),
    "waist_yaw": (40.18, 2.56), "waist_roll": (28.50, 1.81), "waist_pitch": (28.50, 1.81),
    "shoulder_pitch": (14.25, 0.91), "shoulder_roll": (14.25, 0.91),
    "shoulder_yaw": (14.25, 0.91), "elbow": (14.25, 0.91), "wrist_roll": (14.25, 0.91),
    "wrist_pitch": (16.78, 1.07), "wrist_yaw": (16.78, 1.07),
}
for i, name in enumerate(POLICY_JOINTS):
    for pat, (kp, kd) in _PD.items():
        if pat in name:
            KP[i] = kp
            KD[i] = kd
            break


class RLBridge:

    def __init__(self, checkpoint_path: str, mujoco_model, policy_hz: float = 50.0):
        """
        Args:
            checkpoint_path: path to model_*.pt
            mujoco_model: mj.MjModel — needed to build joint index maps
            policy_hz: inference rate
        """
        # ── Build joint index maps ──
        # For each of the 29 policy joints, find its qpos and qvel address
        # in YOUR specific MJCF (which has hands interleaved)
        self.qpos_indices = np.zeros(29, dtype=np.int32)
        self.qvel_indices = np.zeros(29, dtype=np.int32)
        self.actuator_ids = np.full(29, -1, dtype=np.int32)

        for i, name in enumerate(POLICY_JOINTS):
            jid = mj.mj_name2id(mujoco_model, mj.mjtObj.mjOBJ_JOINT, name)
            if jid == -1:
                print(f"[RLBridge] WARNING: joint '{name}' not found in MJCF!")
                continue
            self.qpos_indices[i] = mujoco_model.jnt_qposadr[jid]
            self.qvel_indices[i] = mujoco_model.jnt_dofadr[jid]

            aid = mj.mj_name2id(mujoco_model, mj.mjtObj.mjOBJ_ACTUATOR, name)
            self.actuator_ids[i] = aid

        print(f"[RLBridge] Joint qpos indices: {self.qpos_indices.tolist()}")
        print(f"[RLBridge] Joint qvel indices: {self.qvel_indices.tolist()}")

        # ── Load network weights ──
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        actor = ckpt["actor_state_dict"]

        self.obs_mean = actor["obs_normalizer._mean"].squeeze(0).numpy().astype(np.float32)
        self.obs_std = np.clip(
            actor["obs_normalizer._std"].squeeze(0).numpy().astype(np.float32),
            1e-6, None
        )

        self.weights = [
            (actor[f"mlp.{l}.weight"].numpy().astype(np.float32),
             actor[f"mlp.{l}.bias"].numpy().astype(np.float32))
            for l in [0, 2, 4, 6]
        ]

        obs_dim = self.weights[0][0].shape[1]
        act_dim = self.weights[-1][0].shape[0]
        print(f"[RLBridge] Network: {obs_dim} → 512 → 256 → 128 → {act_dim}")
        print(f"[RLBridge] Running at {policy_hz} Hz")

        # ── Determine actuator types ──
        # Legs (0-11): motor actuators → we send PD-computed torques
        # Upper body (12-28): position actuators → we send position targets directly
        self.is_motor = np.zeros(29, dtype=bool)
        for i in range(29):
            aid = self.actuator_ids[i]
            if aid != -1:
                # MuJoCo actuator transmission type: 0 = motor/general, 1 = position
                # Check gaintype: motor has gaintype=0 (fixed), position has gaintype=1
                # Simpler: check if it's in the first 12 (legs = motor)
                self.is_motor[i] = (i < 12)

        print(f"[RLBridge] Motor actuators (PD torque): joints 0-11")
        print(f"[RLBridge] Position actuators (target): joints 12-28")

        # ── State ──
        self.policy_dt = 1.0 / policy_hz
        self.q_targets = DEFAULT_QPOS.copy()
        self.last_action = np.zeros(act_dim, dtype=np.float32)
        self._last_policy_time = -np.inf

    @staticmethod
    def _elu(x):
        return np.where(x > 0, x, np.exp(np.clip(x, -10, 0)) - 1.0)

    def _forward(self, obs):
        x = obs.astype(np.float32)
        for i, (w, b) in enumerate(self.weights):
            x = x @ w.T + b
            if i < len(self.weights) - 1:
                x = self._elu(x)
        return x

    def _read_joint_pos(self, data):
        """Read the 29 policy joint positions from the correct qpos addresses."""
        return data.qpos[self.qpos_indices].astype(np.float32)

    def _read_joint_vel(self, data):
        """Read the 29 policy joint velocities from the correct qvel addresses."""
        return data.qvel[self.qvel_indices].astype(np.float32)

    def build_observation(self, data, velocity_cmd):
        """
        Build the 99-dim observation.

        Order (matching mjlab training):
          base_lin_vel     (3)
          base_ang_vel     (3)
          projected_gravity(3)
          joint_pos        (29) — relative to default
          joint_vel        (29)
          actions          (29)
          command           (3)
        """
        # Quaternion → rotation matrix (world to body)
        quat = data.qpos[3:7]  # MuJoCo: [w, x, y, z]
        w, x, y, z = quat
        # Body-to-world rotation matrix
        R_bw = np.array([
            [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
            [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
            [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)]
        ])
        R_wb = R_bw.T  # World-to-body

        # 1. Base linear velocity in body frame
        base_lin_vel = (R_wb @ data.qvel[0:3]).astype(np.float32)

        # 2. Base angular velocity (already in body frame in MuJoCo)
        base_ang_vel = data.qvel[3:6].astype(np.float32)

        # 3. Projected gravity: rotate [0,0,-1] into body frame
        proj_grav = (R_wb @ np.array([0.0, 0.0, -1.0])).astype(np.float32)

        # 4. Joint positions relative to default (using correct indices)
        joint_pos = self._read_joint_pos(data) - DEFAULT_QPOS

        # 5. Joint velocities (using correct indices)
        joint_vel = self._read_joint_vel(data)

        # 6. Previous action
        prev_action = self.last_action

        # 7. Command
        cmd = np.array(velocity_cmd, dtype=np.float32)

        obs = np.concatenate([
            base_lin_vel,   # 3
            base_ang_vel,   # 3
            proj_grav,      # 3
            joint_pos,      # 29
            joint_vel,      # 29
            prev_action,    # 29
            cmd,            # 3
        ])
        return obs

    def step(self, data, velocity_cmd, current_time):
        """
        Run policy at policy_hz. Returns 29 joint position targets.
        """
        if current_time - self._last_policy_time >= self.policy_dt:
            obs = self.build_observation(data, velocity_cmd)
            obs_norm = (obs - self.obs_mean) / self.obs_std

            action = self._forward(obs_norm)
            action = np.clip(action, -5.0, 5.0)

            self.q_targets = DEFAULT_QPOS + ACTION_SCALE * action
            self.last_action = action
            self._last_policy_time = current_time

        return self.q_targets

    def apply_control(self, data):
        """
        Apply control to MuJoCo at SIM_HZ.

        - Leg joints (motor actuators): compute PD torque and write to ctrl
        - Upper body (position actuators): write position target directly to ctrl
        """
        q_now = self._read_joint_pos(data)
        dq_now = self._read_joint_vel(data)

        for i in range(29):
            aid = self.actuator_ids[i]
            if aid == -1:
                continue

            if self.is_motor[i]:
                # Legs: PD torque
                tau = KP[i] * (self.q_targets[i] - q_now[i]) - KD[i] * dq_now[i]
                data.ctrl[aid] = tau
            else:
                # Upper body: position target (MuJoCo's built-in PD handles it)
                data.ctrl[aid] = self.q_targets[i]

    def reset(self):
        self.q_targets = DEFAULT_QPOS.copy()
        self.last_action = np.zeros(29, dtype=np.float32)
        self._last_policy_time = -np.inf