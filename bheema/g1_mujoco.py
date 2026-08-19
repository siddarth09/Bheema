import mujoco as mj 
import mujoco.viewer as mjv 
from pathlib import Path 
import time 
import pinocchio as pin 
import numpy as np 

from bheema.g1_config import PinG1Model
from bheema.params import BodyParams


XML_PATH = str(Path(__file__).parent.parent/ "unitree_g1"/ "scenes" / "scene_platform_easy.xml")
class MuJoCo_G1_Model:
    def __init__(self, xml_path=XML_PATH, z_offset: float | None = None):
        self.model = mj.MjModel.from_xml_path(str(xml_path))
        self.data = mj.MjData(self.model)
        self.z_offset = BodyParams().pin_mujoco_z_offset if z_offset is None else z_offset
        self.viewer = None 
        self.base_bid = mj.mj_name2id(self.model,mj.mjtObj.mjOBJ_BODY,"pelvis")

        self.leg_joint_names = [
            # Left Leg
            "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
            "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
            # Right Leg
            "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
            "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint"
        ]

        # Foot collision geoms, for measuring real contact. The G1's collision geoms are
        # unnamed, so they are identified by owning body instead.
        self.foot_geoms = {}
        for side in ("left", "right"):
            bid = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, f"{side}_ankle_roll_link")
            self.foot_geoms[side.upper()] = set(
                np.flatnonzero(self.model.geom_bodyid == bid).tolist()
            ) if bid != -1 else set()
        self._f6 = np.zeros(6)

        self.actuator_ids = [] 
        self.qpos_adrs = [] 
        self.qvel_adrs = [] 

        for name in self.leg_joint_names:
            self.actuator_ids.append(mj.mj_name2id(self.model,mj.mjtObj.mjOBJ_ACTUATOR,name))
            jid = mj.mj_name2id(self.model,mj.mjtObj.mjOBJ_JOINT,name)
            self.qpos_adrs.append(self.model.jnt_qposadr[jid])
            self.qvel_adrs.append(self.model.jnt_dofadr[jid])


    def update_with_q_pin(self,q_pin):

        self.data.qpos[0:3] = q_pin[0:3] + np.array([0.0, 0.0, self.z_offset])
        self.data.qpos[3] = q_pin[6]
        self.data.qpos[4:7] = q_pin[3:6]
        self.data.qpos[7:] = q_pin[7:]

        mj.mj_forward(self.model,self.data)

    def foot_contact_forces(self):
        """Normal contact force magnitude per foot, as (LEFT, RIGHT) in newtons.

        The gait schedule is open-loop, so it can declare a foot to be in stance while it is
        airborne. This is the signal for detecting that.
        """
        out = {"LEFT": 0.0, "RIGHT": 0.0}
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            for side, geoms in self.foot_geoms.items():
                if c.geom1 in geoms or c.geom2 in geoms:
                    mj.mj_contactForce(self.model, self.data, i, self._f6)
                    out[side] += abs(self._f6[0])
        return out["LEFT"], out["RIGHT"]

    def set_arm_posture(self):
        l_shoulder_id = mj.mj_name2id(self.model,mj.mjtObj.mjOBJ_ACTUATOR,"left_shoulder_pitch_joint")
        r_shoulder_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_ACTUATOR, "right_shoulder_pitch_joint")
        self.data.ctrl[l_shoulder_id] = 0.0
        self.data.ctrl[r_shoulder_id] = 0.0


    def set_joint_torque(self,torque:np.ndarray):
        for i,aid in enumerate(self.actuator_ids):
            if aid != -1:
                self.data.ctrl[aid] = torque[i]


    def update_pin_with_mujoco(self, g1: PinG1Model):
        """Extracts the state from MuJoCo and syncs the Pinocchio Kinematics model."""
        mujoco_q  = np.asarray(self.data.qpos, dtype=float).reshape(-1)
        mujoco_dq = np.asarray(self.data.qvel, dtype=float).reshape(-1)
        
        # 1. Base Orientation (Pinocchio uses x,y,z,w; MuJoCo uses w,x,y,z)
        qw, qx, qy, qz = mujoco_q[3:7]
        R = pin.Quaternion(qw, qx, qy, qz).toRotationMatrix() 
        
        # 2. Base Velocities (Converting World Linear Vel to Body Frame)
        v_world = mujoco_dq[0:3]
        w_body = mujoco_dq[3:6]
        v_body = R.T @ v_world

        # 3. Update Pinocchio state
        g1.current_config.base_pos = mujoco_q[0:3] - np.array([0.0, 0.0, self.z_offset])
        g1.current_config.base_quad = np.array([qx, qy, qz, qw]) 
        g1.current_config.base_vel = v_body
        g1.current_config.base_ang_vel = w_body

        # 4. Extract Leg joints
        for i in range(6):
            g1.current_config.left_leg_angle[i] = mujoco_q[self.qpos_adrs[i]]
            g1.current_config.left_leg_vel[i]   = mujoco_dq[self.qvel_adrs[i]]
            g1.current_config.right_leg_angle[i] = mujoco_q[self.qpos_adrs[i+6]]
            g1.current_config.right_leg_vel[i]   = mujoco_dq[self.qvel_adrs[i+6]]

        # 5. Refresh Pinocchio
        q, dq = g1.get_full_q_dq()
        g1.update_model(q, dq)

    def replay_simulation(self, time_log_s, q_log, tau_log_Nm, RENDER_DT, REALTIME_FACTOR):
        model = self.model
        data_replay = mj.MjData(model)

        with mjv.launch_passive(model, data_replay) as viewer:
            viewer.cam.type = mj.mjtCamera.mjCAMERA_TRACKING
            viewer.cam.trackbodyid = self.base_bid
            viewer.opt.flags[mj.mjtVisFlag.mjVIS_CONTACTPOINT] = True

            while viewer.is_running():           
                start_wall = time.perf_counter()
                t0 = time_log_s[0]
                next_render_t = t0
                k = 0
                T = len(time_log_s)

                while k < T and viewer.is_running():
                    t = time_log_s[k]
                    if t >= next_render_t:
                        data_replay.qpos[:] = q_log[k]
                        for i, aid in enumerate(self.actuator_ids):
                            if aid != -1:
                                data_replay.ctrl[aid] = tau_log_Nm[k][i]
                        mj.mj_forward(model, data_replay)

                        target_wall = start_wall + (t - t0) / REALTIME_FACTOR
                        while time.perf_counter() < target_wall:
                            pass 
                        next_render_t += RENDER_DT
                    k += 1