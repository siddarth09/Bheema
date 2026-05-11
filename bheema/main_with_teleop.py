"""
Bheema — Teleop Mode
=====================
Full simulation loop with keyboard-driven velocity commands.
Everything runs: MuJoCo viewer, Pinocchio kinematics, MPC, leg controller.
You drive the robot with WASD + QE.

Controls:
    W/S     Forward / Backward
    A/D     Strafe Left / Right  
    Q/E     Turn Left / Right
    Z/X     Raise / Lower CoM
    SPACE   Emergency Stop
    ESC     Quit
"""

import os
os.environ["MPLBACKEND"] = "TkAgg"
import time
import mujoco as mj
import mujoco.viewer as mjv
import numpy as np
from dataclasses import dataclass, field

from bheema.g1_config import PinG1Model
from bheema.g1_mujoco import MuJoCo_G1_Model
from bheema.com_traj import ComTraj
from bheema.centroidal_mpc import CentroidalMPC
from bheema.leg_controller import LegController
from bheema.gait import Gait
from bheema.teleop import Teleop
import matplotlib.pyplot as plt
from bheema.plotter import plot_mpc_result, plot_swing_foot_traj, plot_solve_time

# --------------------------------------------------------------------------------
# Parameters
# --------------------------------------------------------------------------------

INITIAL_X_POS = 0.0
INITIAL_Y_POS = 0.0
RUN_SIM_LENGTH_S = 120.0   # Long session for teleop

RENDER_HZ = 120.0
RENDER_DT = 1.0 / RENDER_HZ

NOMINAL_Z = 0.66

# Gait
GAIT_HZ = 1.2
GAIT_DUTY = 0.75
GAIT_T = 1.0 / GAIT_HZ

# MuJoCo physics
SIM_HZ = 2000
SIM_DT = 1.0 / SIM_HZ

# Leg controller
CTRL_HZ = 200
CTRL_DT = 1.0 / CTRL_HZ

if SIM_HZ % CTRL_HZ != 0:
    raise ValueError(f"SIM_HZ ({SIM_HZ}) must be divisible by CTRL_HZ ({CTRL_HZ})")
CTRL_DECIM = SIM_HZ // CTRL_HZ

SIM_STEPS = int(RUN_SIM_LENGTH_S * SIM_HZ)
CTRL_STEPS = int(RUN_SIM_LENGTH_S * CTRL_HZ)

# MPC
MPC_DT = GAIT_T / 16
MPC_HZ = 1.0 / MPC_DT
STEPS_PER_MPC = max(1, int(CTRL_HZ // MPC_HZ))

TAU_LIM = np.array([
    88.0, 139.0, 88.0, 139.0, 50.0, 50.0,
    88.0, 139.0, 88.0, 139.0, 50.0, 50.0
])

LEG_SLICE = {
    "LEFT": slice(0, 6),
    "RIGHT": slice(6, 12),
}

# --------------------------------------------------------------------------------
# Storage
# --------------------------------------------------------------------------------
x_vec = np.zeros((12, CTRL_STEPS))
mpc_force_world = np.zeros((12, CTRL_STEPS))
tau_raw = np.zeros((12, CTRL_STEPS))
tau_cmd = np.zeros((12, CTRL_STEPS))
time_log_ctrl_s = np.zeros(CTRL_STEPS)

@dataclass
class FootTraj:
    pos_des: np.ndarray = field(default_factory=lambda: np.zeros((12, CTRL_STEPS)))
    pos_now: np.ndarray = field(default_factory=lambda: np.zeros((12, CTRL_STEPS)))
    vel_des: np.ndarray = field(default_factory=lambda: np.zeros((12, CTRL_STEPS)))
    vel_now: np.ndarray = field(default_factory=lambda: np.zeros((12, CTRL_STEPS)))

foot_traj = FootTraj()
mpc_update_time_ms = []
mpc_solve_time_ms = []

# --------------------------------------------------------------------------------
# Initialization
# --------------------------------------------------------------------------------

g1 = PinG1Model()
mujoco_g1 = MuJoCo_G1_Model()
leg_controller = LegController()
traj = ComTraj(g1)
gait = Gait(GAIT_HZ, GAIT_DUTY)

# Start teleop
teleop = Teleop(
    max_vx=1.0,
    max_vy=0.4,
    max_yaw_rate=0.6,
    nominal_z=NOMINAL_Z,
    ramp_rate=2.0,
)
if not teleop.start():
    print("Teleop failed to start. Running with zero commands.")

# Init robot pose
q_init, _ = g1.get_full_q_dq()
q_init[0], q_init[1] = INITIAL_X_POS, INITIAL_Y_POS
mujoco_g1.update_with_q_pin(q_init)
mujoco_g1.model.opt.timestep = SIM_DT

# Init MPC with zero velocity
traj.generate_traj(g1, gait, 0.0, 0.0, 0.0, NOMINAL_Z, 0.0, time_step=MPC_DT)
mpc = CentroidalMPC(g1, traj)
U_opt = np.zeros((12, traj.N), dtype=float)
N = traj.N

# --------------------------------------------------------------------------------
# Simulation Loop
# --------------------------------------------------------------------------------

print(f"\nRunning Teleop Simulation — use WASD to walk, ESC to quit\n")
sim_start_time = time.perf_counter()

ctrl_i = 0
tau_hold = np.zeros(12, dtype=float)
last_mpc_time = 0.0

with mjv.launch_passive(mujoco_g1.model, mujoco_g1.data) as viewer:

    viewer.cam.type = mj.mjtCamera.mjCAMERA_TRACKING
    viewer.cam.trackbodyid = mujoco_g1.base_bid
    viewer.cam.distance = 2.5
    viewer.cam.elevation = -20
    viewer.cam.azimuth = 90
    viewer.opt.flags[mj.mjtVisFlag.mjVIS_CONTACTPOINT] = True

    for k in range(SIM_STEPS):

        # Exit conditions
        if not viewer.is_running() or not teleop.is_running():
            break

        time_now_s = float(mujoco_g1.data.time)

        # ---- Control tick at CTRL_HZ ----
        if (k % CTRL_DECIM) == 0 and ctrl_i < CTRL_STEPS:

            # Read teleop commands (replaces CMD_SCHEDULE)
            x_vel_des_body, y_vel_des_body, z_pos_des_body, yaw_rate_des_body = teleop.get_cmd()

            # Sync Pinocchio with MuJoCo
            mujoco_g1.update_pin_with_mujoco(g1)
            x_vec[:, ctrl_i] = g1.compute_com_x_vec().reshape(-1)
            time_log_ctrl_s[ctrl_i] = time_now_s

            # ---- MPC solve ----
            if (ctrl_i % STEPS_PER_MPC) == 0:

                # HUD: print current command and sim time
                print(f"\r  t={time_now_s:6.2f}s  "
                      f"vx={x_vel_des_body:+.2f}  vy={y_vel_des_body:+.2f}  "
                      f"yaw={yaw_rate_des_body:+.2f}  z={z_pos_des_body:.3f}  ",
                      end="", flush=True)

                traj.generate_traj(
                    g1, gait, time_now_s,
                    x_vel_des_body, y_vel_des_body,
                    z_pos_des_body, yaw_rate_des_body,
                    time_step=MPC_DT,
                )

                sol = mpc.solve_QP(g1, traj, False)
                mpc_solve_time_ms.append(mpc.solve_time)
                mpc_update_time_ms.append(mpc.update_time)

                N = traj.N
                w_opt = sol["x"].full().flatten()
                U_opt = w_opt[12 * N:].reshape((12, N), order="F")
                last_mpc_time = time_now_s

            # ---- Wrench interpolation ----
            time_since_mpc = time_now_s - last_mpc_time
            k_interp = int(time_since_mpc / MPC_DT)
            k_interp = min(k_interp, N - 1)
            mpc_force_world[:, ctrl_i] = U_opt[:, k_interp]

            # ---- Gait timing ----
            # When standing still, freeze gait at double support
            is_standing = (abs(x_vel_des_body) < 0.01
                           and abs(y_vel_des_body) < 0.01
                           and abs(yaw_rate_des_body) < 0.01)
            gait_time = 0.0 if is_standing else time_now_s

            # ---- Leg controller ----
            LEFT = leg_controller.compute_leg_torque(
                "LEFT", g1, gait,
                mpc_force_world[LEG_SLICE["LEFT"], ctrl_i],
                gait_time,
            )
            tau_raw[LEG_SLICE["LEFT"], ctrl_i] = LEFT.tau
            foot_traj.pos_des[LEG_SLICE["LEFT"], ctrl_i] = np.pad(LEFT.pos_des, (0, 3))
            foot_traj.pos_now[LEG_SLICE["LEFT"], ctrl_i] = np.pad(LEFT.pos_now, (0, 3))
            foot_traj.vel_des[LEG_SLICE["LEFT"], ctrl_i] = np.pad(LEFT.vel_des, (0, 3))
            foot_traj.vel_now[LEG_SLICE["LEFT"], ctrl_i] = np.pad(LEFT.vel_now, (0, 3))

            RIGHT = leg_controller.compute_leg_torque(
                "RIGHT", g1, gait,
                mpc_force_world[LEG_SLICE["RIGHT"], ctrl_i],
                gait_time,
            )
            tau_raw[LEG_SLICE["RIGHT"], ctrl_i] = RIGHT.tau
            foot_traj.pos_des[LEG_SLICE["RIGHT"], ctrl_i] = np.pad(RIGHT.pos_des, (0, 3))
            foot_traj.pos_now[LEG_SLICE["RIGHT"], ctrl_i] = np.pad(RIGHT.pos_now, (0, 3))
            foot_traj.vel_des[LEG_SLICE["RIGHT"], ctrl_i] = np.pad(RIGHT.vel_des, (0, 3))
            foot_traj.vel_now[LEG_SLICE["RIGHT"], ctrl_i] = np.pad(RIGHT.vel_now, (0, 3))

            # Clip and hold torques
            tau_cmd[:, ctrl_i] = np.clip(tau_raw[:, ctrl_i], -TAU_LIM, TAU_LIM)
            tau_hold = tau_cmd[:, ctrl_i].copy()
            ctrl_i += 1

        # ---- Physics step at SIM_HZ ----
        mujoco_g1.set_joint_torque(tau_hold)

        # Upper body posture hold
        for name in ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"]:
            aid = mj.mj_name2id(mujoco_g1.model, mj.mjtObj.mjOBJ_ACTUATOR, name)
            if aid != -1:
                mujoco_g1.data.ctrl[aid] = 0.0

        for name in ["left_shoulder_pitch_joint", "right_shoulder_pitch_joint"]:
            aid = mj.mj_name2id(mujoco_g1.model, mj.mjtObj.mjOBJ_ACTUATOR, name)
            if aid != -1:
                mujoco_g1.data.ctrl[aid] = 0.15

        mj.mj_step(mujoco_g1.model, mujoco_g1.data)

        # ---- Render sync ----
        if k % int(SIM_HZ / RENDER_HZ) == 0:
            viewer.sync()
            time_until_next = mujoco_g1.data.time - (time.perf_counter() - sim_start_time)
            if time_until_next > 0:
                time.sleep(time_until_next)

# --------------------------------------------------------------------------------
# Cleanup and Plots
# --------------------------------------------------------------------------------

teleop.stop()

sim_end_time = time.perf_counter()
print(f"\n\nSimulation ended."
      f"\nElapsed: {sim_end_time - sim_start_time:.1f}s"
      f"\nControl ticks: {ctrl_i}/{CTRL_STEPS}")

t_vec = np.arange(ctrl_i) * CTRL_DT

plot_swing_foot_traj(t_vec, foot_traj.pos_now, foot_traj.pos_des,
                     foot_traj.vel_now, foot_traj.vel_des, block=False)
plot_mpc_result(t_vec, mpc_force_world, tau_cmd, x_vec, block=False)
plot_solve_time(mpc_solve_time_ms, mpc_update_time_ms, MPC_DT, MPC_HZ, block=True)