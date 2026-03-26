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
from bheema.plotter import plot_mpc_result, plot_swing_foot_traj, plot_solve_time, hold_until_all_fig_closed

# --------------------------------------------------------------------------------
# Parameters
# --------------------------------------------------------------------------------

# Simulation Setting
INITIAL_X_POS = -10
INITIAL_Y_POS = 0
RUN_SIM_LENGTH_S = 10.0

RENDER_HZ = 120.0
RENDER_DT = 1.0 / RENDER_HZ

# Locomotion Command
@dataclass
class BodyCmdPhase:
    t_start: float
    t_end: float
    x_vel: float
    y_vel: float
    z_pos: float
    yaw_rate: float


NOMINAL_Z = 0.68

CMD_SCHEDULE = [
    BodyCmdPhase(0.0, 1.0,  0.6,  0.0, NOMINAL_Z, 0.0),    
    
    BodyCmdPhase(3.0, 10.0, 2.0, 0.0, NOMINAL_Z, 0.0), # Second gear: Striding
]

# Gait Setting (Biped Walk)
GAIT_HZ = 1.0
GAIT_DUTY = 0.60
GAIT_T = 1.0 / GAIT_HZ

# Trajectory Reference Setting (defaults)
x_vel_des_body = 0.0
y_vel_des_body = 0.0
z_pos_des_body = NOMINAL_Z
yaw_rate_des_body = 0.0

# MuJoCo Sim Update Rate
SIM_HZ = 2000
SIM_DT = 1.0 / SIM_HZ

# Leg Controller Update Rate
CTRL_HZ = 200       
CTRL_DT = 1.0 / CTRL_HZ

if SIM_HZ % CTRL_HZ != 0:
    raise ValueError(f"SIM_HZ ({SIM_HZ}) must be divisible by CTRL_HZ ({CTRL_HZ})")
CTRL_DECIM = SIM_HZ // CTRL_HZ

SIM_STEPS = int(RUN_SIM_LENGTH_S * SIM_HZ)
CTRL_STEPS = int(RUN_SIM_LENGTH_S * CTRL_HZ)

# MPC loop rate
MPC_DT = GAIT_T / 16
MPC_HZ = 1.0 / MPC_DT
STEPS_PER_MPC = max(1, int(CTRL_HZ // MPC_HZ))  

# TEMPORARY GOD MODE FOR DEBUGGING ONLY
SAFETY = 1.0
HIP_LIM = 300.0       # Was 88.0
HIP_ROLL_LIM = 300.0  # Was 139.0
KNEE_LIM = 300.0      # Was 139.0
ANKLE_LIM = 100.0     # Was 50.0


TAU_LIM = SAFETY * np.array([
    HIP_LIM, HIP_ROLL_LIM, HIP_LIM, KNEE_LIM, ANKLE_LIM, ANKLE_LIM, # Left Leg
    HIP_LIM, HIP_ROLL_LIM, HIP_LIM, KNEE_LIM, ANKLE_LIM, ANKLE_LIM  # Right Leg
])

# Biped Leg Slices
LEG_SLICE = {
    "LEFT": slice(0, 6),
    "RIGHT": slice(6, 12),
}

# --------------------------------------------------------------------------------
# Helper Function
# --------------------------------------------------------------------------------
def get_body_cmd(t: float):
    for phase in CMD_SCHEDULE:
        if phase.t_start <= t < phase.t_end:
            return phase.x_vel, phase.y_vel, phase.z_pos, phase.yaw_rate
    return 0.0, 0.0, NOMINAL_Z, 0.0

# --------------------------------------------------------------------------------
# Storage Variables (CONTROL-rate logs for plots)
# --------------------------------------------------------------------------------

x_vec = np.zeros((12, CTRL_STEPS))
mpc_force_world = np.zeros((12, CTRL_STEPS))
tau_raw = np.zeros((12, CTRL_STEPS))
tau_cmd = np.zeros((12, CTRL_STEPS))

time_log_ctrl_s = np.zeros(CTRL_STEPS)
q_log_ctrl = np.zeros((CTRL_STEPS, 50)) # 50-DoF array for G1 with hands
tau_log_ctrl_Nm = np.zeros((CTRL_STEPS, 12))

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
# Simulation Initialization
# --------------------------------------------------------------------------------

g1 = PinG1Model()
mujoco_g1 = MuJoCo_G1_Model()
leg_controller = LegController()
traj = ComTraj(g1)
gait = Gait(GAIT_HZ, GAIT_DUTY)

# Initialize robot configuration
q_init, _ = g1.get_full_q_dq()
q_init[0], q_init[1] = INITIAL_X_POS, INITIAL_Y_POS
mujoco_g1.update_with_q_pin(q_init)

mujoco_g1.model.opt.timestep = SIM_DT

# Initialize MPC
traj.generate_traj(
    g1, gait, 0.0,
    x_vel_des_body, y_vel_des_body, z_pos_des_body, yaw_rate_des_body,
    time_step=MPC_DT,
)
mpc = CentroidalMPC(g1, traj)

U_opt = np.zeros((12, traj.N), dtype=float)

# --------------------------------------------------------------------------------
# Live Simulation Loop
# --------------------------------------------------------------------------------
print(f"Running Biped Simulation for {RUN_SIM_LENGTH_S}s")
sim_start_time = time.perf_counter()

ctrl_i = 0
tau_hold = np.zeros(12, dtype=float)

# Launch the live viewer
with mjv.launch_passive(mujoco_g1.model, mujoco_g1.data) as viewer:
    
    # Configure camera to track the robot
    viewer.cam.type = mj.mjtCamera.mjCAMERA_TRACKING
    viewer.cam.trackbodyid = mujoco_g1.base_bid
    viewer.cam.distance = 2.5
    viewer.cam.elevation = -20
    viewer.cam.azimuth = 90
    viewer.opt.flags[mj.mjtVisFlag.mjVIS_CONTACTPOINT] = True

    for k in range(SIM_STEPS):
        # Break if the user closes the viewer window early
        if not viewer.is_running():
            break

        time_now_s = float(mujoco_g1.data.time)

        # Control update at CTRL_HZ
        if (k % CTRL_DECIM) == 0 and ctrl_i < CTRL_STEPS:
            x_vel_des_body, y_vel_des_body, z_pos_des_body, yaw_rate_des_body = get_body_cmd(time_now_s)

            # Sync Models
            mujoco_g1.update_pin_with_mujoco(g1)
            x_vec[:, ctrl_i] = g1.compute_com_x_vec().reshape(-1)

            time_log_ctrl_s[ctrl_i] = time_now_s
            q_log_ctrl[ctrl_i, :] = mujoco_g1.data.qpos

            # Update MPC
            if (ctrl_i % STEPS_PER_MPC) == 0:
                print(f"\rSimulation Time: {time_now_s:.3f} s", end="", flush=True)

                traj.generate_traj(
                    g1, gait, time_now_s,
                    x_vel_des_body, y_vel_des_body, z_pos_des_body, yaw_rate_des_body,
                    time_step=MPC_DT,
                )

                sol = mpc.solve_QP(g1, traj, False)
                mpc_solve_time_ms.append(mpc.solve_time)
                mpc_update_time_ms.append(mpc.update_time)

                N = traj.N
                w_opt = sol["x"].full().flatten()
                U_opt = w_opt[12 * (N) :].reshape((12, N), order="F")

            # Extract first 6D Wrench for both legs from MPC
            mpc_force_world[:, ctrl_i] = U_opt[:, 0]

            # Compute joint torques for LEFT leg
            LEFT = leg_controller.compute_leg_torque(
                "LEFT", g1, gait, mpc_force_world[LEG_SLICE["LEFT"], ctrl_i], time_now_s
            )
            tau_raw[LEG_SLICE["LEFT"], ctrl_i] = LEFT.tau
            foot_traj.pos_des[LEG_SLICE["LEFT"], ctrl_i] = np.pad(LEFT.pos_des, (0,3))
            foot_traj.pos_now[LEG_SLICE["LEFT"], ctrl_i] = np.pad(LEFT.pos_now, (0,3))
            foot_traj.vel_des[LEG_SLICE["LEFT"], ctrl_i] = np.pad(LEFT.vel_des, (0,3))
            foot_traj.vel_now[LEG_SLICE["LEFT"], ctrl_i] = np.pad(LEFT.vel_now, (0,3))

            # Compute joint torques for RIGHT leg
            RIGHT = leg_controller.compute_leg_torque(
                "RIGHT", g1, gait, mpc_force_world[LEG_SLICE["RIGHT"], ctrl_i], time_now_s
            )
            tau_raw[LEG_SLICE["RIGHT"], ctrl_i] = RIGHT.tau
            foot_traj.pos_des[LEG_SLICE["RIGHT"], ctrl_i] = np.pad(RIGHT.pos_des, (0,3))
            foot_traj.pos_now[LEG_SLICE["RIGHT"], ctrl_i] = np.pad(RIGHT.pos_now, (0,3))
            foot_traj.vel_des[LEG_SLICE["RIGHT"], ctrl_i] = np.pad(RIGHT.vel_des, (0,3))
            foot_traj.vel_now[LEG_SLICE["RIGHT"], ctrl_i] = np.pad(RIGHT.vel_now, (0,3))

            # Saturate + hold
            tau_cmd[:, ctrl_i] = np.clip(tau_raw[:, ctrl_i], -TAU_LIM, TAU_LIM)
            tau_hold = tau_cmd[:, ctrl_i].copy()

            tau_log_ctrl_Nm[ctrl_i, :] = tau_hold
            ctrl_i += 1

        # Apply held torques at every SIM step
        mj.mj_step1(mujoco_g1.model, mujoco_g1.data)
      
        # l_shoulder_id = mj.mj_name2id(mujoco_g1.model, mj.mjtObj.mjOBJ_ACTUATOR, "left_shoulder_pitch_joint")
        # r_shoulder_id = mj.mj_name2id(mujoco_g1.model, mj.mjtObj.mjOBJ_ACTUATOR, "right_shoulder_pitch_joint")
        
        # arm_amplitude = 0.4 if x_vel_des_body > 0.1 else 0.0 
        
        # Add a phase offset to align the arms with the opposite legs.
        # Note: Depending on whether your gait.py starts on the Left or Right foot at t=0,
        # you might need to change this to np.pi/2 or 0.0 to get it perfectly synced!
        PHASE_OFFSET = np.pi  
        
        # Calculate the synchronized swing
        # arm_swing = arm_amplitude * np.sin(2 * np.pi * GAIT_HZ * time_now_s + PHASE_OFFSET)
        
        # # In the G1, opposite arms must swing in opposite directions
        # mujoco_g1.data.ctrl[l_shoulder_id] = arm_swing
        # mujoco_g1.data.ctrl[r_shoulder_id] = -arm_swing
        # ------------------------------------------------
        mujoco_g1.set_joint_torque(tau_hold)
        mj.mj_step2(mujoco_g1.model, mujoco_g1.data)

        # Render-rate sync and real-time pacing
        if k % int(SIM_HZ / RENDER_HZ) == 0:
            viewer.sync()
            time_until_next_step = mujoco_g1.data.time - (time.perf_counter() - sim_start_time)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

sim_end_time = time.perf_counter()
print(
    f"\nSimulation ended."
    f"\nElapsed time: {sim_end_time - sim_start_time:.3f}s"
    f"\nControl ticks: {ctrl_i}/{CTRL_STEPS}"
)

# --------------------------------------------------------------------------------
# Simulation Results
# --------------------------------------------------------------------------------

t_vec = np.arange(ctrl_i) * CTRL_DT

# Spawn graphs AFTER the viewer closes or the simulation finishes
plot_swing_foot_traj(t_vec, foot_traj.pos_now, foot_traj.pos_des, foot_traj.vel_now, foot_traj.vel_des, block=False)
plot_mpc_result(t_vec, mpc_force_world, tau_cmd, x_vec, block=False)
plot_solve_time(mpc_solve_time_ms, mpc_update_time_ms, MPC_DT, MPC_HZ, block=True)