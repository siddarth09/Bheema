import mujoco
import mujoco.viewer
import numpy as np
import time
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import your Bheema modules
from bheema.state import G1State
from bheema.mpc import CentroidalMPC
from bheema.wbc import WholeBodyController
from bheema.gait_scheduler import BipedGaitScheduler
from bheema.hlip_planner import HLIPFootPlanner
from bheema.com_ref import ComReference
from bheema.centroidal_model import CentroidalDynamics, CentroidalDiscrete

def main():
    # 1. Load MuJoCo Model
    xml_path = "/home/sid/projects25/src/bheema/unitree_g1/scene_with_hands.xml"
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    
    # 2. Initialize Bheema Modules
    print("Initializing Physics Modules...")
    state_est = G1State() 
    
    # --- CHANGE 1: Get the True Inertia ---
    # We must update the state once so Pinocchio computes the composite rigid body inertia
    state_est.update()
    mass = state_est.get_mass() 
    I_body = state_est.data.Ig.inertia # 3x3 Inertia tensor at the CoM
    
    com_height = 0.76

    # Control Rates
    wbc_freq = 500.0  
    mpc_freq = 50.0
    dt_wbc = 1.0 / wbc_freq
    dt_mpc = 1.0 / mpc_freq
    model.opt.timestep = dt_wbc 

    mpc_decimation = int(wbc_freq / mpc_freq)
    N_horizon = 20

    mpc = CentroidalMPC(N=N_horizon, dt=dt_mpc, mass=mass)
    wbc = WholeBodyController(state_est)
    scheduler = BipedGaitScheduler(step_time=0.3, double_support_time=0.05)
    planner = HLIPFootPlanner(com_height=com_height)
    ref_gen = ComReference(N_horizon, dt_mpc)
    
    # Pass the real inertia instead of np.eye(3)
    dyn = CentroidalDynamics(mass, I_body)
    disc = CentroidalDiscrete(dt_mpc)

    planner.get_foot_positions()

    u_mpc = np.zeros(6)
    last_support = None
    tick_count = 0

    print("--- JOINT MAPPING CHECK ---")
    print("Pinocchio Joints (ignoring universe & floating base):")
    pin_joints = [state_est.model.names[i] for i in range(2, state_est.model.njoints)]
    print(pin_joints)

    print("\nMuJoCo Actuators:")
    mj_actuators = [model.actuator(i).name for i in range(model.nu)]
    print(mj_actuators)
    print("---------------------------")
    print("Launching MuJoCo Viewer...")

    pin_joints = [state_est.model.names[i] for i in range(2, state_est.model.njoints)]
    mj_actuators = [model.actuator(i).name for i in range(model.nu)]
    
    torque_map = []
    for name in mj_actuators:
        if name in pin_joints:
            torque_map.append(pin_joints.index(name))
        else:
            print(f"[ERROR] MuJoCo actuator '{name}' not found in URDF!")
            torque_map.append(0) # Safe fallback to avoid crash

    # Telemetry trackers
    wbc_times = []
    mpc_times = []

    with mujoco.viewer.launch_passive(model, data) as viewer:
        
        time.sleep(1.0)
        
        while viewer.is_running():
            step_start = time.time()
            t = data.time

            # --- A. State Estimation ---
            pos = data.qpos[0:3]
            quat_mj = data.qpos[3:7] 
            quat_pin = np.array([quat_mj[1], quat_mj[2], quat_mj[3], quat_mj[0]]) 
            state_est.set_base_pose(pos, quat_pin)

            v_lin_body = data.qvel[0:3]
            v_ang_body = data.qvel[3:6]
            
            R_b2w = state_est.robot.framePlacement(state_est.q, state_est.base_frame).rotation
            v_lin_world = R_b2w @ v_lin_body
            v_ang_world = R_b2w @ v_ang_body
            state_est.set_base_velocity(v_lin_world, v_ang_world)

            q_joints = data.qpos[7:]
            dq_joints = data.qvel[6:]
            state_est.set_joint_state(q_joints, dq_joints)

            state_est.update()
            x_curr = state_est.get_centroidal_state()

            # --- B. Run MPC ---
            
            # 1. THE GRACE PERIOD: Let the robot drop and settle for 0.5s
            if t < 0.5:
                half_weight = mass * 9.81 / 2.0
                u_mpc = np.array([0.0, 0.0, half_weight, 0.0, 0.0, half_weight])
            
            # 2. THE BALANCE TEST
            elif tick_count % mpc_decimation == 0:
                t0_mpc = time.time()
                
                contact_table = np.ones((2, N_horizon), dtype=int) 
                v_cmd = np.array([0.0, 0.0, 0.0]) 
                x_ref = ref_gen.generate(x_curr, v_cmd, com_height)

                pL_real = state_est.data.oMf[state_est.left_foot_frame].translation
                pR_real = state_est.data.oMf[state_est.right_foot_frame].translation

                Ad_seq, Bd_seq, gd_seq = [], [], []
                
                for k in range(N_horizon):
                    pc = x_ref[0:3, k]
                    yaw_k = x_ref[5, k] 
                    
                    rL = pL_real - pc
                    rR = pR_real - pc
                    
                    Ac, Bc, gc = dyn.continuous_matrices(rL, rR, yaw=yaw_k)
                    Ad, Bd, gd = disc.discretize(Ac, Bc, gc)
                    Ad_seq.append(Ad); Bd_seq.append(Bd); gd_seq.append(gd)

                try:
                    u_mpc = mpc.solve(Ad_seq, Bd_seq, gd_seq, x_curr, x_ref, contact_table)
                except Exception as e:
                    u_mpc = np.zeros(6)
                
                mpc_times.append(time.time() - t0_mpc)

            # --- C. Run WBC ---
            contact_now = (1, 1) 
            swing_data = None
            
            tau_full = wbc.compute_torques(
                q=state_est.q,
                dq=state_est.dq,
                f_mpc=u_mpc,
                swing_target=swing_data
            )

            # --- D. Apply Torques ---
            pin_torques = tau_full[6:] 
            mapped_torques = np.array([pin_torques[idx] for idx in torque_map])
            
            try:
                data.ctrl[:] = mapped_torques 
            except ValueError as e:
                print(f"Actuator mismatch: MuJoCo expects {model.nu} actuators, but we gave {len(mapped_torques)}.")
                raise e

            mujoco.mj_step(model, data)
            viewer.sync()
            
            # --- TELEMETRY & PACING ---
            wbc_times.append(time.time() - step_start)
            
            # Print logs every 0.5 simulated seconds
            if tick_count % int(wbc_freq / 2) == 0:
                avg_wbc = 1.0 / np.mean(wbc_times) if len(wbc_times) > 0 else 0
                avg_mpc_time = np.mean(mpc_times) * 1000 if len(mpc_times) > 0 else 0.0
                
                print(f"\n[t={t:.2f}s] Freq: WBC={avg_wbc:.0f}Hz | MPC Solve Avg={avg_mpc_time:.1f}ms")
                print(f"          CoM Z: {x_curr[2,0]:.3f}m | Ref Z: {com_height}m")
                print(f"          Forces (L/R): {u_mpc[2]:.1f}N / {u_mpc[5]:.1f}N")
                
                if np.all(u_mpc == 0) and t > 0.5:
                    print("          [FATAL ERROR] MPC returned exactly ZERO forces. Solver failed or crashed!")
                
                wbc_times.clear()
                mpc_times.clear()

            tick_count += 1

            time_until_next_step = dt_wbc - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
            mujoco.mj_step(model, data)
            viewer.sync()
            tick_count += 1

            time_until_next_step = dt_wbc - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

if __name__ == '__main__':
    main()