import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def plot_contact_forces(U_opt, contact_mask, dt, block, leg_names=("LEFT", "RIGHT")):
    """
    Plots the 6D Wrenches (Fx, Fy, Fz, Tx, Ty, Tz) for the two bipedal feet.
    """
    assert U_opt.shape[0] == 12 # 2 legs * 6D wrenches
    N = U_opt.shape[1]
    t_edges = np.linspace(0, N*dt, N+1)

    # 4 Subplots: Left Forces, Left Torques, Right Forces, Right Torques
    fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
    
    for leg_idx in range(2):
        base = 6 * leg_idx
        fx, fy, fz = U_opt[base, :], U_opt[base+1, :], U_opt[base+2, :]
        tx, ty, tz = U_opt[base+3, :], U_opt[base+4, :], U_opt[base+5, :]
        
        # Plot Forces
        ax_f = axes[leg_idx * 2]
        ax_f.stairs(fx, t_edges, label="fx")
        ax_f.stairs(fy, t_edges, label="fy")
        ax_f.stairs(fz, t_edges, label="fz", linewidth=2)
        ax_f.set_ylabel(f"{leg_names[leg_idx]} Forces [N]")
        ax_f.grid(True, alpha=0.3)
        ax_f.legend(loc="upper right", ncols=3, fontsize=9)
        
        # Plot Torques (Center of Pressure moments)
        ax_t = axes[leg_idx * 2 + 1]
        ax_t.stairs(tx, t_edges, label="tx (Roll)")
        ax_t.stairs(ty, t_edges, label="ty (Pitch)")
        ax_t.stairs(tz, t_edges, label="tz (Yaw)", linewidth=2)
        ax_t.set_ylabel(f"{leg_names[leg_idx]} Torques [Nm]")
        ax_t.grid(True, alpha=0.3)
        ax_t.legend(loc="upper right", ncols=3, fontsize=9)

        # Shade swing intervals (mask==0)
        swing = (contact_mask[leg_idx] == 0)
        for k in np.flatnonzero(swing):
            ax_f.axvspan(t_edges[k], t_edges[k+1], alpha=0.15, hatch='//', edgecolor='none')
            ax_t.axvspan(t_edges[k], t_edges[k+1], alpha=0.15, hatch='//', edgecolor='none')

    axes[-1].set_xlabel("Time [s]")
    fig.suptitle("Biped Leg Contact Wrenches (One Gait Cycle)")
    plt.tight_layout()
    plt.show(block=block)
    plt.pause(0.001)

def plot_traj_tracking(pos_traj_ref, pos_traj_sim, block):
    # --- 3D plot ---
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.scatter(pos_traj_ref[0,0], pos_traj_ref[1,0], pos_traj_ref[2,0], label="Initial Position", color='k', s=50)
    ax.scatter(pos_traj_ref[0,:], pos_traj_ref[1,:], pos_traj_ref[2,:], color='b', marker='.', label="Reference Trajectory")
    ax.plot(pos_traj_sim[0,:], pos_traj_sim[1,:], pos_traj_sim[2,:], 'g-', linewidth=2, label="Actual Trajectory")

    ax.set_title("3D CoM Trajectory", fontsize=13)
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")

    # --- Auto-zoom with equal scale ---
    data = np.hstack([pos_traj_ref, pos_traj_sim])   
    mins, maxs = data.min(axis=1), data.max(axis=1)
    ctr = (mins + maxs) / 2.0
    half = (maxs - mins).max() / 2.0                 
    r = max(half * 1.05, 1e-9)                     

    ax.set_xlim(ctr[0]-r, ctr[0]+r)
    ax.set_ylim(ctr[1]-r, ctr[1]+r)
    ax.set_zlim(ctr[2]-r, ctr[2]+r)
    ax.set_box_aspect([1, 1, 1])                     

    ax.grid(True)
    ax.legend(loc="best")
    plt.show(block=block)
    plt.pause(0.001)

def plot_mpc_result(t_vec, wrench, tau, x_vec, block):
    """
    wrench: (12, N) [Left Fxyz Txyz, Right Fxyz Txyz]
    tau: (12, N) [Left 6 joints, Right 6 joints]
    x_vec: (12, N) [Pos, Euler, Vel, Omega]
    """
    fig, axes = plt.subplots(4, 3, figsize=(16, 11), constrained_layout=True)

    # --- COLUMN 1: WRENCHES ---
    axes[0,0].step(t_vec, wrench[0, :], label='fx'); axes[0,0].step(t_vec, wrench[1, :], label='fy'); axes[0,0].step(t_vec, wrench[2, :], label='fz')
    axes[0,0].set_title("Left Foot Forces (N)"); axes[0,0].legend(); axes[0,0].grid(True)

    axes[1,0].step(t_vec, wrench[3, :], label='tx'); axes[1,0].step(t_vec, wrench[4, :], label='ty'); axes[1,0].step(t_vec, wrench[5, :], label='tz')
    axes[1,0].set_title("Left Foot Torques (Nm)"); axes[1,0].legend(); axes[1,0].grid(True)

    axes[2,0].step(t_vec, wrench[6, :], label='fx'); axes[2,0].step(t_vec, wrench[7, :], label='fy'); axes[2,0].step(t_vec, wrench[8, :], label='fz')
    axes[2,0].set_title("Right Foot Forces (N)"); axes[2,0].legend(); axes[2,0].grid(True)

    axes[3,0].step(t_vec, wrench[9, :], label='tx'); axes[3,0].step(t_vec, wrench[10, :], label='ty'); axes[3,0].step(t_vec, wrench[11, :], label='tz')
    axes[3,0].set_title("Right Foot Torques (Nm)"); axes[3,0].legend(); axes[3,0].grid(True)

    # --- COLUMN 2: JOINT TORQUES ---
    # Split the 6 leg joints into two plots per leg for readability
    axes[0,1].step(t_vec, tau[0, :], label='Hip Pitch'); axes[0,1].step(t_vec, tau[1, :], label='Hip Roll'); axes[0,1].step(t_vec, tau[2, :], label='Hip Yaw')
    axes[0,1].set_title("Left Leg Torques - Hips (Nm)"); axes[0,1].legend(); axes[0,1].grid(True)

    axes[1,1].step(t_vec, tau[3, :], label='Knee'); axes[1,1].step(t_vec, tau[4, :], label='Ankle Pitch'); axes[1,1].step(t_vec, tau[5, :], label='Ankle Roll')
    axes[1,1].set_title("Left Leg Torques - Knee/Ankle (Nm)"); axes[1,1].legend(); axes[1,1].grid(True)

    axes[2,1].step(t_vec, tau[6, :], label='Hip Pitch'); axes[2,1].step(t_vec, tau[7, :], label='Hip Roll'); axes[2,1].step(t_vec, tau[8, :], label='Hip Yaw')
    axes[2,1].set_title("Right Leg Torques - Hips (Nm)"); axes[2,1].legend(); axes[2,1].grid(True)

    axes[3,1].step(t_vec, tau[9, :], label='Knee'); axes[3,1].step(t_vec, tau[10, :], label='Ankle Pitch'); axes[3,1].step(t_vec, tau[11, :], label='Ankle Roll')
    axes[3,1].set_title("Right Leg Torques - Knee/Ankle (Nm)"); axes[3,1].legend(); axes[3,1].grid(True)

    # --- COLUMN 3: COM STATE ---
    axes[0,2].step(t_vec, x_vec[0, :], label='x'); axes[0,2].step(t_vec, x_vec[1, :], label='y'); axes[0,2].step(t_vec, x_vec[2, :], label='z')
    axes[0,2].set_title("CoM Position (m)"); axes[0,2].legend(); axes[0,2].grid(True)

    axes[1,2].step(t_vec, x_vec[3, :], label='roll'); axes[1,2].step(t_vec, x_vec[4, :], label='pitch'); axes[1,2].step(t_vec, x_vec[5, :], label='yaw')
    axes[1,2].set_title("Base ZYX Euler (rad)"); axes[1,2].legend(); axes[1,2].grid(True)

    axes[2,2].step(t_vec, x_vec[6, :], label='vx'); axes[2,2].step(t_vec, x_vec[7, :], label='vy'); axes[2,2].step(t_vec, x_vec[8, :], label='vz')
    axes[2,2].set_title("CoM Linear Velocity (m/s)"); axes[2,2].legend(); axes[2,2].grid(True)

    axes[3,2].step(t_vec, x_vec[9, :], label='wx'); axes[3,2].step(t_vec, x_vec[10, :], label='wy'); axes[3,2].step(t_vec, x_vec[11, :], label='wz')
    axes[3,2].set_title("Base Angular Velocity (rad/s)"); axes[3,2].legend(); axes[3,2].grid(True)

    plt.show(block=block)
    plt.pause(0.001)

def plot_swing_foot_traj(t_vec, pos_now, pos_des, vel_now, vel_des, block, leg_name="Left"):
    """
    Expects (3, N) arrays for pos and vel.
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), constrained_layout=True)
    plt.suptitle(f"{leg_name} Foot Swing Trajectory")

    axis = axes[0]
    axis.plot(t_vec, pos_now[0,:], color='r', label="Actual x")
    axis.plot(t_vec, pos_now[1,:], color='g', label="Actual y")
    axis.plot(t_vec, pos_now[2,:], color='b', label="Actual z")
    axis.plot(t_vec, pos_des[0,:], color='r', linestyle=':', linewidth=2.5, label="Desired x")
    axis.plot(t_vec, pos_des[1,:], color='g', linestyle=':', linewidth=2.5, label="Desired y")
    axis.plot(t_vec, pos_des[2,:], color='b', linestyle=':', linewidth=2.5, label="Desired z")
    axis.legend(loc="upper right", ncols=2)
    axis.grid(True)

    axis = axes[1]
    axis.plot(t_vec, vel_now[0,:], color='r', label="Actual vx")
    axis.plot(t_vec, vel_now[1,:], color='g', label="Actual vy")
    axis.plot(t_vec, vel_now[2,:], color='b', label="Actual vz")
    axis.plot(t_vec, vel_des[0,:], color='r', linestyle=':', label="Desired vx")
    axis.plot(t_vec, vel_des[1,:], color='g', linestyle=':', label="Desired vy")
    axis.plot(t_vec, vel_des[2,:], color='b', linestyle=':', label="Desired vz")
    axis.legend(loc="upper right", ncols=2)
    axis.grid(True)

    plt.show(block=block)
    plt.pause(0.001)

def plot_solve_time(mpc_solve_time_ms, mpc_compute_time_ms, MPC_DT, MPC_HZ, block):
    fig, axis = plt.subplots(figsize=(10, 6))
    mpc_solve_time_ms = np.asarray(mpc_solve_time_ms)
    mpc_compute_time_ms = np.asarray(mpc_compute_time_ms)
    total_time_ms  = mpc_solve_time_ms + mpc_compute_time_ms
    avg_total_ms   = np.mean(total_time_ms)
    avg_solve_ms   = np.mean(mpc_solve_time_ms)
    avg_update_ms  = np.mean(mpc_compute_time_ms)
    iters = np.arange(len(mpc_solve_time_ms))  
    required_time_ms = MPC_DT * 1e3         

    axis.set_xlabel("MPC Step", fontweight='bold')
    axis.set_ylabel("Time (ms)", fontweight='bold')
    axis.bar(iters, mpc_compute_time_ms, label='Model Update Time (ms)')
    axis.bar(iters, mpc_solve_time_ms, bottom=mpc_compute_time_ms, label='QP Solve Time (ms)')
    axis.axhline(required_time_ms, linestyle='--', linewidth=2.0, color='r',
                 label=f'Real-Time Budget {MPC_HZ} Hz ({required_time_ms:.1f} ms)')
    axis.tick_params(axis='y')
    axis.set_ylim(bottom=0)

    text_str = (
        f"Avg Model Update: {avg_update_ms:.2f} ms\n"
        f"Avg QP Solve:  {avg_solve_ms:.2f} ms\n"
        f"Avg Total Cycle:  {avg_total_ms:.2f} ms"
    )
    axis.text(
        0.02, 0.85, text_str,
        transform=axis.transAxes,
        va='top', ha='left',
        bbox=dict(boxstyle="round", alpha=0.8, facecolor="white")
    )

    plt.title("MPC Iteration Latency")
    plt.tight_layout()
    plt.legend(loc="upper right")
    plt.show(block=block)
    plt.pause(0.001)

def plot_full_traj(traj_ref, x_sim, block):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    N = len(traj_ref[0,:])
    t_vec = range(N)

    axis = axes[0,0]
    axis.plot(t_vec, traj_ref[0,:], color='r', label="pos_x_des")
    axis.plot(t_vec, traj_ref[1,:], color='g', label="pos_y_des")
    axis.plot(t_vec, traj_ref[2,:], color='b', label="pos_z_des")
    axis.plot(t_vec, x_sim[0,:], color='r', linestyle=':', linewidth=2.5, label="pos_x_sim")
    axis.plot(t_vec, x_sim[1,:], color='g', linestyle=':', linewidth=2.5, label="pos_y_sim")
    axis.plot(t_vec, x_sim[2,:], color='b', linestyle=':', linewidth=2.5, label="pos_z_sim")
    axis.set_title("CoM Position")
    axis.legend(); axis.grid(True)

    axis = axes[1,0]
    axis.plot(t_vec, traj_ref[3,:], color='r', label="roll_des")
    axis.plot(t_vec, traj_ref[4,:], color='g', label="pitch_des")
    axis.plot(t_vec, traj_ref[5,:], color='b', label="yaw_des")
    axis.plot(t_vec, x_sim[3,:], color='r', linestyle=':', label="roll_sim")
    axis.plot(t_vec, x_sim[4,:], color='g', linestyle=':', label="pitch_sim")
    axis.plot(t_vec, x_sim[5,:], color='b', linestyle=':', label="yaw_sim")
    axis.set_title("Base Euler Angles")
    axis.legend(); axis.grid(True)

    axis = axes[0,1]
    axis.plot(t_vec, traj_ref[6,:], color='r', label="vel_x_des")
    axis.plot(t_vec, traj_ref[7,:], color='g', label="vel_y_des")
    axis.plot(t_vec, traj_ref[8,:], color='b', label="vel_z_des")
    axis.plot(t_vec, x_sim[6,:], color='r', linestyle=':', linewidth=2.5, label="vel_x_sim")
    axis.plot(t_vec, x_sim[7,:], color='g', linestyle=':', linewidth=2.5, label="vel_y_sim")
    axis.plot(t_vec, x_sim[8,:], color='b', linestyle=':', linewidth=2.5, label="vel_z_sim")
    axis.set_title("CoM Linear Velocity")
    axis.legend(); axis.grid(True)

    axis = axes[1,1]
    axis.plot(t_vec, traj_ref[9,:], color='r', label="wx_des")
    axis.plot(t_vec, traj_ref[10,:], color='g', label="wy_des")
    axis.plot(t_vec, traj_ref[11,:], color='b', label="wz_des")
    axis.plot(t_vec, x_sim[9,:], color='r', linestyle=':', label="wx_sim")
    axis.plot(t_vec, x_sim[10,:], color='g', linestyle=':', label="wy_sim")
    axis.plot(t_vec, x_sim[11,:], color='b', linestyle=':', label="wz_sim")
    axis.set_title("Base Angular Velocity")
    axis.legend(); axis.grid(True)

    plt.show(block=block)
    plt.pause(0.001)



def preview_footsteps(traj):
    """Generates a professional 2D top-down view of the planned CoM and Footsteps."""
    fig, ax = plt.subplots(figsize=(10, 3)) # Wide aspect ratio like your image
    
    com_x = traj.pos_traj_world[0, :]
    com_y = traj.pos_traj_world[1, :]
    
    # G1 Foot Dimensions (Adjust these if your URDF differs)
    FOOT_L = 0.20  # length in meters
    FOOT_W = 0.10  # width in meters

    # Calculate absolute world positions for both feet
    l_x_abs = com_x + traj.r_l_foot_world[0, :]
    l_y_abs = com_y + traj.r_l_foot_world[1, :]
    r_x_abs = com_x + traj.r_r_foot_world[0, :]
    r_y_abs = com_y + traj.r_r_foot_world[1, :]

    zmp_x, zmp_y = [], []

    def get_stance_blocks(contact_array):
        """Helper to find the start and end indices of when a foot is on the ground"""
        blocks = []
        in_stance = False
        start = 0
        for i in range(len(contact_array)):
            if contact_array[i] == 1 and not in_stance:
                start = i
                in_stance = True
            elif contact_array[i] == 0 and in_stance:
                blocks.append((start, i-1))
                in_stance = False
        if in_stance: blocks.append((start, len(contact_array)-1))
        return blocks

    # 1. Draw Left Feet (Grey Boxes)
    for (start, end) in get_stance_blocks(traj.contact_table[0, :]):
        mid = (start + end) // 2 # Grab the coordinate from the middle of the stance phase
        rect = patches.Rectangle((l_x_abs[mid] - FOOT_L/2, l_y_abs[mid] - FOOT_W/2), 
                                 FOOT_L, FOOT_W, facecolor='lightgrey', edgecolor='none')
        ax.add_patch(rect)

    # 2. Draw Right Feet (Grey Boxes)
    for (start, end) in get_stance_blocks(traj.contact_table[1, :]):
        mid = (start + end) // 2
        rect = patches.Rectangle((r_x_abs[mid] - FOOT_L/2, r_y_abs[mid] - FOOT_W/2), 
                                 FOOT_L, FOOT_W, facecolor='lightgrey', edgecolor='none')
        ax.add_patch(rect)

    # 3. Calculate the Red ZMP line (Averages the contact points)
    for i in range(traj.N):
        if traj.contact_table[0, i] == 1 and traj.contact_table[1, i] == 0:
            zmp_x.append(l_x_abs[i]); zmp_y.append(l_y_abs[i])
        elif traj.contact_table[1, i] == 1 and traj.contact_table[0, i] == 0:
            zmp_x.append(r_x_abs[i]); zmp_y.append(r_y_abs[i])
        else: # Double Support (Draws the line directly between both feet)
            zmp_x.append((l_x_abs[i] + r_x_abs[i])/2)
            zmp_y.append((l_y_abs[i] + r_y_abs[i])/2)

    # Plot the lines
    ax.plot(zmp_x, zmp_y, 'r-', label='Desired ZMP', linewidth=1.5)
    ax.plot(com_x, com_y, 'b-', label='Nominal CoM', linewidth=2)
    
    # Formatting
    ax.set_title("Top-Down Footstep Preview")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.axis('equal') # CRITICAL: This stops the boxes from stretching/warping!
    ax.legend(loc='upper left')
    
    plt.tight_layout()
    plt.show()


def hold_until_all_fig_closed():
    plt.show()