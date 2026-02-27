import numpy as np
# Import correctly from your package
from bheema.mpc import CentroidalMPC 
from bheema.centroidal_model import CentroidalDynamics, CentroidalDiscrete
from bheema.gait_scheduler import BipedGaitScheduler
from bheema.hlip_planner import HLIPFootPlanner
from bheema.com_ref import ComReference

def simulate_walking_test():
    print("\n=== TEST: Closed Loop Walking (Clean) ===")

    dt = 0.02
    N = 20
    sim_time = 3.0
    steps = int(sim_time / dt)
    
    mass = 40.0
    I = np.diag([2.0, 2.5, 1.2])
    com_height = 1.0

    # Modules
    dyn = CentroidalDynamics(mass, I)
    disc = CentroidalDiscrete(dt)
    mpc = CentroidalMPC(N=N, dt=dt, mass=mass)  # Using fixed class
    planner = HLIPFootPlanner(com_height=com_height)
    scheduler = BipedGaitScheduler(step_time=0.4, double_support_time=0.1)
    ref_gen = ComReference(N, dt)

    # State Init
    x = np.zeros((12, 1))
    x[0] = 0.0
    x[2] = com_height
    
    planner.get_foot_positions()
    p_foot_L = planner.pL.copy()
    p_foot_R = planner.pR.copy()

    # Targets
    v_target = np.array([0.4, 0.0, 0.0])
    
    history_x = []
    last_support = None

    for i in range(steps):
        t = i * dt
        
        # --- 1. Gait & Plan ---
        contact_table = scheduler.get_contact_table(t, dt, N)
        
        sL = contact_table[0,0]
        sR = contact_table[1,0]
        current_support = "left" if (sL and not sR) else "right" if (sR and not sL) else None
        
        if current_support and current_support != last_support:
            planner.update_step(current_support, x[0,0], x[6,0])
            p_foot_L = planner.pL.copy()
            p_foot_R = planner.pR.copy()
            print(f"[{t:.2f}s] Step -> {current_support}")
            
        last_support = current_support
        
        # --- 2. Reference ---
        # Ramp up to avoid jerk
        ramp = min(1.0, t/0.5)
        x_ref = ref_gen.generate(x, v_target*ramp, com_height)
        
        # --- 3. Dynamics Loop (Time-Varying) ---
        Ad_seq, Bd_seq, gd_seq = [], [], []
        
        for k in range(N):
            # Predict foot lever arms based on Ref COM
            p_c_k = x_ref[0:3, k]
            rL = p_foot_L - p_c_k
            rR = p_foot_R - p_c_k
            
            Ac, Bc, gc = dyn.continuous_matrices(rL, rR)
            Ad, Bd, gd = disc.discretize(Ac, Bc, gc)
            Ad_seq.append(Ad); Bd_seq.append(Bd); gd_seq.append(gd)
            
        # --- 4. Solve ---
        try:
            u0 = mpc.solve(Ad_seq, Bd_seq, gd_seq, x, x_ref, contact_table)
        except RuntimeError:
            print("Solver fail, stopping.")
            break
            
        u = u0.reshape(6,1)
        
        # --- 5. Integrate ---
        # Real dynamics use real COM state
        rL_real = p_foot_L - x[0:3, 0]
        rR_real = p_foot_R - x[0:3, 0]
        Ac, Bc, gc = dyn.continuous_matrices(rL_real, rR_real)
        Ad, Bd, gd = disc.discretize(Ac, Bc, gc)
        
        x = Ad @ x + Bd @ u + gd
        history_x.append(x[0,0])

    print(f"\nFinal X: {history_x[-1]:.3f} m")
    if history_x[-1] > 0.5:
        print("✔ SUCCESS")
    else:
        print("✘ FAILURE")

if __name__ == "__main__":
    simulate_walking_test()