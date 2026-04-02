import casadi as ca
import numpy as np
import scipy.sparse as sp
import time

from .com_traj import ComTraj 
from .g1_config import PinG1Model 

# --------------------------------------------------------------------------------
# Model Predictive Control Setting (BIPED)
# --------------------------------------------------------------------------------
# State Vector: [x, y, z, roll, pitch, yaw, vx, vy, vz, wx, wy, wz]
COST_MATRIX_Q = np.diag([
    500.0,  1000.0, 3000.0,    # x, y, z position
    3000.0, 5000.0, 400.0,    # roll, pitch, yaw  
    500.0,  1000.0,  200.0,    # vx, vy, vz
    100.0,   100.0,  100.0     # wx, wy, wz
])

COST_MATRIX_R = np.diag([
    1.0, 1.0, 1e-2,  10.0, 10.0, 10.0,   # Left:  Fx, Fy, Fz, Tx, Ty, Tz
    1.0, 1.0, 1e-2,  10.0, 10.0, 10.0    # Right: Fx, Fy, Fz, Tx, Ty, Tz
])

MU = 0.8            # Linear friction coefficient
MU_TAU = 0.1       # Torsional friction coefficient (yaw rotation of the foot)

# Foot dimensions for Center of Pressure (CoP) constraints
FOOT_LX = 0.12    # Half-length of the foot (m) (front/back)
FOOT_LY = 0.05     # Half-width of the foot (m) (left/right)
FOOT_LX_FRONT = 0.12  
FOOT_LX_BACK = 0.05
NX = 12     # State size (6-DOF 12 states)
NU = 12     # Input size (2 feet x 6D Wrenches: Fx, Fy, Fz, Tx, Ty, Tz)

OPTS = {
    'warm_start_primal': True,
    'warm_start_dual': True,
    'error_on_fail': False,
    "osqp": {
        "eps_abs": 1e-2,
        "eps_rel": 1e-2,
        "max_iter": 1400,
        "polish": False,
        "verbose": False,
        'adaptive_rho': True,
        "check_termination": 10,
        'adaptive_rho_interval': 25,
        "scaling": 5,
        "scaled_termination": True
    }
}

SOLVER_NAME: str = "osqp"

class CentroidalMPC:
    def __init__(self, g1: PinG1Model, traj: ComTraj):
        self.Q = COST_MATRIX_Q 
        self.R = COST_MATRIX_R 
        self.nvars = traj.N * NX + traj.N * NU
        self.solve_time: float = 0 
        self.N = traj.N

        self.I_block = ca.DM.eye(self.N * NX)
        ones_N_minus_1 = np.ones(self.N - 1)
        S_scipy = sp.kron(sp.diags([ones_N_minus_1], [-1]), sp.eye(NX))
        self.S_block = self._scipy_to_casadi(S_scipy)

        self.A_ineq_static = self._precompute_friction_and_cop_matrix(traj)
        self.dyn_builder = self._create_dynamics_function()

        self._build_sparse_matrix(traj, verbose=True)

    def solve_QP(self, g1: PinG1Model, traj: ComTraj, verbose: bool = False):
        t0 = time.perf_counter()

        [g, A, lba, uba] = self._update_sparse_matrix(traj) 
        [lbx, ubx] = self._compute_bounds(traj)             
        t1 = time.perf_counter()

        qp_args = {
            'h': self.H_const, 'g': g, 'a': A,
            'lba': lba, 'uba': uba, 'lbx': lbx, 'ubx': ubx
        }
        
        if hasattr(self, 'x_prev') and self.x_prev is not None:
            qp_args['x0'] = self.x_prev            
            qp_args['lam_x0'] = self.lam_x_prev    
            qp_args['lam_a0'] = self.lam_a_prev    

        sol = self.solver(**qp_args)
        t2 = time.perf_counter()

        t_compute = t1 - t0         
        t_solve   = t2 - t1         
        self.update_time = t_compute * 1e3
        self.solve_time = t_solve * 1e3

        self.x_prev = sol["x"]              
        self.lam_x_prev = sol["lam_x"]      
        self.lam_a_prev = sol["lam_a"]      

        if verbose:
            stats = self.solver.stats()
            print(f"[QP SOLVER] update matrix takes {t_compute*1e3:.3f} ms")
            print(f"[QP SOLVER] solver takes {t_solve*1e3:.3f} ms")
            print(f"[QP SOLVER] total time = {(t_compute + t_solve)*1e3:.3f} ms")
            print(f"[QP SOLVER] status: {stats.get('return_status')}")
        return sol

    def _compute_bounds(self, traj: ComTraj):
        fz_min = 23.0   # Minimum pressure to prevent foot slip
        fz_max = 1200.0 # Maximum allowed vertical push 
        N = traj.N      
        nvars = self.nvars
        start_u = N * 12

        lbx_np = np.full((nvars, 1), -np.inf, dtype=float)
        ubx_np = np.full((nvars, 1),  np.inf, dtype=float)

        # ---------------------------------------------------------
        # 1. STATE CONSTRAINTS (The first N*12 variables)
        # ---------------------------------------------------------
        # State block layout: [x, y, z, roll, pitch, yaw, vx, vy, vz, wx, wy, wz]
        # for i in range(N):
        #     roll_idx  = i * 12 + 3
        #     pitch_idx = i * 12 + 4
        #     yaw_idx   = i * 12 + 5
            
        #     # Lock Roll and Pitch to +/- 15 degrees (0.26 rad)
        #     # This prevents the torso from tipping over and causing Gimbal Lock
        #     lbx_np[roll_idx, 0] = -0.26
        #     ubx_np[roll_idx, 0] =  0.26
        #     lbx_np[pitch_idx, 0] = -0.26
        #     ubx_np[pitch_idx, 0] =  0.26

        #     # Lock Yaw to +/- 30 degrees (0.52 rad)
        #     # This prevents the 180-degree "Helicopter" spin
        #     lbx_np[yaw_idx, 0] = -0.52
        #     ubx_np[yaw_idx, 0] =  0.52
        
        # ---------------------------------------------------------
        # 2. CONTROL CONSTRAINTS (The variables after N*12)
        # ---------------------------------------------------------
        force_block = (np.arange(12)[:, None] + 12*np.arange(N)[None, :])  
        force_idx   = start_u + force_block                               

        contact = np.asarray(traj.contact_table, dtype=bool)  

        # --- A) Swing Legs: Zero out everything ---
        swing = ~contact
        mask_swing = np.zeros((12, N), dtype=bool)
        for i in range(2): # For each leg
            if i == 0: # Left
                mask_swing[0:6, :] = swing[0, :]
            else: # Right
                mask_swing[6:12, :] = swing[1, :]
        
        lbx_np[force_idx[mask_swing], 0] = 0.0
        ubx_np[force_idx[mask_swing], 0] = 0.0

        # --- B) Stance Legs: Ceiling on Vertical Force (fz) ---
        # Rows 2 (Left fz) and 8 (Right fz)
        for i in range(N):
            # Left Stance
            if contact[0, i]:
                fz_L_idx = force_idx[2, i]
                lbx_np[fz_L_idx, 0] = fz_min
                ubx_np[fz_L_idx, 0] = fz_max # Stop the "Nuclear Spikes"
            
            # Right Stance
            if contact[1, i]:
                fz_R_idx = force_idx[8, i]
                lbx_np[fz_R_idx, 0] = fz_min
                ubx_np[fz_R_idx, 0] = fz_max # Stop the "Nuclear Spikes"


        # --- C) Stance Legs: Cap Horizontal Forces (Fx, Fy) ---
        for i in range(N):
            if contact[0, i]:  # Left stance
                fx_L_idx = force_idx[0, i]
                fy_L_idx = force_idx[1, i]
                lbx_np[fx_L_idx, 0] = -400.0
                ubx_np[fx_L_idx, 0] =  400.0
                lbx_np[fy_L_idx, 0] = -400.0
                ubx_np[fy_L_idx, 0] =  400.0
            if contact[1, i]:  # Right stance
                fx_R_idx = force_idx[6, i]
                fy_R_idx = force_idx[7, i]
                lbx_np[fx_R_idx, 0] = -400.0
                ubx_np[fx_R_idx, 0] =  400.0
                lbx_np[fy_R_idx, 0] = -400.0
                ubx_np[fy_R_idx, 0] =  400.0


        return ca.DM(lbx_np), ca.DM(ubx_np)
    
    def _build_sparse_matrix(self, traj: ComTraj, verbose: bool = False):
        rows, cols, vals = [], [], []
        for k in range(self.N):
            base = k*NX
            for i in range(NX):
                if self.Q[i,i] != 0:
                    rows.append(base+i); cols.append(base+i); vals.append(2*self.Q[i,i])
        
        for k in range(self.N):
            base = self.N*NX + k*NU
            for i in range(NU):
                if self.R[i,i] != 0:
                    rows.append(base+i); cols.append(base+i); vals.append(2*self.R[i,i])
        
        self.H_const = ca.DM.triplet(rows, cols, ca.DM(vals), self.nvars, self.nvars)
        self.H_sp = self.H_const.sparsity()

        Ad_dm = ca.DM(traj.Ad)
        Bd_stacked_np = traj.Bd.reshape(self.N * NX, NU)
        Bd_seq_dm = ca.DM(Bd_stacked_np)
        
        A_init = self._assemble_A_matrix(Ad_dm, Bd_seq_dm)
        self.A_sp = A_init.sparsity()

        qp = {'h': self.H_sp, 'a': self.A_sp}
        self.solver = ca.conic('S', SOLVER_NAME, qp, OPTS)

    def _update_sparse_matrix(self, traj: ComTraj):
        Ad_dm = ca.DM(traj.Ad) 
        Bd_stacked_np = traj.Bd.reshape(self.N * NX, NU)
        Bd_seq_dm = ca.DM(Bd_stacked_np)

        A_dm = self._assemble_A_matrix(Ad_dm, Bd_seq_dm)

        Q_mat = ca.DM(self.Q)
        x_ref_np = traj.compute_x_ref_vec() 
        x_ref_dm = ca.DM(x_ref_np)
        gx_mat = -2 * (Q_mat @ x_ref_dm)
        g_x = ca.vec(gx_mat)
        g = ca.vertcat(g_x, ca.DM.zeros(self.N*NU, 1))

        x0 = ca.DM(traj.initial_x_vec)              
        gd = ca.DM(traj.gd)                         
        beq_first = Ad_dm @ x0 + gd                 
        beq_rest  = ca.repmat(gd, self.N-1, 1)
        beq = ca.vertcat(beq_first, beq_rest)

        # 10 constraints per leg * 2 legs * N horizon
        n_ineq = 2 * 10 * self.N
        l_ineq = -ca.inf * ca.DM.ones(n_ineq, 1)
        
        u_ineq_np = np.inf * np.ones(n_ineq)
        ct = traj.contact_table
        
        idx = 0
        for k in range(self.N):
            for leg in range(2):
                if ct[leg, k] == 1: 
                    # STANCE: Enforce friction pyramid and CoP limits <= 0
                    u_ineq_np[idx:idx+10] = 0.0
                idx += 10
        
        u_ineq = ca.DM(u_ineq_np)

        lb = ca.vertcat(beq, l_ineq)
        ub = ca.vertcat(beq, u_ineq)

        return g, A_dm, lb, ub
     
    def _assemble_A_matrix(self, Ad, Bd):
        big_minus_Ad, big_minus_Bd = self.dyn_builder(Ad, Bd)
        term_Ad = self.S_block @ big_minus_Ad
        A_eq = ca.horzcat(self.I_block + term_Ad, big_minus_Bd)
        A_total = ca.vertcat(A_eq, self.A_ineq_static)
        return A_total
    
    def _create_dynamics_function(self):
        Ad_sym = ca.SX.sym('Ad', NX, NX)
        Bd_seq_sym = ca.SX.sym('Bd_seq', self.N * NX, NU)
        
        list_Ad = [-Ad_sym] * self.N
        list_Bd = []

        for k in range(self.N):
            idx_start = k * NX
            idx_end   = (k + 1) * NX
            Bk = Bd_seq_sym[idx_start:idx_end, :]
            list_Bd.append(-Bk)
        
        big_Ad = ca.diagcat(*list_Ad)
        big_Bd = ca.diagcat(*list_Bd) 
        
        return ca.Function('dyn_builder', [Ad_sym, Bd_seq_sym], [big_Ad, big_Bd])

    def _precompute_friction_and_cop_matrix(self, traj):
        """
        Builds the Static Matrix for Linear Friction and Center of Pressure (CoP) constraints.
        BIPED: Each foot has 6 inputs. We constrain Forces (Friction) and Torques (CoP).
        """
        rows, cols, vals = [], [], []
        baseU = self.N * NX 
        r0 = 0
        
        for k in range(self.N):
            uk0 = baseU + k * NU
            for leg in range(2):
                # 6D Wrench mapping
                fx, fy, fz = 6*leg, 6*leg+1, 6*leg+2
                tx, ty, tz = 6*leg+3, 6*leg+4, 6*leg+5
                
                # --- LINEAR FRICTION PYRAMID ---
                # 1. fx - mu*fz <= 0
                rows.extend([r0, r0]); cols.extend([uk0+fx, uk0+fz]); vals.extend([1.0, -MU]); r0+=1
                # 2. -fx - mu*fz <= 0
                rows.extend([r0, r0]); cols.extend([uk0+fx, uk0+fz]); vals.extend([-1.0, -MU]); r0+=1
                # 3. fy - mu*fz <= 0
                rows.extend([r0, r0]); cols.extend([uk0+fy, uk0+fz]); vals.extend([1.0, -MU]); r0+=1
                # 4. -fy - mu*fz <= 0
                rows.extend([r0, r0]); cols.extend([uk0+fy, uk0+fz]); vals.extend([-1.0, -MU]); r0+=1

                # --- CENTER OF PRESSURE (ZMP) LIMITS ---
                # Ankle roll torque (tx) cannot exceed what the foot width (Y) can support
                # 5. tx - Ly*fz <= 0
                rows.extend([r0, r0]); cols.extend([uk0+tx, uk0+fz]); vals.extend([1.0, -FOOT_LY]); r0+=1
                # 6. -tx - Ly*fz <= 0
                rows.extend([r0, r0]); cols.extend([uk0+tx, uk0+fz]); vals.extend([-1.0, -FOOT_LY]); r0+=1
                
                # Ankle pitch torque (ty) cannot exceed what the foot length (X) can support
                # 7. ty - L_back * fz <= 0 (Max POSITIVE pitch torque happens when leaning BACK on the heel)
                rows.extend([r0, r0]); cols.extend([uk0+ty, uk0+fz]); vals.extend([1.0, -FOOT_LX_BACK]); r0+=1
                
                # 8. -ty - L_front * fz <= 0 (Max NEGATIVE pitch torque happens when leaning FORWARD on the toes)
                rows.extend([r0, r0]); cols.extend([uk0+ty, uk0+fz]); vals.extend([-1.0, -FOOT_LX_FRONT]); r0+=1
                # --- TORSIONAL FRICTION ---
                # Yaw torque (tz) limited by normal force to prevent spinning in place
                # 9. tz - mu_tau*fz <= 0
                rows.extend([r0, r0]); cols.extend([uk0+tz, uk0+fz]); vals.extend([1.0, -MU_TAU]); r0+=1
                # 10. -tz - mu_tau*fz <= 0
                rows.extend([r0, r0]); cols.extend([uk0+tz, uk0+fz]); vals.extend([-1.0, -MU_TAU]); r0+=1

        A_sp = sp.csc_matrix((vals, (rows, cols)), shape=(r0, self.nvars))
        return self._scipy_to_casadi(A_sp)

    @staticmethod
    def _scipy_to_casadi(M):
        M = M.tocsc()
        return ca.DM(ca.Sparsity(M.shape[0], M.shape[1], M.indptr, M.indices), M.data)