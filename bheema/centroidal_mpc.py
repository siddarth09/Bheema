import casadi as ca
import numpy as np
import scipy.sparse as sp
import time

from .com_traj import ComTraj
from .g1_config import PinG1Model
from .params import MPCParams, FootParams

# --------------------------------------------------------------------------------
# Model Predictive Control (BIPED)
# --------------------------------------------------------------------------------
# State vector: [x, y, z, roll, pitch, yaw, vx, vy, vz, wx, wy, wz]
# Input vector: 2 feet x 6D wrench [Fx, Fy, Fz, Tx, Ty, Tz]
#
# All weights, friction coefficients, foot geometry and solver settings now live in
# MPCParams / FootParams (see bheema/params.py) and arrive through the constructor.
NX = 12     # state dimension -- structural, not a tunable
NU = 12     # input dimension -- structural, not a tunable

N_INEQ_PER_LEG = 10   # friction pyramid (4) + CoP (4) + torsional friction (2)

SOLVER_NAME: str = "osqp"

class CentroidalMPC:
    def __init__(self, g1: PinG1Model, traj: ComTraj,
                 params: MPCParams | None = None,
                 foot: FootParams | None = None):
        self.p = params or MPCParams()
        self.foot = foot or FootParams()
        self.Q = np.diag(np.concatenate([self.p.q_pos, self.p.q_rpy,
                                         self.p.q_vel, self.p.q_omega]))
        self.R = np.diag(np.concatenate([self.p.r_force, self.p.r_torque,
                                         self.p.r_force, self.p.r_torque]))
        self.nvars = traj.N * NX + traj.N * NU #Number of total decision vars
        self.solve_time: float = 0 
        self.N = traj.N
        # Subdiagnoal shift matrix S
        self.I_block = ca.DM.eye(self.N * NX)
        ones_N_minus_1 = np.ones(self.N - 1)
        S_scipy = sp.kron(sp.diags([ones_N_minus_1], [-1]), sp.eye(NX)) # kronecker product 
        self.S_block = self._scipy_to_casadi(S_scipy)

        self.A_ineq_static = self._precompute_friction_and_cop_matrix(traj) #Static matrix since there are no dependency on robot states
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
        fz_min = self.p.fz_min
        fz_max = self.p.fz_max
        f_xy_max = self.p.f_xy_max
        N = traj.N      
        nvars = self.nvars
        start_u = N * 12

        lbx_np = np.full((nvars, 1), -np.inf, dtype=float)
        ubx_np = np.full((nvars, 1),  np.inf, dtype=float)

       
        # ---------------------------------------------------------
        # 2. CONTROL CONSTRAINTS (The variables after N*12)
        # ---------------------------------------------------------
        force_block = (np.arange(12)[:, None] + 12*np.arange(N)[None, :])  
        force_idx   = start_u + force_block                               

        contact = np.asarray(traj.contact_table, dtype=bool)  

        # --- A) Swing Legs
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
                lbx_np[fx_L_idx, 0] = -f_xy_max
                ubx_np[fx_L_idx, 0] =  f_xy_max
                lbx_np[fy_L_idx, 0] = -f_xy_max
                ubx_np[fy_L_idx, 0] =  f_xy_max
            if contact[1, i]:  # Right stance
                fx_R_idx = force_idx[6, i]
                fy_R_idx = force_idx[7, i]
                lbx_np[fx_R_idx, 0] = -f_xy_max
                ubx_np[fx_R_idx, 0] =  f_xy_max
                lbx_np[fy_R_idx, 0] = -f_xy_max
                ubx_np[fy_R_idx, 0] =  f_xy_max


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
        self.solver = ca.conic('S', SOLVER_NAME, qp, self._solver_opts())

    def _solver_opts(self) -> dict:
        p = self.p
        return {
            'warm_start_primal': True,
            'warm_start_dual': True,
            'error_on_fail': False,
            "osqp": {
                "eps_abs": p.osqp_eps_abs,
                "eps_rel": p.osqp_eps_rel,
                "max_iter": p.osqp_max_iter,
                "polish": p.osqp_polish,
                "verbose": False,
                'adaptive_rho': True,
                "check_termination": p.osqp_check_termination,
                'adaptive_rho_interval': p.osqp_adaptive_rho_interval,
                "scaling": p.osqp_scaling,
                "scaled_termination": True,
            },
        }

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
        n_ineq = 2 * N_INEQ_PER_LEG * self.N
        l_ineq = -ca.inf * ca.DM.ones(n_ineq, 1)
        
        u_ineq_np = np.inf * np.ones(n_ineq)
        ct = traj.contact_table
        
        idx = 0
        for k in range(self.N):
            for leg in range(2):
                if ct[leg, k] == 1: 
                    # STANCE: Enforce friction pyramid and CoP limits <= 0
                    u_ineq_np[idx:idx+N_INEQ_PER_LEG] = 0.0
                idx += N_INEQ_PER_LEG
        
        u_ineq = ca.DM(u_ineq_np)

        lb = ca.vertcat(beq, l_ineq)
        ub = ca.vertcat(beq, u_ineq)

        return g, A_dm, lb, ub
     
    def _assemble_A_matrix(self, Ad, Bd):
        big_minus_Ad, big_minus_Bd = self.dyn_builder(Ad, Bd)
        term_Ad = self.S_block @ big_minus_Ad
        A_eq = ca.horzcat(self.I_block + term_Ad, big_minus_Bd) #concatenates horizontally
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
        mu, mu_tau = self.p.mu, self.p.mu_tau
        lx_front, lx_back, ly = self.foot.lx_front, self.foot.lx_back, self.foot.ly
        
        for k in range(self.N):
            uk0 = baseU + k * NU
            for leg in range(2):
                # 6D Wrench mapping
                fx, fy, fz = 6*leg, 6*leg+1, 6*leg+2
                tx, ty, tz = 6*leg+3, 6*leg+4, 6*leg+5
                
                # --- LINEAR FRICTION PYRAMID ---
                # 1. fx - mu*fz <= 0
                rows.extend([r0, r0]); cols.extend([uk0+fx, uk0+fz]); vals.extend([1.0, -mu]); r0+=1
                # 2. -fx - mu*fz <= 0
                rows.extend([r0, r0]); cols.extend([uk0+fx, uk0+fz]); vals.extend([-1.0, -mu]); r0+=1
                # 3. fy - mu*fz <= 0
                rows.extend([r0, r0]); cols.extend([uk0+fy, uk0+fz]); vals.extend([1.0, -mu]); r0+=1
                # 4. -fy - mu*fz <= 0
                rows.extend([r0, r0]); cols.extend([uk0+fy, uk0+fz]); vals.extend([-1.0, -mu]); r0+=1

                # --- CENTER OF PRESSURE (ZMP) LIMITS ---
                # Ankle roll torque (tx) cannot exceed what the foot width (Y) can support
                # 5. tx - Ly*fz <= 0
                rows.extend([r0, r0]); cols.extend([uk0+tx, uk0+fz]); vals.extend([1.0, -ly]); r0+=1
                # 6. -tx - Ly*fz <= 0
                rows.extend([r0, r0]); cols.extend([uk0+tx, uk0+fz]); vals.extend([-1.0, -ly]); r0+=1
                
                # Ankle pitch torque (ty) cannot exceed what the foot length (X) can support
                # 7. ty - L_back * fz <= 0 (Max POSITIVE pitch torque happens when leaning BACK on the heel)
                rows.extend([r0, r0]); cols.extend([uk0+ty, uk0+fz]); vals.extend([1.0, -lx_back]); r0+=1
                
                # 8. -ty - L_front * fz <= 0 (Max NEGATIVE pitch torque happens when leaning FORWARD on the toes)
                rows.extend([r0, r0]); cols.extend([uk0+ty, uk0+fz]); vals.extend([-1.0, -lx_front]); r0+=1
                # --- TORSIONAL FRICTION ---
                # Yaw torque (tz) limited by normal force to prevent spinning in place
                # 9. tz - mu_tau*fz <= 0
                rows.extend([r0, r0]); cols.extend([uk0+tz, uk0+fz]); vals.extend([1.0, -mu_tau]); r0+=1
                # 10. -tz - mu_tau*fz <= 0
                rows.extend([r0, r0]); cols.extend([uk0+tz, uk0+fz]); vals.extend([-1.0, -mu_tau]); r0+=1

        A_sp = sp.csc_matrix((vals, (rows, cols)), shape=(r0, self.nvars))
        return self._scipy_to_casadi(A_sp)

    @staticmethod
    def _scipy_to_casadi(M):
        M = M.tocsc()
        return ca.DM(ca.Sparsity(M.shape[0], M.shape[1], M.indptr, M.indices), M.data)