import casadi as ca
import numpy as np
import scipy.sparse as sp
import time

class CentroidalMPC:
    """
    Convex MPC for Bheema.
    """

    def __init__(self, N: int, dt: float):

        self.N = N
        self.dt = dt
        self.nx = 12
        self.nu = 6

        self.nz = N * self.nx + N * self.nu

        # Cost weights
        self.Q = np.diag([
            50, 50, 200,   # Position
            100, 100, 100, # Orientation (Increased from 20) -> FIGHT TIPPING
            10, 10, 10,    # Linear Vel
            10, 10, 10     # Angular Vel (Increased from 2) -> FIGHT ROLLING
        ])
        
        # Force regularization
        self.R = 1e-4 * np.eye(self.nu)

        # Contact friction
        self.mu = 0.6
        self.fz_min = 10.0
        self.fz_max = 1500.0

        self.opts = {
            "osqp": {
                "verbose": False,
                "eps_abs": 1e-4,
                "eps_rel": 1e-4,
                "max_iter": 1000
            },
            "warm_start_primal": True,
            "warm_start_dual": True
        }

        # Initialize static matrices
        self._static_cost()
        self._static_shift_matrix()
        self._static_friction_matrix()
        self._initialized = False

    def _static_cost(self):
        """
        Builds the Hessian (H) for the QP cost function.
        H = block_diag(Q, ..., Q, R, ..., R)
        """
        rows, cols, vals = [], [], []

        # 1. State Cost (Q)
        for k in range(self.N):
            base = k * self.nx
            for i in range(self.nx):
                if self.Q[i, i] != 0:
                    rows.append(base + i)
                    cols.append(base + i)
                    vals.append(2 * self.Q[i, i])

        # 2. Control Cost (R)
        for k in range(self.N):
            base = self.N * self.nx + k * self.nu
            for i in range(self.nu):
                if self.R[i, i] != 0:
                    rows.append(base + i)
                    cols.append(base + i)
                    vals.append(2 * self.R[i, i])

        # Create Sparse Hessian
        self.H = ca.DM.triplet(
            rows, cols, ca.DM(vals),
            self.nz, self.nz
        )
        self.H_sp = self.H.sparsity()

    def _static_shift_matrix(self):
        """
        Creates a shift matrix S.
        """
        # Create Scipy Sparse Matrix
        S_scipy = sp.diags(
            [np.ones(self.N - 1)],
            [-1],
            shape=(self.N, self.N)
        )
        
        # Convert to CSC format for CasADi compatibility
        S = S_scipy.tocsc()

        # Create CasADi DM
        self.S_sparsity = ca.Sparsity(
            S.shape[0], S.shape[1],
            S.indptr, S.indices
        )
        self.S = ca.DM(self.S_sparsity, S.data)

    def _static_friction_matrix(self):
        """
        Friction pyramid Constraints:
            |fx| < mu*fz 
            |fy| < mu*fz
        """
        rows, cols, vals = [], [], []
        row = 0
        base_u = self.N * self.nx

        for k in range(self.N):
            u0 = base_u + k * self.nu

            for foot in range(2):
                fx = u0 + 3 * foot
                fy = fx + 1
                fz = fx + 2

                # fx - mu*fz <= 0
                rows += [row, row]
                cols += [fx, fz]
                vals += [1.0, -self.mu]
                row += 1

                # -fx - mu*fz <= 0
                rows += [row, row]
                cols += [fx, fz]
                vals += [-1.0, -self.mu]
                row += 1

                # fy - mu*fz <= 0
                rows += [row, row]
                cols += [fy, fz]
                vals += [1.0, -self.mu]
                row += 1

                # -fy - mu*fz <= 0
                rows += [row, row]
                cols += [fy, fz]
                vals += [-1.0, -self.mu]
                row += 1

        A = sp.csc_matrix((vals, (rows, cols)), shape=(row, self.nz))

        self.A_friction = ca.DM(
            ca.Sparsity(A.shape[0], A.shape[1], A.indptr, A.indices),
            A.data
        )

    def build_and_solve(self, Ad_seq, Bd_seq, gd_seq, x0, x_ref, contact_table):
        """
        Solves the MPC QP.
        """
        t0 = time.time()

        # 1. Dynamic Constraints (Aeq * z = Beq)
        Aeq, Beq = self._dynamic_constraints(Ad_seq, Bd_seq, gd_seq, x0)

        # 2. Cost Gradient (linear term)
        g = self._cost_gradient(x_ref)

        # 3. Variable Bounds (lbx, ubx) - Contact scheduling
        lbx, ubx = self._enforcing_bounds(contact_table)

        # 4. Stack Constraints (Dynamics + Friction)
        A = ca.vertcat(Aeq, self.A_friction)

        # Equality bounds (Beq) and Friction upper bounds (0)
        # Note: Friction constraints are A_fric * z <= 0, so lb is -inf, ub is 0
        n_friction = self.A_friction.size1()
        
        lb = ca.vertcat(Beq, -ca.inf * ca.DM.ones(n_friction, 1))
        ub = ca.vertcat(Beq,  ca.DM.zeros(n_friction, 1))

        # 5. Solver Init
        if not self._initialized:
            qp = {"h": self.H_sp, "a": A.sparsity()}
            self.solver = ca.conic("mpc", "osqp", qp, self.opts)
            self._initialized = True

        # 6. Solve
        try:
            sol = self.solver(
                h=self.H,
                g=g,
                a=A,
                lba=lb,
                uba=ub,
                lbx=lbx,
                ubx=ubx
            )
            
            # Check for success
            if self.solver.stats()['return_status'] != 'Solve_Succeeded':
                pass # You can log warning here if needed

            z = sol["x"].full().flatten()
            self.solve_time_ms = 1e3 * (time.time() - t0)

            # Extract first control input (u0)
            # z structure: [x_0...x_N, u_0...u_N]
            u_start = self.N * self.nx
            u0 = z[u_start : u_start + self.nu]
            return u0

        except Exception as e:
            print(f"MPC Exception: {e}")
            return np.zeros(self.nu)

    # ----------------------------------------------------------------
    # Utils
    # ----------------------------------------------------------------

    def _dynamic_constraints(self, Ad_seq, Bd_seq, gd_seq, x0):
        """
        Builds Aeq and Beq for x_{k+1} = A x_k + B u_k + g
        """
        rows = []
        rhs = []

        for k in range(self.N):
            row = ca.DM.zeros(self.nx, self.nz)

            # x_{k+1} term (Identity)
            # Indices for x_{k+1}
            row[:, k * self.nx : (k + 1) * self.nx] = ca.DM.eye(self.nx)

            # x_k term (-A)
            if k == 0:
                # Initial state is fixed (moves to RHS)
                rhs.append(Ad_seq[0] @ x0 + gd_seq[0])
            else:
                # Indices for x_k
                row[:, (k - 1) * self.nx : k * self.nx] = -Ad_seq[k]
                rhs.append(gd_seq[k])

            # u_k term (-B)
            u_start = self.N * self.nx + k * self.nu
            row[:, u_start : u_start + self.nu] = -Bd_seq[k]

            rows.append(row)

        # [FIX]: Return statement moved OUTSIDE the loop
        Aeq = ca.vertcat(*rows)
        Beq = ca.vertcat(*rhs)

        return Aeq, Beq

    def _cost_gradient(self, x_ref):
        """
        Linear cost term: g = -2 * Q * x_ref
        """
        g = ca.DM.zeros(self.nz, 1)
        for k in range(self.N):
            idx = k * self.nx
            # Q is diagonal, so Q @ vec is element-wise mult if flattened
            # But using matrix mult for safety
            g[idx : idx + self.nx] = -2 * ca.DM(self.Q) @ x_ref[:, k]
        return g

    def _enforcing_bounds(self, contact):
        """
        Box constraints for State and Input.
        Forces set to 0 if contact=0 (swing).
        """
        # [FIX]: Use Numpy for safe infinity generation, avoid 'GenDM_ones' error
        lbx = -np.inf * np.ones(self.nz)
        ubx =  np.inf * np.ones(self.nz)

        base_u = self.N * self.nx

        for k in range(self.N):
            for foot in range(2):
                # Indices: fx, fy, fz for each foot
                fx = base_u + k * self.nu + 3 * foot
                fy = fx + 1
                fz = fx + 2

                if contact[foot, k] == 0:
                    # SWING: Force = 0
                    lbx[fx : fx + 3] = 0.0
                    ubx[fx : fx + 3] = 0.0
                else:
                    # STANCE: Fz in [min, max], Fx/Fy unbounded (handled by friction cone)
                    lbx[fz] = self.fz_min
                    ubx[fz] = self.fz_max

        return lbx, ubx