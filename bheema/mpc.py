import casadi as ca
import numpy as np
import scipy.sparse as sp


class CentroidalMPC:
    """
    Humanoid Centroidal MPC (2 contacts).
    Robustified for walking stability.
    """

    def __init__(self, N: int, dt: float, mass: float = 40.0):

        self.N = N
        self.dt = dt
        self.mass = mass

        self.nx = 12
        self.nu = 6

        self.nz = N * self.nx + N * self.nu

        # -------------------------------------------------
        # Cost Weights
        # -------------------------------------------------
        #                   p        rpy       v        w
        self.Q = np.diag([
            10, 10, 1,     # Relaxed Z from 10 to 1 so it doesn't violently jump
            50, 50, 50,    # Orientation
            10, 10, 10,    # Linear Velocity
            1, 1, 1        # Angular Velocity
        ])

        self.R = 1e-1 * np.eye(self.nu)

        # -------------------------------------------------
        # Constraints
        # -------------------------------------------------
        self.mu = 0.6
        
        
        self.fz_min = 20.0
        self.fz_max = 1500.0

        self._build_hessian()
        self._build_friction_matrix()

        self._initialized = False

    def _build_hessian(self):
        rows, cols, vals = [], [], []

        # State cost
        for k in range(self.N):
            base = k * self.nx
            for i in range(self.nx):
                rows.append(base + i)
                cols.append(base + i)
                vals.append(2 * self.Q[i, i])

        # Input cost
        for k in range(self.N):
            base = self.N * self.nx + k * self.nu
            for i in range(self.nu):
                rows.append(base + i)
                cols.append(base + i)
                vals.append(2 * self.R[i, i])

        self.H = ca.DM.triplet(rows, cols, ca.DM(vals), self.nz, self.nz)
        self.H_sp = self.H.sparsity()

    def _build_friction_matrix(self):
        rows, cols, vals = [], [], []
        row = 0
        base_u = self.N * self.nx

        for k in range(self.N):
            uk = base_u + k * self.nu
            for foot in range(2):
                fx, fy, fz = uk + 3 * foot, uk + 3 * foot + 1, uk + 3 * foot + 2

                # Pyramid Friction Cone
                # fx - mu fz <= 0
                rows += [row, row]; cols += [fx, fz]; vals += [1.0, -self.mu]; row += 1
                # -fx - mu fz <= 0
                rows += [row, row]; cols += [fx, fz]; vals += [-1.0, -self.mu]; row += 1
                # fy - mu fz <= 0
                rows += [row, row]; cols += [fy, fz]; vals += [1.0, -self.mu]; row += 1
                # -fy - mu fz <= 0
                rows += [row, row]; cols += [fy, fz]; vals += [-1.0, -self.mu]; row += 1

        A = sp.csc_matrix((vals, (rows, cols)), shape=(row, self.nz))
        self.A_fric = ca.DM(ca.Sparsity(A.shape[0], A.shape[1], A.indptr, A.indices), A.data)

    def solve(self, Ad_seq, Bd_seq, gd_seq, x0, x_ref, contact_table):
        """
        Solves the MPC QP.
        """

        # 1. Dynamics Constraints (Aeq * z = beq)
        Aeq_rows = []
        beq_rows = []

        for k in range(self.N):
            row = ca.DM.zeros(self.nx, self.nz)
            
            # x_k
            row[:, k*self.nx:(k+1)*self.nx] = ca.DM.eye(self.nx)
            
            # u_k
            u_start = self.N*self.nx + k*self.nu
            row[:, u_start:u_start+self.nu] = -Bd_seq[k]

            if k == 0:
                beq_rows.append(Ad_seq[0] @ x0 + gd_seq[0])
            else:
                row[:, (k-1)*self.nx:k*self.nx] = -Ad_seq[k]
                beq_rows.append(gd_seq[k])

            Aeq_rows.append(row)

        Aeq = ca.vertcat(*Aeq_rows)
        beq = ca.vertcat(*beq_rows)

        # 2. Cost Gradient
        g = ca.DM.zeros(self.nz, 1)
        Q_dm = ca.DM(self.Q)
        for k in range(self.N):
            idx = k * self.nx
            g[idx:idx+self.nx] = -2 * Q_dm @ x_ref[:, k]

        # 3. Variable Bounds
        lbx = -np.inf * np.ones(self.nz)
        ubx =  np.inf * np.ones(self.nz)
        base_u = self.N * self.nx

        for k in range(self.N):
            for foot in range(2):
                fx = base_u + k*self.nu + 3*foot
                fy, fz = fx + 1, fx + 2

                if contact_table[foot, k] == 0:
                    lbx[fx:fx+3] = 0.0
                    ubx[fx:fx+3] = 0.0
                else:
                    lbx[fz] = self.fz_min
                    ubx[fz] = self.fz_max

        # 4. Final Stack
        A = ca.vertcat(Aeq, self.A_fric)
        n_fric = self.A_fric.size1()
        
        lba = ca.vertcat(beq, -np.inf * ca.DM.ones(n_fric, 1))
        uba = ca.vertcat(beq,  ca.DM.zeros(n_fric, 1))

        # 5. Init Solver (Robust Options)
        if not self._initialized:
            opts = {
                'osqp': {
                    'verbose': False,
                    'eps_abs': 1e-4,   # Relaxed
                    'eps_rel': 1e-4,   # Relaxed
                    'max_iter': 10000  # High iteration limit
                }
            }
            qp = {"h": self.H_sp, "a": A.sparsity()}
            self.solver = ca.conic("mpc", "osqp", qp, opts)
            self._initialized = True

        # 6. Call
        sol = self.solver(h=self.H, g=g, a=A, lba=lba, uba=uba, lbx=lbx, ubx=ubx)
        
        # Check status manually if needed, but OSQP usually handles it well with loose eps
        z = sol["x"].full().flatten()
        u0_start = self.N * self.nx
        return z[u0_start:u0_start+self.nu]