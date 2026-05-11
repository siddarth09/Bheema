"""
Convex Bezier Swing Trajectory Optimizer for Bheema
====================================================

The key insight: a Bezier curve is LINEAR in its control points.

    P(s) = sum_i  B_i(s) * P_i

where B_i(s) = C(n,i) * s^i * (1-s)^(n-i) are fixed basis functions
and P_i are the control points (decision variables).

This means:
  - Position at any s is a LINEAR function of control points
  - Velocity at any s is a LINEAR function of control points  
  - Acceleration at any s is a LINEAR function of control points

So if we write cost and constraints in terms of control points,
we get a QP (quadratic cost, linear constraints) = CONVEX.

The optimization finds the Z-heights of 8 control points that:
  1. Minimize actuator effort (acceleration squared)
  2. Minimize jerk (smoothness)
  3. Track a desired clearance height at mid-swing
  4. Subject to: ground clearance, max height, smooth landing

This replaces the hand-tuned `make_swing_trajectory` in gait.py.

Usage:
    optimizer = BezierSwingOptimizer(n_points=8)
    
    # Solve for optimal control points
    cp_z = optimizer.solve(
        h_clearance=0.18,   # desired peak clearance
        h_max=0.25,         # max allowed height
        h_ground=0.02,      # min clearance at all points
        w_accel=1.0,        # acceleration penalty weight
        w_jerk=0.5,         # jerk penalty weight
        w_track=10.0,       # midswing height tracking weight
    )
    
    # Build the trajectory function (same interface as make_swing_trajectory)
    eval_at = optimizer.make_trajectory(p0, pf, t_swing, cp_z)
    pos, vel, acl = eval_at(t)
"""

import numpy as np
from scipy.optimize import minimize
from dataclasses import dataclass
from typing import Callable, Tuple

# ==============================================================================
# PART 1: Bernstein Basis Math
# ==============================================================================
# 
# A Bezier curve of degree n with control points P_0, ..., P_n is:
#
#   P(s) = sum_{i=0}^{n}  B_{i,n}(s) * P_i       s in [0, 1]
#
# where the Bernstein basis polynomials are:
#
#   B_{i,n}(s) = C(n,i) * s^i * (1-s)^{n-i}
#
# First derivative (velocity):
#
#   P'(s) = n * sum_{i=0}^{n-1}  B_{i,n-1}(s) * (P_{i+1} - P_{i})
#
# Second derivative (acceleration):
#
#   P''(s) = n*(n-1) * sum_{i=0}^{n-2}  B_{i,n-2}(s) * (P_{i+2} - 2*P_{i+1} + P_{i})
#
# Third derivative (jerk):
#
#   P'''(s) = n*(n-1)*(n-2) * sum_{i=0}^{n-3} B_{i,n-3}(s) * (P_{i+3} - 3*P_{i+2} + 3*P_{i+1} - P_{i})
#
# KEY PROPERTY: All derivatives are LINEAR in the control points P_i.
# This is what makes the optimization convex.
# ==============================================================================


def _binomial(n: int, k: int) -> float:
    """Binomial coefficient C(n, k)."""
    if k < 0 or k > n:
        return 0.0
    result = 1.0
    for i in range(min(k, n - k)):
        result = result * (n - i) / (i + 1)
    return result


def bernstein_matrix(n: int, s_samples: np.ndarray) -> np.ndarray:
    """
    Build the Bernstein basis matrix.
    
    Returns B where B[k, i] = B_{i,n}(s_k)
    
    So if P is the control point vector (n+1,), then:
        positions = B @ P     (gives position at each sample point)
    
    This is the matrix form of: P(s) = sum_i B_i(s) * P_i
    
    Args:
        n: Bezier degree (n+1 control points)
        s_samples: (K,) array of parameter values in [0, 1]
    
    Returns:
        B: (K, n+1) Bernstein basis matrix
    """
    K = len(s_samples)
    B = np.zeros((K, n + 1))
    for i in range(n + 1):
        coeff = _binomial(n, i)
        B[:, i] = coeff * (s_samples ** i) * ((1 - s_samples) ** (n - i))
    return B


def bernstein_derivative_matrix(n: int, s_samples: np.ndarray, order: int = 1) -> np.ndarray:
    """
    Build matrix that maps control points to the k-th derivative at sample points.
    
    Uses the property that the k-th derivative of a degree-n Bezier is a 
    degree-(n-k) Bezier on the k-th finite differences of the control points.
    
    For velocity (order=1):
        D1 @ P  gives  P'(s) / n   at each sample
        Actual velocity = n * (D1 @ P)
        
    We return the FULL matrix including the scaling factor, so:
        velocity_samples = Md @ P
    
    Args:
        n: Bezier degree
        s_samples: (K,) sample points
        order: derivative order (1=velocity, 2=acceleration, 3=jerk)
    
    Returns:
        Md: (K, n+1) matrix such that  Md @ P = d^k P / ds^k  at samples
    """
    if order == 0:
        return bernstein_matrix(n, s_samples)

    # Finite difference matrix for control points
    # 1st diff: D[i] = P[i+1] - P[i],  shape (n, n+1)
    # 2nd diff: D[i] = P[i+2] - 2*P[i+1] + P[i],  shape (n-1, n+1)
    # k-th diff: shape (n-k+1, n+1)
    
    m = n + 1  # number of control points
    D = np.eye(m)  # Start with identity
    
    for _ in range(order):
        rows = D.shape[0] - 1
        D_new = np.zeros((rows, m))
        for i in range(rows):
            D_new[i, :] = D[i + 1, :] - D[i, :]
        D = D_new
    
    # D now maps: P -> k-th finite differences of P
    # The k-th derivative is a Bezier of degree (n-k) on these differences
    
    # Scaling factor: n! / (n-k)!
    scale = 1.0
    for i in range(order):
        scale *= (n - i)
    
    # Bernstein basis of degree (n - order)
    B_reduced = bernstein_matrix(n - order, s_samples)  # (K, n-order+1)
    
    # Full matrix: scale * B_reduced @ D
    Md = scale * (B_reduced @ D)  # (K, n+1)
    
    return Md


# ==============================================================================
# PART 2: QP Formulation
# ==============================================================================
#
# Decision variables: z = [z_1, z_2, z_3, z_4, z_5, z_6]  (6 free heights)
# Fixed endpoints:    z_0 = 0 (takeoff on ground), z_7 = 0 (land on ground)
#
# We split the full control point vector P = [0, z_1, ..., z_6, 0]
# into fixed + free parts:
#
#   P = E_fixed * p_fixed  +  E_free * z
#
# where E_fixed picks out indices 0 and 7, E_free picks out indices 1-6.
#
# Then any linear-in-P quantity becomes linear-in-z:
#   B @ P = B @ E_fixed * p_fixed + B @ E_free * z
#         = b_const + M_free * z
#
# Cost: min  z^T H z + f^T z     (quadratic in z -> convex)
# Subject to: A_ineq * z <= b_ineq   (linear -> convex)
#
# This is a standard QP. We solve it with scipy for portability,
# but you could plug it into OSQP or CasADi to match your MPC solver.
# ==============================================================================


@dataclass
class BezierSwingOptimizer:
    """
    Solves for optimal Bezier control point heights as a convex QP.
    
    The X-coordinates are fixed (evenly spaced along the step).
    Only the Z-heights of the interior control points are optimized.
    """
    n_points: int = 8          # Total control points (degree = n_points - 1)
    n_samples: int = 50        # Discretization for cost integration
    
    def __post_init__(self):
        self.degree = self.n_points - 1  # Bezier degree
        self.n_free = self.n_points - 2  # Free variables (endpoints fixed)
        
        # Sample points for numerical integration (trapezoidal rule)
        self.s_samples = np.linspace(0, 1, self.n_samples)
        self.ds = 1.0 / (self.n_samples - 1)
        
        # Trapezoidal weights for integration: int_0^1 f(s) ds ≈ w^T f
        self.w_trap = np.full(self.n_samples, self.ds)
        self.w_trap[0] = self.ds / 2
        self.w_trap[-1] = self.ds / 2
        
        # Build all the matrices once (they depend only on n_points)
        self._build_matrices()
    
    def _build_matrices(self):
        """
        Precompute the basis matrices and split into fixed/free parts.
        
        Full control point vector: P = [z_0, z_1, ..., z_7]  (8 values)
        Fixed: z_0 = 0, z_7 = 0  (foot on ground at start and end)
        Free:  z_1, ..., z_6      (6 decision variables)
        
        Selector matrices:
            E_free:  (8, 6) picks columns 1-6
            E_fixed: (8, 2) picks columns 0 and 7
        """
        n = self.n_points
        
        # Selector for free variables (indices 1 to n-2)
        self.E_free = np.zeros((n, self.n_free))
        for i in range(self.n_free):
            self.E_free[i + 1, i] = 1.0
        
        # Selector for fixed endpoints (indices 0 and n-1)
        self.E_fixed = np.zeros((n, 2))
        self.E_fixed[0, 0] = 1.0
        self.E_fixed[n - 1, 1] = 1.0
        
        # Fixed endpoint values: [z_0, z_7] = [0, 0]
        self.p_fixed = np.array([0.0, 0.0])
        
        # Position basis: B @ P gives position at samples
        B_pos = bernstein_matrix(self.degree, self.s_samples)
        self.M_pos_free = B_pos @ self.E_free       # (K, 6)
        self.b_pos_const = B_pos @ self.E_fixed @ self.p_fixed  # (K,) = zeros
        
        # Velocity basis (1st derivative w.r.t. s)
        B_vel = bernstein_derivative_matrix(self.degree, self.s_samples, order=1)
        self.M_vel_free = B_vel @ self.E_free
        self.b_vel_const = B_vel @ self.E_fixed @ self.p_fixed
        
        # Acceleration basis (2nd derivative w.r.t. s)
        B_acl = bernstein_derivative_matrix(self.degree, self.s_samples, order=2)
        self.M_acl_free = B_acl @ self.E_free
        self.b_acl_const = B_acl @ self.E_fixed @ self.p_fixed
        
        # Jerk basis (3rd derivative w.r.t. s)
        B_jrk = bernstein_derivative_matrix(self.degree, self.s_samples, order=3)
        self.M_jrk_free = B_jrk @ self.E_free
        self.b_jrk_const = B_jrk @ self.E_fixed @ self.p_fixed
    
    def solve(
        self,
        h_clearance: float = 0.18,
        h_max: float = 0.25,
        h_ground: float = 0.02,
        w_accel: float = 1.0,
        w_jerk: float = 0.5,
        w_track: float = 10.0,
        t_swing: float = 0.25,
    ) -> np.ndarray:
        """
        Solve the convex QP for optimal control point heights.
        
        Cost function (all quadratic in z -> convex):
        
            J = w_accel * integral |a(s)|^2 ds       (minimize effort)
              + w_jerk  * integral |j(s)|^2 ds       (minimize jerk)
              + w_track * |z(0.5) - h_clearance|^2   (track desired height)
        
        where a(s) = P''(s) / T^2  and  j(s) = P'''(s) / T^3.
        
        Constraints (all linear in z -> convex):
            z(s) >= h_ground   for s in [0.1, 0.9]    (ground clearance)
            z(s) <= h_max      for all s               (max height)
            z_i >= 0           for all free points     (non-negative heights)
            z'(0) >= 0                                 (foot lifts up at start)
            z'(1) <= 0                                 (foot comes down at end)
        
        Args:
            h_clearance: desired height at midswing (m)
            h_max: maximum allowed height (m)  
            h_ground: minimum clearance during swing core [0.1, 0.9] (m)
            w_accel: weight on acceleration minimization
            w_jerk: weight on jerk minimization
            w_track: weight on midswing height tracking
            t_swing: swing duration (s), affects accel/jerk scaling
        
        Returns:
            cp_z: (8,) full control point heights including fixed endpoints
        """
        n_free = self.n_free
        W = self.w_trap  # trapezoidal weights (n_samples,)
        
        # =====================================================================
        # BUILD QUADRATIC COST:  J = 0.5 * z^T H z + f^T z + const
        # =====================================================================
        
        # --- Term 1: Acceleration penalty ---
        # a(s) = M_acl @ z + b_acl  (in parameter space)
        # Physical accel = a(s) / T^2
        # Cost = w_accel/T^4 * integral |M_acl @ z + b_acl|^2 ds
        #      = w_accel/T^4 * (M_acl @ z + b_acl)^T diag(W) (M_acl @ z + b_acl)
        #
        # Expanding:
        #   = z^T [w_accel/T^4 * M_acl^T diag(W) M_acl] z
        #     + 2 * [w_accel/T^4 * b_acl^T diag(W) M_acl] z
        #     + const
        
        inv_T2 = 1.0 / (t_swing ** 2)
        inv_T4 = inv_T2 * inv_T2
        
        Ma = self.M_acl_free  # (K, 6)
        ba = self.b_acl_const  # (K,)
        WMa = W[:, None] * Ma  # diag(W) @ Ma, shape (K, 6)
        
        H_accel = w_accel * inv_T4 * (Ma.T @ WMa)  # (6, 6)
        f_accel = w_accel * inv_T4 * (ba @ WMa)     # (6,)
        
        # --- Term 2: Jerk penalty ---
        inv_T3 = inv_T2 / t_swing
        inv_T6 = inv_T3 * inv_T3
        
        Mj = self.M_jrk_free
        bj = self.b_jrk_const
        WMj = W[:, None] * Mj
        
        H_jerk = w_jerk * inv_T6 * (Mj.T @ WMj)
        f_jerk = w_jerk * inv_T6 * (bj @ WMj)
        
        # --- Term 3: Midswing height tracking ---
        # z_mid = M_pos_free[mid_idx, :] @ z + b_pos_const[mid_idx]
        # Cost = w_track * (z_mid - h_clearance)^2
        #      = w_track * (m^T z + c - h_clearance)^2
        #      = w_track * z^T (m m^T) z + 2 * w_track * (c - h_clearance) * m^T z + const
        
        mid_idx = self.n_samples // 2
        m_mid = self.M_pos_free[mid_idx, :]       # (6,)
        c_mid = self.b_pos_const[mid_idx]          # scalar
        
        H_track = w_track * np.outer(m_mid, m_mid)  # (6, 6)
        f_track = w_track * (c_mid - h_clearance) * m_mid  # (6,)
        
        # --- Total cost ---
        H = H_accel + H_jerk + H_track  # (6, 6) positive semi-definite
        f = f_accel + f_jerk + f_track   # (6,)
        
        # Symmetrize (numerical safety)
        H = 0.5 * (H + H.T)
        
        # =====================================================================
        # BUILD LINEAR CONSTRAINTS
        # =====================================================================
        # All constraints are:  A_ub @ z <= b_ub  (linear in z -> convex)
        
        A_ub_list = []
        b_ub_list = []
        
        # --- Constraint 1: Ground clearance ---
        # z(s) >= h_ground  for s in [0.1, 0.9]
        # Rewrite as: -z(s) <= -h_ground
        #             -(M_pos_free @ z + b_pos_const) <= -h_ground
        #             -M_pos_free @ z <= -h_ground + b_pos_const
        
        core_mask = (self.s_samples >= 0.1) & (self.s_samples <= 0.9)
        M_core = self.M_pos_free[core_mask, :]
        b_core_const = self.b_pos_const[core_mask]
        
        A_ub_list.append(-M_core)
        b_ub_list.append(-h_ground + b_core_const)
        
        # --- Constraint 2: Maximum height ---
        # z(s) <= h_max  for all s
        # M_pos_free @ z + b_pos_const <= h_max
        # M_pos_free @ z <= h_max - b_pos_const
        
        A_ub_list.append(self.M_pos_free)
        b_ub_list.append(h_max - self.b_pos_const)
        
        # --- Constraint 3: Non-negative control points ---
        # z_i >= 0  ->  -z_i <= 0  ->  -I @ z <= 0
        
        A_ub_list.append(-np.eye(n_free))
        b_ub_list.append(np.zeros(n_free))
        
        # --- Constraint 4: Foot lifts at start ---
        # z'(0) >= 0, i.e. the foot must be going UP at takeoff
        # -M_vel_free[0, :] @ z <= b_vel_const[0]
        
        A_ub_list.append(-self.M_vel_free[0:1, :])
        b_ub_list.append(self.b_vel_const[0:1])
        
        # --- Constraint 5: Foot descends at end ---
        # z'(1) <= 0, i.e. the foot must be coming DOWN at touchdown
        # M_vel_free[-1, :] @ z <= -b_vel_const[-1]
        
        A_ub_list.append(self.M_vel_free[-1:, :])
        b_ub_list.append(-self.b_vel_const[-1:])
        
        # Stack all constraints
        A_ub = np.vstack(A_ub_list)
        b_ub = np.concatenate(b_ub_list)
        
        # =====================================================================
        # SOLVE QP
        # =====================================================================
        # min  0.5 * z^T H z + f^T z
        # s.t. A_ub @ z <= b_ub
        #
        # scipy.optimize.minimize with method='SLSQP' handles this.
        # For production, swap to OSQP for speed.
        
        def cost(z):
            return 0.5 * z @ H @ z + f @ z
        
        def cost_grad(z):
            return H @ z + f
        
        # Build constraint dicts for SLSQP
        constraints = []
        for i in range(A_ub.shape[0]):
            # A_ub[i] @ z <= b_ub[i]  ->  b_ub[i] - A_ub[i] @ z >= 0
            constraints.append({
                'type': 'ineq',
                'fun': lambda z, row=i: b_ub[row] - A_ub[row] @ z,
                'jac': lambda z, row=i: -A_ub[row],
            })
        
        # Warm start: uniform height at clearance
        z0 = np.full(n_free, h_clearance * 0.8)
        
        result = minimize(
            cost, z0,
            jac=cost_grad,
            method='SLSQP',
            constraints=constraints,
            options={'maxiter': 200, 'ftol': 1e-10},
        )
        
        if not result.success:
            print(f"[BezierSwingOptimizer] Warning: {result.message}")
        
        # Assemble full control point vector
        cp_z = np.zeros(self.n_points)
        cp_z[0] = 0.0                     # Takeoff: on ground
        cp_z[1:-1] = result.x             # Optimized heights
        cp_z[-1] = 0.0                    # Touchdown: on ground
        
        return cp_z
    
    def make_trajectory(
        self,
        p0: np.ndarray,
        pf: np.ndarray,
        t_swing: float,
        cp_z: np.ndarray,
    ) -> Callable[[float], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Build a trajectory closure with the SAME INTERFACE as the original
        make_swing_trajectory in gait.py.
        
        Ground-plane (X, Y) motion uses the same minimum-jerk profile as before.
        Only the Z-axis uses the optimized Bezier curve.
        
        Args:
            p0: (3,) takeoff position in world
            pf: (3,) touchdown position in world
            t_swing: swing duration (s)
            cp_z: (n_points,) optimized control point heights
        
        Returns:
            eval_at: function(t) -> (pos, vel, acl), each (3,)
        """
        p0 = np.asarray(p0, dtype=float)
        pf = np.asarray(pf, dtype=float)
        dp = pf - p0
        T = float(t_swing)
        n = self.degree
        
        # Precompute X control points (evenly spaced along step)
        cp_x_local = np.linspace(0, 1, self.n_points)  # Normalized
        
        def eval_at(t: float):
            s = np.clip(t / T, 0.0, 1.0)
            s_arr = np.array([s])
            
            # --- Ground plane: minimum-jerk (same as original) ---
            mj   = 10*s**3 - 15*s**4 + 6*s**5
            dmj  = 30*s**2 - 60*s**3 + 30*s**4
            d2mj = 60*s    - 180*s**2 + 120*s**3
            
            p_xy = p0 + dp * mj
            v_xy = dp * dmj / T
            a_xy = dp * d2mj / (T ** 2)
            
            # --- Vertical: optimized Bezier ---
            B0 = bernstein_matrix(n, s_arr)[0]         # (n+1,)
            B1 = bernstein_derivative_matrix(n, s_arr, 1)[0]
            B2 = bernstein_derivative_matrix(n, s_arr, 2)[0]
            
            p_z = B0 @ cp_z
            v_z = (B1 @ cp_z) / T
            a_z = (B2 @ cp_z) / (T ** 2)
            
            # Combine (override Z from Bezier)
            pos = p_xy.copy()
            vel = v_xy.copy()
            acl = a_xy.copy()
            
            pos[2] = p_z
            vel[2] = v_z
            acl[2] = a_z
            
            return pos, vel, acl
        
        return eval_at


# ==============================================================================
# PART 3: Drop-in replacement for gait.py
# ==============================================================================

def make_swing_trajectory_bezier(p0, pf, t_swing, h_sw=0.18):
    """
    Drop-in replacement for Gait.make_swing_trajectory().
    
    Solves the convex optimization once, then returns the same
    eval_at(t) -> (pos, vel, acl) closure.
    
    Usage in gait.py:
        # Old:
        # eval_at = self.make_swing_trajectory(foot_pos, pos_touchdown_world, t_swing, h_sw=HEIGHT_SWING)
        
        # New:
        eval_at = make_swing_trajectory_bezier(foot_pos, pos_touchdown_world, t_swing, h_sw=HEIGHT_SWING)
    """
    optimizer = BezierSwingOptimizer(n_points=8, n_samples=50)
    
    cp_z = optimizer.solve(
        h_clearance=h_sw,
        h_max=h_sw * 1.4,       # Allow 40% overshoot
        h_ground=0.02,           # 2cm minimum clearance
        w_accel=1.0,
        w_jerk=0.5,
        w_track=10.0,
        t_swing=t_swing,
    )
    
    return optimizer.make_trajectory(p0, pf, t_swing, cp_z)


# ==============================================================================
# DEMO / VERIFICATION
# ==============================================================================

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    print("=" * 60)
    print("Convex Bezier Swing Trajectory Optimizer")
    print("=" * 60)
    
    optimizer = BezierSwingOptimizer(n_points=8, n_samples=50)
    
    # Solve
    cp_z = optimizer.solve(
        h_clearance=0.18,
        h_max=0.25,
        h_ground=0.02,
        w_accel=1.0,
        w_jerk=0.5,
        w_track=10.0,
        t_swing=0.25,
    )
    
    print(f"\nOptimal control point heights (m):")
    for i, z in enumerate(cp_z):
        fixed = " (fixed)" if i == 0 or i == len(cp_z) - 1 else ""
        print(f"  P{i}: z = {z:.4f}{fixed}")
    
    # Build trajectory
    p0 = np.array([0.0, 0.12, 0.0])
    pf = np.array([0.35, 0.12, 0.0])
    t_swing = 0.25
    
    eval_bezier = optimizer.make_trajectory(p0, pf, t_swing, cp_z)
    
    # Compare with original minimum-jerk
    def min_jerk_eval(t):
        s = np.clip(t / t_swing, 0.0, 1.0)
        dp = pf - p0
        mj   = 10*s**3 - 15*s**4 + 6*s**5
        dmj  = 30*s**2 - 60*s**3 + 30*s**4
        d2mj = 60*s    - 180*s**2 + 120*s**3
        
        p = p0 + dp * mj
        v = dp * dmj / t_swing
        a = dp * d2mj / (t_swing ** 2)
        
        h_sw = 0.18
        b    = 64 * s**3 * (1 - s)**3
        db   = 192 * s**2 * (1 - s)**2 * (1 - 2*s)
        d2b  = 192 * (2*s*(1-s)**2*(1-2*s) - 2*s**2*(1-s)*(1-2*s) - 2*s**2*(1-s)**2)
        
        p[2] += h_sw * b
        v[2] += h_sw * db / t_swing
        a[2] += h_sw * d2b / (t_swing ** 2)
        return p, v, a
    
    # Sample both
    dt = 0.001
    t_vec = np.arange(0, t_swing + dt, dt)
    
    bez_pos = np.zeros((3, len(t_vec)))
    bez_vel = np.zeros((3, len(t_vec)))
    bez_acl = np.zeros((3, len(t_vec)))
    mj_pos  = np.zeros((3, len(t_vec)))
    mj_vel  = np.zeros((3, len(t_vec)))
    mj_acl  = np.zeros((3, len(t_vec)))
    
    for i, t in enumerate(t_vec):
        p, v, a = eval_bezier(t)
        bez_pos[:, i] = p; bez_vel[:, i] = v; bez_acl[:, i] = a
        
        p, v, a = min_jerk_eval(t)
        mj_pos[:, i] = p; mj_vel[:, i] = v; mj_acl[:, i] = a
    
    # Compute cost metrics
    acl_cost_bez = np.trapz(bez_acl[2, :] ** 2, t_vec)
    acl_cost_mj  = np.trapz(mj_acl[2, :] ** 2, t_vec)
    
    jerk_bez = np.gradient(bez_acl[2, :], dt)
    jerk_mj  = np.gradient(mj_acl[2, :], dt)
    jrk_cost_bez = np.trapz(jerk_bez ** 2, t_vec)
    jrk_cost_mj  = np.trapz(jerk_mj ** 2, t_vec)
    
    print(f"\nCost comparison:")
    print(f"  Acceleration L2:  Bezier = {acl_cost_bez:.1f},  MinJerk = {acl_cost_mj:.1f}")
    print(f"  Jerk L2:          Bezier = {jrk_cost_bez:.1f},  MinJerk = {jrk_cost_mj:.1f}")
    
    # --- Plot ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    
    # 1. Side view (X vs Z)
    ax = axes[0, 0]
    ax.plot(bez_pos[0, :], bez_pos[2, :], color='#D85A30', linewidth=2, label='Bezier (optimized)')
    ax.plot(mj_pos[0, :], mj_pos[2, :], color='#3266ad', linewidth=2, label='Min-jerk (original)')
    
    # Control points
    cp_x = np.linspace(p0[0], pf[0], len(cp_z))
    ax.scatter(cp_x, cp_z, color='#D85A30', s=40, zorder=5, marker='o')
    ax.plot(cp_x, cp_z, color='#D85A30', linewidth=0.5, linestyle='--', alpha=0.5)
    
    ax.axhline(0, color='gray', linewidth=0.5, linestyle='--')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Z (m)')
    ax.set_title('Swing Arc (side view)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Height vs time
    ax = axes[0, 1]
    ax.plot(t_vec, bez_pos[2, :], color='#D85A30', linewidth=2, label='Bezier Z')
    ax.plot(t_vec, mj_pos[2, :], color='#3266ad', linewidth=2, label='Min-jerk Z')
    ax.axhline(0.18, color='gray', linewidth=0.5, linestyle=':', label='Target clearance')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Height (m)')
    ax.set_title('Foot clearance over time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Vertical velocity
    ax = axes[1, 0]
    ax.plot(t_vec, bez_vel[2, :], color='#D85A30', linewidth=2, label='Bezier vz')
    ax.plot(t_vec, mj_vel[2, :], color='#3266ad', linewidth=2, label='Min-jerk vz')
    ax.axhline(0, color='gray', linewidth=0.5, linestyle='--')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Velocity (m/s)')
    ax.set_title('Vertical velocity (should be 0 at endpoints)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Vertical acceleration
    ax = axes[1, 1]
    ax.plot(t_vec, bez_acl[2, :], color='#D85A30', linewidth=2, label='Bezier az')
    ax.plot(t_vec, mj_acl[2, :], color='#3266ad', linewidth=2, label='Min-jerk az')
    ax.axhline(0, color='gray', linewidth=0.5, linestyle='--')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Acceleration (m/s²)')
    ax.set_title(f'Vertical acceleration (effort cost: Bez={acl_cost_bez:.0f}, MJ={acl_cost_mj:.0f})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('Convex Bezier vs Minimum-Jerk Swing Trajectory', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()