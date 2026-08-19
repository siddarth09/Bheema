"""
Whole-body controller.

Resolves every task in one QP over (qddot, F) subject to the full floating-base dynamics,
contact consistency, friction, centre-of-pressure and torque limits. Joint torques follow
from the actuated rows:

    tau = S (M qddot + b - Jc^T F)

Tasks are weighted rather than strictly prioritised: a weighted least-squares stack keeps
the QP a single convex problem and degrades gracefully when tasks conflict, which a strict
null-space hierarchy does not.

    contact wrench   F  ->  F_mpc                       (from the centroidal MPC)
    swing foot       Jsw qddot + Jsw_dot qdot -> a_des   (from the swing trajectory)
    centre of mass   Jcom qddot + drift -> a_com_des
    torso attitude   Jtorso_w qddot + drift -> alpha_des
    posture          qddot_joints -> PD about q_nominal

This supersedes the per-leg split in leg_controller: the stance J^T mapping becomes the
contact-force task and the posture PD becomes the posture task, but both are now solved
against the contact constraints and torque limits rather than independently of them.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import casadi as ca
import numpy as np
import pinocchio as pin

from bheema.g1_config import PinG1Model
from bheema.params import FootParams, LegControlParams, MPCParams

N_INEQ_PER_LEG = 10
LEGS = ("LEFT", "RIGHT")


@dataclass(frozen=True)
class WBCParams:
    """Task weights and gains for the whole-body QP."""

    w_contact_force: float = 1.0      # track the MPC wrench
    w_swing: float = 100.0            # swing foot acceleration
    w_com: float = 50.0               # centre-of-mass acceleration
    w_torso: float = 20.0             # torso angular acceleration
    w_posture: float = 1.0            # joint posture
    w_reg_qddot: float = 1e-3         # regularisation
    w_reg_force: float = 1e-4

    # Swing foot task, 6D Cartesian PD feeding the desired acceleration.
    kp_swing: tuple[float, ...] = (400.0, 400.0, 400.0, 40.0, 40.0, 40.0)
    kd_swing: tuple[float, ...] = (40.0, 40.0, 40.0, 4.0, 4.0, 4.0)

    kp_com: float = 100.0
    kd_com: float = 20.0
    kp_torso: float = 200.0
    kd_torso: float = 20.0
    kp_posture: float = 60.0
    kd_posture: float = 6.0

    # Joint-level PD added to the QP torque for stance legs, as in WBIC. The QP alone
    # tracks the MPC wrench faithfully, and the wrench by itself is not stabilising; the
    # contact constraints leave qddot nearly determined, so the posture task inside the QP
    # cannot supply joint stiffness.
    kp_joint: float = 150.0
    kd_joint: float = 30.0
    joint_pd_on_swing: bool = False

    # Upper-body joints are position-actuated in the model, so their accelerations are
    # constrained to zero rather than solved for.
    freeze_upper_body: bool = True

    osqp_eps_abs: float = 1e-4
    osqp_eps_rel: float = 1e-4
    osqp_max_iter: int = 400


@dataclass
class WBCOutput:
    tau: np.ndarray            # (12,) leg joint torques
    qddot: np.ndarray          # (nv,)
    forces: np.ndarray         # (12,) stacked 6D wrenches, [left, right]
    solve_ms: float
    ok: bool


class WholeBodyController:
    """Weighted whole-body QP for the two-legged G1."""

    def __init__(self, g1: PinG1Model, params: WBCParams | None = None,
                 foot: FootParams | None = None, leg: LegControlParams | None = None,
                 mpc: MPCParams | None = None):
        self.p = params or WBCParams()
        self.foot = foot or FootParams()
        self.leg_p = leg or LegControlParams()
        self.mpc_p = mpc or MPCParams()

        self.nv = g1.model.nv
        self.leg_cols = np.array(g1.vcols_L + g1.vcols_R, dtype=int)
        base = np.arange(6)
        self.upper_cols = np.array(
            sorted(set(range(self.nv)) - set(self.leg_cols.tolist()) - set(base.tolist())),
            dtype=int)

        self.nz = self.nv + 12            # [qddot, F_left(6), F_right(6)]
        self.iF = self.nv

        self.q_nominal = np.asarray(self.leg_p.q_nominal, dtype=float)
        self.Kp_sw = np.diag(self.p.kp_swing)
        self.Kd_sw = np.diag(self.p.kd_swing)

        self.torso_id = g1.model.getFrameId("torso_link")
        self._build_solver()
        self.solve_ms = 0.0
        self.last_status = ""

    # -- solver construction -----------------------------------------------------

    def _n_eq(self) -> int:
        # 6 floating-base dynamics + optional upper-body freeze + 6 per foot contact
        n_freeze = len(self.upper_cols) if self.p.freeze_upper_body else 0
        return 6 + n_freeze + 12

    def _n_ineq(self) -> int:
        return 2 * N_INEQ_PER_LEG + 24     # friction/CoP + two-sided torque limits

    def _build_solver(self) -> None:
        n_eq, n_ineq = self._n_eq(), self._n_ineq()
        H_sp = ca.Sparsity.dense(self.nz, self.nz)
        A_sp = ca.Sparsity.dense(n_eq + n_ineq, self.nz)
        opts = {
            "warm_start_primal": True,
            "warm_start_dual": True,
            "error_on_fail": False,
            "osqp": {
                "eps_abs": self.p.osqp_eps_abs,
                "eps_rel": self.p.osqp_eps_rel,
                "max_iter": self.p.osqp_max_iter,
                "polish": False,
                "verbose": False,
            },
        }
        self.solver = ca.conic("wbc", "osqp", {"h": H_sp, "a": A_sp}, opts)
        self._z_prev = None

    # -- task assembly -----------------------------------------------------------

    def _contact_jacobians(self, g1: PinG1Model):
        """Stacked 6D contact Jacobians and their drift terms, per foot."""
        J, drift = {}, {}
        for name in LEGS:
            fid = g1.left_foot_id if name == "LEFT" else g1.right_foot_id
            J[name] = pin.getFrameJacobian(g1.model, g1.data, fid,
                                           pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
            drift[name] = g1.compute_Jdot_dq_world(name)
        return J, drift

    def _com_task(self, g1: PinG1Model, com_pos_des, com_vel_des, com_acc_des):
        Jcom = pin.jacobianCenterOfMass(g1.model, g1.data, g1.current_config.get_q())
        pin.centerOfMass(g1.model, g1.data, g1.current_config.get_q(),
                         g1.current_config.get_dq(), np.zeros(self.nv))
        drift = np.asarray(g1.data.acom[0]).reshape(3)
        a_des = (np.asarray(com_acc_des)
                 + self.p.kp_com * (np.asarray(com_pos_des) - g1.pos_com_world)
                 + self.p.kd_com * (np.asarray(com_vel_des) - g1.vel_com_world))
        A = np.zeros((3, self.nz))
        A[:, :self.nv] = Jcom
        return A, a_des - drift

    def _torso_task(self, g1: PinG1Model, rpy_des, omega_des):
        J = pin.getFrameJacobian(g1.model, g1.data, self.torso_id,
                                 pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)[3:6, :]
        Jdot = pin.getFrameJacobianTimeVariation(
            g1.model, g1.data, self.torso_id,
            pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)[3:6, :]
        drift = Jdot @ g1.current_config.get_dq()

        R_now = g1.data.oMf[self.torso_id].rotation
        R_des = pin.rpy.rpyToMatrix(float(rpy_des[0]), float(rpy_des[1]), float(rpy_des[2]))
        err = R_now @ pin.log3(R_now.T @ R_des)
        w_now = pin.getFrameVelocity(g1.model, g1.data, self.torso_id,
                                     pin.ReferenceFrame.LOCAL_WORLD_ALIGNED).angular
        a_des = self.p.kp_torso * err + self.p.kd_torso * (np.asarray(omega_des) - w_now)

        A = np.zeros((3, self.nz))
        A[:, :self.nv] = J
        return A, a_des - drift

    def _swing_task(self, g1: PinG1Model, name: str, pos_des, vel_des, acc_des):
        fid = g1.left_foot_id if name == "LEFT" else g1.right_foot_id
        J = pin.getFrameJacobian(g1.model, g1.data, fid,
                                 pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        drift = g1.compute_Jdot_dq_world(name)

        oMf = g1.data.oMf[fid]
        pos_now = oMf.translation
        v_now = pin.getFrameVelocity(g1.model, g1.data, fid,
                                     pin.ReferenceFrame.LOCAL_WORLD_ALIGNED).vector

        ori_err = oMf.rotation @ pin.log3(oMf.rotation.T @ g1.R_z)
        err6 = np.concatenate([np.asarray(pos_des) - pos_now, ori_err])
        vel6 = np.concatenate([np.asarray(vel_des), np.zeros(3)]) - v_now
        acc6 = np.concatenate([np.asarray(acc_des), np.zeros(3)])

        a_des = acc6 + self.Kp_sw @ err6 + self.Kd_sw @ vel6
        A = np.zeros((6, self.nz))
        A[:, :self.nv] = J
        return A, a_des - drift

    def _posture_task(self, g1: PinG1Model):
        q_leg = np.concatenate([g1.current_config.left_leg_angle,
                                g1.current_config.right_leg_angle])
        dq_leg = np.concatenate([g1.current_config.left_leg_vel,
                                 g1.current_config.right_leg_vel])
        q_ref = np.tile(self.q_nominal, 2)
        a_des = self.p.kp_posture * (q_ref - q_leg) - self.p.kd_posture * dq_leg
        A = np.zeros((12, self.nz))
        A[np.arange(12), self.leg_cols] = 1.0
        return A, a_des

    # -- constraints --------------------------------------------------------------

    def _friction_cop_rows(self, normals: dict[str, np.ndarray]) -> np.ndarray:
        """Friction pyramid, CoP and torsional-friction rows, in each contact frame."""
        mu, mu_tau = self.mpc_p.mu, self.mpc_p.mu_tau
        A = np.zeros((2 * N_INEQ_PER_LEG, self.nz))
        for li, name in enumerate(LEGS):
            n = np.asarray(normals[name], dtype=float)
            n = n / max(np.linalg.norm(n), 1e-9)
            t1 = np.array([1.0, 0.0, 0.0]) - n * n[0]
            if np.linalg.norm(t1) < 1e-6:
                t1 = np.array([0.0, 1.0, 0.0]) - n * n[1]
            t1 /= np.linalg.norm(t1)
            t2 = np.cross(n, t1)

            f0 = self.iF + 6 * li
            r0 = N_INEQ_PER_LEG * li
            F, T = slice(f0, f0 + 3), slice(f0 + 3, f0 + 6)

            A[r0 + 0, F] = t1 - mu * n
            A[r0 + 1, F] = -t1 - mu * n
            A[r0 + 2, F] = t2 - mu * n
            A[r0 + 3, F] = -t2 - mu * n
            A[r0 + 4, T], A[r0 + 4, F] = t1, -self.foot.ly * n
            A[r0 + 5, T], A[r0 + 5, F] = -t1, -self.foot.ly * n
            A[r0 + 6, T], A[r0 + 6, F] = t2, -self.foot.lx_back * n
            A[r0 + 7, T], A[r0 + 7, F] = -t2, -self.foot.lx_front * n
            A[r0 + 8, T], A[r0 + 8, F] = n, -mu_tau * n
            A[r0 + 9, T], A[r0 + 9, F] = -n, -mu_tau * n
        return A

    # -- solve --------------------------------------------------------------------

    def solve(self, g1: PinG1Model, contact: np.ndarray, force_ref: np.ndarray,
              swing_ref: dict, com_ref: tuple, torso_ref: tuple,
              normals: dict[str, np.ndarray] | None = None,
              tau_limit: np.ndarray | None = None) -> WBCOutput:
        """One control tick.

        contact    (2,) bool, stance mask for (LEFT, RIGHT)
        force_ref  (12,) MPC wrench, [left 6, right 6]
        swing_ref  {leg: (pos_des, vel_des, acc_des)} for legs in swing
        com_ref    (pos_des, vel_des, acc_des)
        torso_ref  (rpy_des, omega_des)
        normals    {leg: contact normal}; defaults to +z
        """
        t0 = time.perf_counter()
        p = self.p
        nv, nz = self.nv, self.nz
        q, dq = g1.current_config.get_q(), g1.current_config.get_dq()
        g_vec, C, M = g1.compute_dynamics_terms()
        b = C @ dq + g_vec
        normals = normals or {name: np.array([0.0, 0.0, 1.0]) for name in LEGS}
        tau_limit = (np.tile(np.asarray(self.leg_p.q_nominal) * 0 + 1e9, 2)
                     if tau_limit is None else np.asarray(tau_limit, dtype=float))

        Jc, Jc_drift = self._contact_jacobians(g1)
        Jc_stack = np.vstack([Jc["LEFT"], Jc["RIGHT"]])          # (12, nv)

        # --- cost: H = sum w A^T A, g = -sum w A^T rhs
        H = np.zeros((nz, nz))
        gvec = np.zeros(nz)

        def add(A, rhs, w):
            nonlocal H, gvec
            H += w * (A.T @ A)
            gvec += -w * (A.T @ np.asarray(rhs).reshape(-1))

        Af = np.zeros((12, nz))
        Af[np.arange(12), self.iF + np.arange(12)] = 1.0
        add(Af, force_ref, p.w_contact_force)

        for name, ref in swing_ref.items():
            A, rhs = self._swing_task(g1, name, *ref)
            add(A, rhs, p.w_swing)

        A, rhs = self._com_task(g1, *com_ref)
        add(A, rhs, p.w_com)
        A, rhs = self._torso_task(g1, *torso_ref)
        add(A, rhs, p.w_torso)
        A, rhs = self._posture_task(g1)
        add(A, rhs, p.w_posture)

        H[np.arange(nv), np.arange(nv)] += p.w_reg_qddot
        H[self.iF + np.arange(12), self.iF + np.arange(12)] += p.w_reg_force
        H = 0.5 * (H + H.T) + 1e-9 * np.eye(nz)

        # --- equalities
        rows, lb, ub = [], [], []

        A_base = np.zeros((6, nz))
        A_base[:, :nv] = M[:6, :]
        A_base[:, self.iF:] = -Jc_stack.T[:6, :]
        rows.append(A_base); lb.append(-b[:6]); ub.append(-b[:6])

        if p.freeze_upper_body and len(self.upper_cols):
            A_up = np.zeros((len(self.upper_cols), nz))
            A_up[np.arange(len(self.upper_cols)), self.upper_cols] = 1.0
            z = np.zeros(len(self.upper_cols))
            rows.append(A_up); lb.append(z); ub.append(z)

        for li, name in enumerate(LEGS):
            A_c = np.zeros((6, nz))
            if contact[li]:
                A_c[:, :nv] = Jc[name]
                rhs = -Jc_drift[name]
            else:
                # Swing foot: no contact constraint, force pinned to zero instead.
                A_c[np.arange(6), self.iF + 6 * li + np.arange(6)] = 1.0
                rhs = np.zeros(6)
            rows.append(A_c); lb.append(rhs); ub.append(rhs)

        # --- inequalities
        A_fc = self._friction_cop_rows(normals)
        fc_lo = np.full(2 * N_INEQ_PER_LEG, -np.inf)
        fc_hi = np.zeros(2 * N_INEQ_PER_LEG)
        for li in range(2):
            if not contact[li]:
                fc_hi[N_INEQ_PER_LEG * li:N_INEQ_PER_LEG * (li + 1)] = np.inf
        rows.append(A_fc); lb.append(fc_lo); ub.append(fc_hi)

        S = np.zeros((12, nv))
        S[np.arange(12), self.leg_cols] = 1.0
        A_tau = np.zeros((12, nz))
        A_tau[:, :nv] = S @ M
        A_tau[:, self.iF:] = -(S @ Jc_stack.T)
        rows.append(A_tau); lb.append(-tau_limit - S @ b); ub.append(tau_limit - S @ b)
        rows.append(-A_tau); lb.append(np.full(12, -np.inf)); ub.append(tau_limit + S @ b)

        A_all = np.vstack(rows)
        lba = np.concatenate(lb)
        uba = np.concatenate(ub)

        args = {"h": ca.DM(H), "g": ca.DM(gvec.reshape(-1, 1)), "a": ca.DM(A_all),
                "lba": ca.DM(lba.reshape(-1, 1)), "uba": ca.DM(uba.reshape(-1, 1))}
        if self._z_prev is not None:
            args["x0"] = self._z_prev
        sol = self.solver(**args)
        self._z_prev = sol["x"]
        z = np.asarray(sol["x"]).reshape(-1)

        qddot = z[:nv]
        forces = z[self.iF:]
        tau = S @ (M @ qddot + b - Jc_stack.T @ forces)

        if p.kp_joint > 0.0:
            q_leg = np.concatenate([g1.current_config.left_leg_angle,
                                    g1.current_config.right_leg_angle])
            dq_leg = np.concatenate([g1.current_config.left_leg_vel,
                                     g1.current_config.right_leg_vel])
            q_ref = np.tile(self.q_nominal, 2)
            pd = p.kp_joint * (q_ref - q_leg) - p.kd_joint * dq_leg
            for li in range(2):
                if contact[li] or p.joint_pd_on_swing:
                    tau[6 * li:6 * (li + 1)] += pd[6 * li:6 * (li + 1)]

        self.solve_ms = (time.perf_counter() - t0) * 1e3
        self.last_status = str(self.solver.stats().get("return_status", ""))
        ok = np.all(np.isfinite(tau))
        return WBCOutput(tau=tau, qddot=qddot, forces=forces,
                         solve_ms=self.solve_ms, ok=bool(ok))
