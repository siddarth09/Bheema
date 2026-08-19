"""
Controller parameters.

Grouped by subsystem. Every field is exposed as a command-line flag by add_arguments()
and rebuilt by from_args(), so any parameter can be swept without editing source:

    python -m bheema.main --gait-frequency-hz 1.1 --mpc-q-rpy 3000,3000,900
    python -m bheema.benchmark bench --leg-kp-stance 220

Vector fields are tuples so the dataclasses stay hashable; convert with np.asarray at
the point of use.
"""

from __future__ import annotations

import argparse
from dataclasses import MISSING, dataclass, field, fields, is_dataclass
from typing import get_args, get_origin


# --------------------------------------------------------------------------------
# Physical geometry shared by several subsystems
# --------------------------------------------------------------------------------

@dataclass(frozen=True)
class FootParams:
    """Sole geometry relative to the ankle-roll frame.

    Used by the MPC for centre-of-pressure limits and by the gait planner for toe
    clearance.
    """
    lx_front: float = 0.12   # m, toe reach ahead of the ankle frame
    lx_back: float = 0.05    # m, heel reach behind the ankle frame
    ly: float = 0.05         # m, half-width


# --------------------------------------------------------------------------------
# Centroidal MPC
# --------------------------------------------------------------------------------

@dataclass(frozen=True)
class MPCParams:
    """Convex QP: state weights, input weights, contact constraints, solver settings."""

    # State cost, diag(Q) grouped by physical meaning. State is
    # [pos(3), rpy(3), vel(3), omega(3)] of the CoM in world frame.
    q_pos: tuple[float, float, float] = (200.0, 1000.0, 3000.0)      # x, y, z
    q_rpy: tuple[float, float, float] = (5000.0, 5000.0, 900.0)      # roll, pitch, yaw
    q_vel: tuple[float, float, float] = (500.0, 1000.0, 200.0)       # vx, vy, vz
    q_omega: tuple[float, float, float] = (100.0, 100.0, 100.0)      # wx, wy, wz

    # Input cost, per foot. Input is a 6D wrench [F(3), tau(3)] at each foot.
    # fz is cheap (1e-2) because vertical force is what carries the robot.
    r_force: tuple[float, float, float] = (1.0, 1.0, 1e-2)
    r_torque: tuple[float, float, float] = (10.0, 10.0, 10.0)

    mu: float = 0.8          # linear friction coefficient (pyramid, not cone)
    mu_tau: float = 0.1      # torsional friction: yaw torque per unit normal force

    fz_min: float = 23.0     # N, minimum normal force on a stance foot (prevents slip)
    fz_max: float = 1200.0   # N, ceiling on normal force (prevents force spikes)
    f_xy_max: float = 400.0  # N, cap on horizontal stance force (limits aggressiveness)

    horizon_periods: float = 1.5   # MPC horizon length, in gait periods
    horizon_steps_per_period: int = 32  # discretisation: mpc_dt = period / this

    # OSQP settings. Loose tolerances: the QP is re-solved every 20 ms.
    osqp_eps_abs: float = 1e-2
    osqp_eps_rel: float = 1e-2
    osqp_max_iter: int = 1400
    osqp_polish: bool = False
    osqp_scaling: int = 5
    osqp_check_termination: int = 10
    osqp_adaptive_rho_interval: int = 25


# --------------------------------------------------------------------------------
# Gait scheduling and footstep planning
# --------------------------------------------------------------------------------

@dataclass(frozen=True)
class GaitParams:
    """Contact schedule and Raibert footstep heuristic."""

    frequency_hz: float = 1.3     # gait cycles per second
    duty: float = 0.65            # fraction of the cycle each foot spends in stance
    phase_offset: tuple[float, float] = (0.0, 0.5)   # (left, right), fraction of a cycle

    nominal_stance_width: float = 0.25   # m, lateral spacing of the two hips
    min_foot_gap: float = 0.15           # m, floor on lateral spacing (self-collision)

    swing_height: float = 0.18           # m, minimum swing apex above the footholds
    swing_clearance_margin: float = 0.05 # m, extra apex above intervening terrain
    max_swing_apex: float = 0.45         # m, kinematic sanity cap
    clearance_samples: int = 17          # samples along the swing for the toe check
    clearance_min_bump: float = 0.02     # ignore samples where the apex basis b(s) < this

    # Raibert heuristic (Cheetah paper Eq. 12-15). The velocity gains are scaled by
    # T = t_swing + stance_frac_in_T * t_stance, so they stay dimensionally consistent
    # if the gait frequency changes.
    k_vel_x_scale: float = 1.1    # multiplies T to give the x velocity feedback gain
    k_pos_x: float = 0.3          # x position feedback gain
    k_vel_y_scale: float = 0.5    # multiplies T to give the y velocity feedback gain
    k_pos_y: float = 0.05         # y position feedback gain
    stance_frac_in_T: float = 0.5 # how much stance time enters the prediction horizon T
    pred_time_frac: float = 0.5   # touchdown is predicted at pred_time_frac * T

    # Event-based contact. The phase clock is corrected by measured contact instead of
    # running open-loop; with no events the correction stays zero.
    contact_force_threshold: float = 20.0  # N, normal force counting as contact
    early_touchdown_min_frac: float = 0.55 # ignore contact before this much of swing is done
    late_hold_max_frac: float = 0.60       # cap the swing extension, in swing durations


# --------------------------------------------------------------------------------
# Reference trajectory
# --------------------------------------------------------------------------------

@dataclass(frozen=True)
class TrajParams:
    """CoM reference: sway, height, and how far the reference may lead the robot."""

    max_pos_error: float = 0.15        # m, clamp on how far the xy reference may lead
    sway_filter_alpha: float = 0.85    # lateral ZMP sway low-pass; 1.0 = never moves
    sway_fraction: float = 0.6         # sway target as a fraction of half stance width
    height_filter_alpha: float = 0.90  # support-height low-pass, applied across solves


# --------------------------------------------------------------------------------
# Leg control (operational-space impedance + stance wrench mapping)
# --------------------------------------------------------------------------------

@dataclass(frozen=True)
class LegControlParams:
    """Swing: 6D operational-space impedance. Stance: J^T wrench + posture PD."""

    # Cartesian impedance gains, [x, y, z, rx, ry, rz]. Translational gains are far
    # stiffer than rotational because foot position matters more than foot orientation.
    kp_swing: tuple[float, ...] = (2500.0, 3500.0, 1000.0, 400.0, 400.0, 400.0)
    kd_swing: tuple[float, ...] = (120.0, 120.0, 120.0, 10.0, 10.0, 10.0)

    # Posture PD. q_nominal sets the bent-knee crouch as well as anchoring drift.
    kp_stance: float = 150.0
    kd_stance: float = 30.0
    q_nominal: tuple[float, ...] = (-0.3, 0.0, 0.0, 0.6, -0.3, 0.0)
    # [hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll]



# --------------------------------------------------------------------------------
# Robot / body
# --------------------------------------------------------------------------------

@dataclass(frozen=True)
class BodyParams:
    """Whole-body setpoints and actuator limits."""

    nominal_com_height: float = 0.66   # m, CoM height above the support surface

    # Hardware-matched torque limits, per leg:
    # [hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll]
    tau_limit_leg: tuple[float, ...] = (88.0, 139.0, 88.0, 139.0, 50.0, 50.0)

    # remove them and the robot falls by t = 2 s, because the legs alone do not stabilise
    # the torso without a task-space posture objective.
    waist_yaw: float = 0.0
    waist_roll: float = 0.0
    waist_pitch: float = 0.0
    shoulder_pitch: float = 0.15

    # Pinocchio's MJCF parser treats the freejoint as body-relative in 3.x, absolute in
    # 4.x. 0.793 for pinocchio 3.x, 0.0 for >= 4.x.
    # Prefer measuring it at startup over trusting this default.
    pin_mujoco_z_offset: float = 0.793


# --------------------------------------------------------------------------------
# Aggregate
# --------------------------------------------------------------------------------

@dataclass(frozen=True)
class Params:
    """All controller parameters."""
    foot: FootParams = field(default_factory=FootParams)
    mpc: MPCParams = field(default_factory=MPCParams)
    gait: GaitParams = field(default_factory=GaitParams)
    traj: TrajParams = field(default_factory=TrajParams)
    leg: LegControlParams = field(default_factory=LegControlParams)
    body: BodyParams = field(default_factory=BodyParams)

    @property
    def gait_period(self) -> float:
        return 1.0 / self.gait.frequency_hz

    @property
    def stance_time(self) -> float:
        return self.gait.duty * self.gait_period

    @property
    def swing_time(self) -> float:
        return (1.0 - self.gait.duty) * self.gait_period

    @property
    def mpc_dt(self) -> float:
        return self.gait_period / self.mpc.horizon_steps_per_period

    @property
    def horizon_seconds(self) -> float:
        return self.gait_period * self.mpc.horizon_periods


# --------------------------------------------------------------------------------
# --------------------------------------------------------------------------------

def _parse_tuple(text: str) -> tuple[float, ...]:
    return tuple(float(v) for v in text.replace(" ", "").split(","))


def _flag(group: str, name: str) -> str:
    return f"--{group}-{name}".replace("_", "-")


def _dest(group: str, name: str) -> str:
    return f"{group}__{name}"


def add_arguments(parser: argparse.ArgumentParser, params: Params | None = None) -> None:
    """Add one flag per parameter field, grouped by subsystem.

    Generated from the dataclass definitions, so a new field is exposed automatically.
    """
    params = params or Params()
    for grp in fields(Params):
        sub = getattr(params, grp.name)
        if not is_dataclass(sub):
            continue
        section = parser.add_argument_group(f"{grp.name} parameters")
        for f in fields(sub):
            cur = getattr(sub, f.name)
            dest = _dest(grp.name, f.name)
            flag = _flag(grp.name, f.name)
            if get_origin(f.type) is tuple or isinstance(cur, tuple):
                n = len(cur)
                section.add_argument(flag, dest=dest, type=_parse_tuple, default=None,
                                     metavar=",".join(["v"] * n),
                                     help=f"{n} comma-separated values (default {list(cur)})")
            elif f.type is bool or isinstance(cur, bool):
                section.add_argument(flag, dest=dest, type=lambda s: s.lower() not in
                                     ("0", "false", "no"), default=None,
                                     metavar="BOOL", help=f"(default {cur})")
            elif f.type is int or isinstance(cur, int):
                section.add_argument(flag, dest=dest, type=int, default=None,
                                     help=f"(default {cur})")
            else:
                section.add_argument(flag, dest=dest, type=float, default=None,
                                     help=f"(default {cur})")


def from_args(args: argparse.Namespace, base: Params | None = None) -> Params:
    """Build Params from parsed args; any flag left unset keeps its default."""
    base = base or Params()
    groups = {}
    for grp in fields(Params):
        sub = getattr(base, grp.name)
        if not is_dataclass(sub):
            groups[grp.name] = sub
            continue
        overrides = {}
        for f in fields(sub):
            val = getattr(args, _dest(grp.name, f.name), None)
            if val is not None:
                overrides[f.name] = val
        groups[grp.name] = type(sub)(**{**{f.name: getattr(sub, f.name)
                                          for f in fields(sub)}, **overrides})
    return Params(**groups)


def describe(params: Params) -> str:
    """Human-readable dump, for logging what a run actually used."""
    lines = []
    for grp in fields(Params):
        sub = getattr(params, grp.name)
        if not is_dataclass(sub):
            continue
        lines.append(f"[{grp.name}]")
        for f in fields(sub):
            lines.append(f"  {f.name:<28} {getattr(sub, f.name)}")
    lines.append("[derived]")
    for k in ("gait_period", "stance_time", "swing_time", "mpc_dt", "horizon_seconds"):
        lines.append(f"  {k:<28} {getattr(params, k):.6f}")
    return "\n".join(lines)


def diff(params: Params, base: Params | None = None) -> dict[str, tuple]:
    """Fields that differ from the defaults, as {"group.field": (default, actual)}."""
    base = base or Params()
    out = {}
    for grp in fields(Params):
        a, b = getattr(base, grp.name), getattr(params, grp.name)
        if not is_dataclass(a):
            continue
        for f in fields(a):
            va, vb = getattr(a, f.name), getattr(b, f.name)
            if va != vb:
                out[f"{grp.name}.{f.name}"] = (va, vb)
    return out
