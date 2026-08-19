
from __future__ import annotations

import time
from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path

import mujoco as mj
import mujoco.viewer as mjv
import numpy as np

from bheema.centroidal_mpc import CentroidalMPC
from bheema.com_traj import ComTraj
from bheema.g1_config import PinG1Model
from bheema.g1_mujoco import MuJoCo_G1_Model
from bheema.gait import Gait
from bheema.leg_controller import LegController
from bheema.params import Params

UNITREE_DIR = Path(__file__).parent.parent / "unitree_g1"
DEFAULT_SCENE = str(UNITREE_DIR / "scene_with_hands.xml")

LEG_SLICE = {"LEFT": slice(0, 6), "RIGHT": slice(6, 12)}

# Upper-body posture setpoints, written every sim step.
#
# indefinitely; without them it falls by t = 2 s. The legs alone do not stabilise the
# torso, because there is no whole-body controller yet -- the waist and shoulders are
@dataclass
class BodyCmdPhase:
    """A commanded body velocity/height over a time window."""
    t_start: float
    t_end: float
    x_vel: float
    y_vel: float
    z_pos: float
    yaw_rate: float


NOMINAL_Z = Params().body.nominal_com_height

# The schedule main.py shipped with. Note it stands still for the first 15 s and then
# jumps to 1.0 m/s -- fine for a demo, poor for iterating on terrain. See
# WALK_CMD_SCHEDULE for a terrain-development alternative.
DEFAULT_CMD_SCHEDULE = (
    BodyCmdPhase(0.0, 3.0, 0.0, 0.0, NOMINAL_Z, 0.0),     # Stand still
    BodyCmdPhase(15.0, 20.0, 0.3, 0.0, NOMINAL_Z, 0.0),   # Walk forward (warmup)
    BodyCmdPhase(20.0, 40.0, 1.0, 0.0, NOMINAL_Z, 0.0),   # Run forward
    BodyCmdPhase(55.0, 80.0, 1.0, 0.0, NOMINAL_Z, 0.0),   # Run forward
)

# Short stand, then a steady 0.3 m/s. Reaches an obstacle at x ~ 1.2 m by t ~ 6 s, so a
# terrain iteration is seconds rather than a minute.
WALK_CMD_SCHEDULE = (
    BodyCmdPhase(0.0, 2.0, 0.0, 0.0, NOMINAL_Z, 0.0),
    BodyCmdPhase(2.0, 600.0, 0.3, 0.0, NOMINAL_Z, 0.0),
)

# Hardware-matched actuator limits: hip_p, hip_r, hip_y, knee, ankle_p, ankle_r per leg.
@dataclass
class SimConfig:
    scene: str = DEFAULT_SCENE
    duration_s: float = 120.0

    sim_hz: int = 2000
    ctrl_hz: int = 200
    render_hz: float = 120.0

    params: Params = field(default_factory=Params)

    initial_xy: tuple[float, float] = (0.0, 0.0)
    cmd_schedule: tuple[BodyCmdPhase, ...] = DEFAULT_CMD_SCHEDULE

    headless: bool = False
    verbose: bool = True
    # None => pace to wall-clock only when a viewer is open. Explicit True/False overrides.
    realtime: bool | None = None
    # Off by default so the default run is byte-for-byte what it always was. The benchmark
    # harness turns it on to avoid simulating 100 s of a robot lying on the floor.

    stop_on_fall: bool = False
    fall_z: float = 0.40
    # Terrain source for the footstep planner. "oracle" ray-casts the sim; None reproduces
    # the pre-Phase-1 behaviour (every foothold planned at z = 0). On flat ground the two

    def __post_init__(self):
        if self.sim_hz % self.ctrl_hz != 0:
            raise ValueError(
                f"sim_hz ({self.sim_hz}) must be divisible by ctrl_hz ({self.ctrl_hz})"
            )
        self.sim_dt = 1.0 / self.sim_hz
        self.ctrl_dt = 1.0 / self.ctrl_hz
        self.ctrl_decim = self.sim_hz // self.ctrl_hz
        self.sim_steps = int(self.duration_s * self.sim_hz)
        self.ctrl_steps = int(self.duration_s * self.ctrl_hz)

        self.gait_period = self.params.gait_period
        self.mpc_dt = self.params.mpc_dt
        self.mpc_hz = 1.0 / self.mpc_dt
        self.steps_per_mpc = max(1, int(self.ctrl_hz // self.mpc_hz))
        self.tau_lim = np.tile(np.asarray(self.params.body.tau_limit_leg, dtype=float), 2)
        self.nominal_z = self.params.body.nominal_com_height

        self.render_every = max(1, int(self.sim_hz / self.render_hz))

    @property
    def paces_wallclock(self) -> bool:
        return (not self.headless) if self.realtime is None else self.realtime


@dataclass
class FootTrajLog:
    """Desired vs actual foot state, 12 rows = [left xyz + 3 pad, right xyz + 3 pad]."""
    pos_des: np.ndarray
    pos_now: np.ndarray
    vel_des: np.ndarray
    vel_now: np.ndarray

    @classmethod
    def alloc(cls, n: int) -> "FootTrajLog":
        return cls(*(np.zeros((12, n)) for _ in range(4)))

    def trim(self, n: int) -> "FootTrajLog":
        return FootTrajLog(self.pos_des[:, :n], self.pos_now[:, :n],
                           self.vel_des[:, :n], self.vel_now[:, :n])


@dataclass
class SimResult:
    cfg: SimConfig
    n_ticks: int
    t: np.ndarray                  # (n,) control-tick times, from sim clock
    x_vec: np.ndarray              # (12, n) CoM state
    mpc_force_world: np.ndarray    # (12, n) applied MPC wrench
    tau_raw: np.ndarray            # (12, n) before clipping
    tau_cmd: np.ndarray            # (12, n) after clipping
    q_log: np.ndarray              # (n, nq)
    com_z: np.ndarray              # (n_mpc,) actual CoM z at each MPC solve
    ref_z: np.ndarray              # (n_mpc,) reference CoM z at each MPC solve
    mpc_solve_time_ms: np.ndarray
    mpc_update_time_ms: np.ndarray
    foot: FootTrajLog
    pelvis_z: np.ndarray           # (n,) convenience: did it stay up?
    # One entry per swing takeoff: (t, leg, td_x, td_y, td_z_planned). The planner's
    # requested foothold. Comparing td_z_planned against the true surface height there is
    touchdowns: list[tuple[float, str, float, float, float]]
    fell: bool
    fall_time_s: float | None
    interrupted: bool
    wall_time_s: float
    sim_time_s: float


def get_body_cmd(t: float, schedule, nominal_z: float):
    """Commanded (x_vel, y_vel, z_pos, yaw_rate) at time t; zeros outside any phase."""
    for phase in schedule:
        if phase.t_start <= t < phase.t_end:
            return phase.x_vel, phase.y_vel, phase.z_pos, phase.yaw_rate
    return 0.0, 0.0, nominal_z, 0.0


def run(cfg: SimConfig) -> SimResult:
    P = cfg.params
    g1 = PinG1Model(body=P.body, leg=P.leg)
    mujoco_g1 = MuJoCo_G1_Model(xml_path=cfg.scene, z_offset=P.body.pin_mujoco_z_offset)
    leg_controller = LegController(params=P.leg)
    traj = ComTraj(g1, params=P.traj, mpc=P.mpc)
    gait = Gait(params=P.gait, foot=P.foot)

    model, data = mujoco_g1.model, mujoco_g1.data


    q_init, _ = g1.get_full_q_dq()
    q_init[0], q_init[1] = cfg.initial_xy
    mujoco_g1.update_with_q_pin(q_init)
    model.opt.timestep = cfg.sim_dt

    x_vel, y_vel, z_pos, yaw_rate = get_body_cmd(0.0, cfg.cmd_schedule, cfg.nominal_z)
    traj.generate_traj(g1, gait, 0.0, x_vel, y_vel, z_pos, yaw_rate, time_step=cfg.mpc_dt)
    mpc = CentroidalMPC(g1, traj, params=P.mpc, foot=P.foot)

    n = cfg.ctrl_steps
    x_vec = np.zeros((12, n))
    mpc_force_world = np.zeros((12, n))
    tau_raw = np.zeros((12, n))
    tau_cmd = np.zeros((12, n))
    q_log = np.zeros((n, model.nq))
    t_log = np.zeros(n)
    pelvis_z = np.zeros(n)
    foot = FootTrajLog.alloc(n)
    com_z_log: list[float] = []
    ref_z_log: list[float] = []
    solve_ms: list[float] = []
    update_ms: list[float] = []

    # Cache the upper-body actuator ids. main.py looked these up with mj_name2id on every
    # sim step -- 5 lookups x 2000 Hz.
    holds = (("waist_yaw_joint", P.body.waist_yaw),
             ("waist_roll_joint", P.body.waist_roll),
             ("waist_pitch_joint", P.body.waist_pitch),
             ("left_shoulder_pitch_joint", P.body.shoulder_pitch),
             ("right_shoulder_pitch_joint", P.body.shoulder_pitch))
    upper_body = [(aid, val) for aid, val in
                  ((mj.mj_name2id(model, mj.mjtObj.mjOBJ_ACTUATOR, nm), val)
                   for nm, val in holds) if aid != -1]

    U_opt = np.zeros((12, traj.N))
    tau_hold = np.zeros(12)
    ctrl_i = 0
    last_mpc_time = 0.0          # explicit: do not rely on the first tick being an MPC tick
    fell = False
    fall_time_s: float | None = None
    interrupted = False
    touchdowns: list[tuple[float, str, float, float, float]] = []
    prev_td: dict[str, np.ndarray | None] = {"LEFT": None, "RIGHT": None}

    if cfg.verbose:
        print(f"Running Biped Simulation for {cfg.duration_s}s"
              f"{' (headless)' if cfg.headless else ''}")
        _print_actuator_gains(mujoco_g1)

    wall_start = time.perf_counter()
    with ExitStack() as stack:
        viewer = None
        if not cfg.headless:
            viewer = stack.enter_context(mjv.launch_passive(model, data))
            viewer.cam.type = mj.mjtCamera.mjCAMERA_TRACKING
            viewer.cam.trackbodyid = mujoco_g1.base_bid
            viewer.cam.distance = 2.5
            viewer.cam.elevation = -20
            viewer.cam.azimuth = 90
            viewer.opt.flags[mj.mjtVisFlag.mjVIS_CONTACTPOINT] = True

        try:
            for k in range(cfg.sim_steps):
                if viewer is not None and not viewer.is_running():
                    break

                time_now_s = float(data.time)

                if (k % cfg.ctrl_decim) == 0 and ctrl_i < n:
                    x_vel, y_vel, z_pos, yaw_rate = get_body_cmd(
                        time_now_s, cfg.cmd_schedule, cfg.nominal_z
                    )

                    mujoco_g1.update_pin_with_mujoco(g1)
                    x_vec[:, ctrl_i] = g1.compute_com_x_vec().reshape(-1)
                    t_log[ctrl_i] = time_now_s
                    q_log[ctrl_i, :] = data.qpos
                    pelvis_z[ctrl_i] = data.qpos[2]

                    if (ctrl_i % cfg.steps_per_mpc) == 0:
                        if cfg.verbose and not cfg.headless:
                            print(f"\rSimulation Time: {time_now_s:.3f} s", end="", flush=True)

                        traj.generate_traj(g1, gait, time_now_s, x_vel, y_vel, z_pos,
                                           yaw_rate, time_step=cfg.mpc_dt)
                        com_z_log.append(float(g1.compute_com_x_vec().flatten()[2]))
                        ref_z_log.append(float(traj.compute_x_ref_vec()[2, 0]))

                        sol = mpc.solve_QP(g1, traj, False)
                        solve_ms.append(mpc.solve_time)
                        update_ms.append(mpc.update_time)

                        w_opt = sol["x"].full().flatten()
                        U_opt = w_opt[12 * traj.N:].reshape((12, traj.N), order="F")
                        last_mpc_time = time_now_s

                    # Standing: freeze gait phase so both feet stay in stance.
                    gait_time = 0.0 if (x_vel == 0.0 and y_vel == 0.0) else time_now_s

                    k_interp = min(int((time_now_s - last_mpc_time) / cfg.mpc_dt), traj.N - 1)
                    mpc_force_world[:, ctrl_i] = U_opt[:, k_interp]

                    for name in ("LEFT", "RIGHT"):
                        sl = LEG_SLICE[name]
                        out = leg_controller.compute_leg_torque(
                            name, g1, gait, mpc_force_world[sl, ctrl_i], gait_time
                        )
                        tau_raw[sl, ctrl_i] = out.tau
                        foot.pos_des[sl, ctrl_i] = np.pad(out.pos_des, (0, 3))
                        foot.pos_now[sl, ctrl_i] = np.pad(out.pos_now, (0, 3))
                        foot.vel_des[sl, ctrl_i] = np.pad(out.vel_des, (0, 3))
                        foot.vel_now[sl, ctrl_i] = np.pad(out.vel_now, (0, 3))

                        # LegController stashes the swing target at takeoff; a change in
                        # it means a new footstep was planned.
                        td = getattr(leg_controller, f"{name}_td_pos", None)
                        if td is not None and (
                            prev_td[name] is None or not np.array_equal(td, prev_td[name])
                        ):
                            prev_td[name] = np.array(td, dtype=float)
                            touchdowns.append(
                                (time_now_s, name, float(td[0]), float(td[1]), float(td[2]))
                            )

                    tau_cmd[:, ctrl_i] = np.clip(tau_raw[:, ctrl_i], -cfg.tau_lim, cfg.tau_lim)
                    tau_hold = tau_cmd[:, ctrl_i].copy()
                    ctrl_i += 1

                mujoco_g1.set_joint_torque(tau_hold)
                for aid, val in upper_body:
                    data.ctrl[aid] = val

                mj.mj_step(model, data)

                if not fell and data.qpos[2] < cfg.fall_z:
                    fell, fall_time_s = True, float(data.time)
                    if cfg.stop_on_fall:
                        break

                if (k % cfg.render_every) == 0:
                    if viewer is not None:
                        viewer.sync()
                    if cfg.paces_wallclock:
                        slack = data.time - (time.perf_counter() - wall_start)
                        if slack > 0:
                            time.sleep(slack)
        except KeyboardInterrupt:
            interrupted = True
            if cfg.verbose:
                print("\n[sim] interrupted -- returning partial result")

    wall_time_s = time.perf_counter() - wall_start
    if cfg.verbose:
        print(f"\nSimulation ended."
              f"\nElapsed wall time: {wall_time_s:.3f}s"
              f"\nSim time: {float(data.time):.3f}s"
              f"\nControl ticks: {ctrl_i}/{n}"
              + (f"\nFELL at t={fall_time_s:.2f}s" if fell else ""))

    return SimResult(
        cfg=cfg,
        n_ticks=ctrl_i,
        t=t_log[:ctrl_i],
        x_vec=x_vec[:, :ctrl_i],
        mpc_force_world=mpc_force_world[:, :ctrl_i],
        tau_raw=tau_raw[:, :ctrl_i],
        tau_cmd=tau_cmd[:, :ctrl_i],
        q_log=q_log[:ctrl_i, :],
        com_z=np.asarray(com_z_log),
        ref_z=np.asarray(ref_z_log),
        mpc_solve_time_ms=np.asarray(solve_ms),
        mpc_update_time_ms=np.asarray(update_ms),
        foot=foot.trim(ctrl_i),
        pelvis_z=pelvis_z[:ctrl_i],
        touchdowns=touchdowns,
        fell=fell,
        fall_time_s=fall_time_s,
        interrupted=interrupted,
        wall_time_s=wall_time_s,
        sim_time_s=float(data.time),
    )


def _print_actuator_gains(mujoco_g1: MuJoCo_G1_Model) -> None:
    """The actuator dump main.py printed at startup, kept for parity."""
    for i in range(12):
        aid = mujoco_g1.actuator_ids[i]
        name = mujoco_g1.leg_joint_names[i]
        m = mujoco_g1.model
        print(f"{name}: gain={m.actuator_gainprm[aid][0]:.1f}, "
              f"biastype={m.actuator_biastype[aid]}, gaintype={m.actuator_gaintype[aid]}, "
              f"biasprm={m.actuator_biasprm[aid][:3]}")
