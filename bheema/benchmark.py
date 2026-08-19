"""
Benchmark and regression harness.

Two modes:

    pixi run check      # regression gate: determinism + parity against the recorded baseline
    pixi run bench      # sweep the terrain scenes and print a metrics table

`check` is the one to run after touching anything in the control path. It asserts the
things that are cheap to break and expensive to notice: run-to-run determinism, that the
generated flat scene is byte-identical to the original hand-written scene, and that
standing still holds the baseline pelvis height.

td_err_cm is the planner's requested foothold height minus the true surface height there.
It equals the obstacle height while footholds are planned on the ground plane.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mujoco as mj
import numpy as np

from bheema import terrain as T
from bheema.params import add_arguments, diff, from_args
from bheema.sim import (DEFAULT_SCENE, WALK_CMD_SCHEDULE, SimConfig, SimResult, run)

SCENES_DIR = Path(__file__).parent.parent / "unitree_g1" / "scenes"

BASELINE_STAND_Z = 0.7634
STAND_TOL = 1e-3
# Flat-ground forward progress over 16 s at 0.3 m/s. This exists because the original
# every assert. A behavioural anchor on flat ground is the thing that catches it.
BASELINE_FLAT_X = 2.67
FLAT_X_TOL = 0.05

# Scenes to sweep, in the diagnostic ladder order from easiest to hardest.
SWEEP = ("flat", "platform", "ramp", "stairs", "gap_field", "beam",
         "stepping_stones", "atlas_gym")


# --------------------------------------------------------------------------------
# Ground truth queries
# --------------------------------------------------------------------------------

def ground_height(model, data, x: float, y: float, z_from: float = 3.0) -> float:
    """True terrain height at (x, y) by downward ray, masked to the terrain geom group.

    This duplicates a fraction of what SurfaceQuery will do, deliberately: the benchmark
    must measure against ground truth independently of the controller's own terrain
    estimate, or a bug in SurfaceQuery would hide itself in its own gate.
    """
    grp = np.zeros(6, dtype=np.uint8)
    grp[T.TERRAIN_GEOM_GROUP] = 1
    gid = np.zeros(1, dtype=np.int32)
    dist = mj.mj_ray(model, data, np.array([x, y, z_from]), np.array([0.0, 0.0, -1.0]),
                     grp, 1, -1, gid)
    return float(z_from - dist) if dist >= 0 else float("nan")


def _scene_probe(res: SimResult):
    model = mj.MjModel.from_xml_path(res.cfg.scene)
    data = mj.MjData(model)
    mj.mj_forward(model, data)
    return model, data


def touchdown_errors(res: SimResult, probe=None) -> np.ndarray:
    """Planned foothold height minus true surface height, per footstep, in metres."""
    model, data = probe or _scene_probe(res)
    return np.array([
        ground_height(model, data, x, y) - z_planned
        for _, _, x, y, z_planned in res.touchdowns
    ])


def com_height_error(res: SimResult, probe=None) -> np.ndarray:
    """CoM height error measured RELATIVE TO THE TERRAIN under it, in metres.

    Measuring against a world-frame nominal is wrong the moment the robot leaves z=0: on a
    15 cm box a perfectly-tracking CoM reads as 15 cm of error. That would make a correct
    controller look worse than a broken one -- the same class of mistake as scoring a scene
    `ok` because the robot never fell while never moving.
    """
    model, data = probe or _scene_probe(res)
    if not res.n_ticks:
        return np.array([0.0])
    com = res.x_vec[:3, :]
    ref = np.array([ground_height(model, data, com[0, i], com[1, i]) + res.cfg.nominal_z
                    for i in range(res.n_ticks)])
    return com[2, :] - ref


# --------------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------------

def first_obstacle_x(scene: "T.TerrainScene") -> float:
    """Leading x edge of the first element that is actually raised off the floor."""
    raised = [b.pos[0] - b.size[0] for b in scene.boxes if b.pos[2] + b.size[2] > 0.01]
    return min(raised) if raised else float("inf")


def stalled(res: SimResult, window_s: float = 2.0, min_speed: float = 0.05) -> bool:
    """Commanded to walk but not moving: pushing against an obstacle face without falling.

    Without this, a scene where the robot jams against a step reports `ok` -- it never fell
    and never planned a foothold on the obstacle, so `td_err_cm` is a vacuous 0.0.
    """
    if res.n_ticks < 2:
        return False
    n = max(2, int(window_s / res.cfg.ctrl_dt))
    x = res.q_log[-n:, 0]
    span = res.t[-1] - res.t[-n]
    if span <= 0:
        return False
    return abs(x[-1] - x[0]) / span < min_speed


def metrics(res: SimResult) -> dict:
    probe = _scene_probe(res)
    err = touchdown_errors(res, probe)
    finite = err[np.isfinite(err)]
    dz = com_height_error(res, probe)
    dz = dz[np.isfinite(dz)]
    return {
        "fell": res.fell,
        "stalled": stalled(res),
        "fall_t": res.fall_time_s,
        "sim_t": res.sim_time_s,
        "x_end": float(res.q_log[-1, 0]) if res.n_ticks else float("nan"),
        # Furthest point reached. x_end alone is misleading: after a fall the robot slides
        # backwards, so a run that got further can report a smaller x_end than one that did not.
        "x_max": float(res.q_log[:, 0].max()) if res.n_ticks else float("nan"),
        "steps": len(res.touchdowns),
        "td_err_cm": float(np.abs(finite).max() * 100) if finite.size else float("nan"),
        "com_z_rmse_cm": float(np.sqrt(np.mean(dz ** 2)) * 100) if dz.size else float("nan"),
        "solve_p50": float(np.percentile(res.mpc_solve_time_ms, 50)) if len(res.mpc_solve_time_ms) else float("nan"),
        "solve_p99": float(np.percentile(res.mpc_solve_time_ms, 99)) if len(res.mpc_solve_time_ms) else float("nan"),
        "update_p50": float(np.percentile(res.mpc_update_time_ms, 50)) if len(res.mpc_update_time_ms) else float("nan"),
        "wall": res.wall_time_s,
    }


# --------------------------------------------------------------------------------
# --------------------------------------------------------------------------------

def _hl(**kw) -> SimResult:
    return run(SimConfig(headless=True, verbose=False, **kw))


def check(duration: float = 6.0, params=None) -> bool:
    flat = str(SCENES_DIR / "scene_flat_easy.xml")
    if not Path(flat).exists():
        T.build("flat", "easy").write()

    results: list[tuple[str, bool, str]] = []

    kw = {} if params is None else {"params": params}
    a = _hl(scene=flat, duration_s=duration, **kw)
    b = _hl(scene=flat, duration_s=duration, **kw)
    for name in ("tau_cmd", "x_vec", "mpc_force_world", "pelvis_z"):
        results.append((
            f"determinism: {name}",
            np.array_equal(getattr(a, name), getattr(b, name)),
            "two identical headless runs must agree bit-for-bit",
        ))

    orig = _hl(scene=DEFAULT_SCENE, duration_s=duration, **kw)
    dmax = float(np.abs(a.tau_cmd - orig.tau_cmd).max())
    results.append((
        "generated flat scene == original scene",
        dmax == 0.0,
        f"max|dtau| = {dmax:.3e} (terrain layer must be neutral)",
    ))

    stand = _hl(scene=DEFAULT_SCENE, duration_s=4.0, **kw)
    dz = abs(float(stand.pelvis_z[-1]) - BASELINE_STAND_Z)
    results.append((
        "standing holds baseline height",
        (not stand.fell) and dz < STAND_TOL,
        f"pelvis_z = {stand.pelvis_z[-1]:.4f} vs baseline {BASELINE_STAND_Z} (dz={dz:.1e})",
    ))

    from bheema.sim import WALK_CMD_SCHEDULE
    walked = _hl(scene=flat, duration_s=16.0, cmd_schedule=WALK_CMD_SCHEDULE, **kw)
    x_end = float(walked.q_log[-1, 0])
    results.append((
        "flat walking matches baseline distance",
        (not walked.fell) and abs(x_end - BASELINE_FLAT_X) < FLAT_X_TOL,
        f"x_end = {x_end:.2f} m vs baseline {BASELINE_FLAT_X} m",
    ))


    ok = all(passed for _, passed, _ in results)
    width = max(len(n) for n, _, _ in results)
    print("=" * (width + 34))
    for name, passed, note in results:
        print(f"  {'PASS' if passed else 'FAIL'}  {name:<{width}}  {note}")
    print("=" * (width + 34))
    print("check:", "PASS" if ok else "FAIL")
    return ok


# --------------------------------------------------------------------------------
# bench: scene sweep
# --------------------------------------------------------------------------------

def bench(difficulty: str, duration: float, scenes=SWEEP, params=None) -> dict[str, dict]:
    hdr = (f"{'scene':<18}{'result':<13}{'steps':>6}{'x_max':>7}{'obst_x':>8}"
           f"{'td_err_cm':>10}{'comz_rmse':>10}{'p50':>7}{'p99':>7}")
    print(hdr)
    print("-" * len(hdr))
    out = {}
    for name in scenes:
        scene = T.build(name, difficulty)
        path = scene.write()
        res = run(SimConfig(scene=str(path), duration_s=duration, headless=True,
                            verbose=False, cmd_schedule=WALK_CMD_SCHEDULE,
                            stop_on_fall=True,
                            **({} if params is None else {"params": params})))
        m = metrics(res)
        m["obstacle_x"] = first_obstacle_x(scene)
        m["reached"] = m["x_max"] > m["obstacle_x"] - 0.25
        out[name] = m
        if m["fell"]:
            verdict = f"fell@{m['fall_t']:.1f}s"
        elif m["stalled"]:
            verdict = "STALLED" if m["reached"] else "stalled-early"
        else:
            verdict = "ok"
        ox = m["obstacle_x"]
        print(f"{name:<18}{verdict:<13}{m['steps']:>6}{m['x_max']:>7.2f}"
              f"{(ox if np.isfinite(ox) else float('nan')):>8.2f}"
              f"{m['td_err_cm']:>10.1f}{m['com_z_rmse_cm']:>10.1f}"
              f"{m['solve_p50']:>7.2f}{m['solve_p99']:>7.2f}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.strip().split("\n")[0])
    ap.add_argument("mode", nargs="?", default="bench", choices=("bench", "check"))
    ap.add_argument("--difficulty", default="easy")
    ap.add_argument("--duration", type=float, default=None,
                    help="sim seconds (default: 6 for check, 16 for bench)")
    ap.add_argument("--scene", action="append", default=None,
                    help="restrict the sweep to these scenes (repeatable)")
    add_arguments(ap)
    args = ap.parse_args()
    params = from_args(args)
    for k, (was, now) in diff(params).items():
        print(f"override {k}: {was} -> {now}")

    if args.mode == "check":
        sys.exit(0 if check(args.duration or 6.0, params) else 1)

    bench(args.difficulty, args.duration or 16.0,
          tuple(args.scene) if args.scene else SWEEP, params)


if __name__ == "__main__":
    main()
