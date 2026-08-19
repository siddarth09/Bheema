"""
Entry point for the MPC walk: build a default config, run it, plot the telemetry.

The simulation loop itself lives in `bheema/sim.py`. Everything that used to be
module-level state here (log arrays sized at import, CMD_SCHEDULE, the viewer opened at
module scope) is now a SimConfig field or a SimResult field, so a run can be scripted,
repeated, or run headless.

    pixi run walk                       # viewer + plots, as before
    python -m bheema.main --headless    # no viewer, no plots, prints a summary
"""

import argparse
import os

os.environ.setdefault("MPLBACKEND", "TkAgg")

from bheema.plotter import plot_mpc_result, plot_solve_time, plot_swing_foot_traj
from bheema.params import add_arguments, describe, diff, from_args
from bheema.sim import DEFAULT_CMD_SCHEDULE, DEFAULT_SCENE, WALK_CMD_SCHEDULE, SimConfig, run


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.strip().split("\n")[0])
    ap.add_argument("--scene", default=DEFAULT_SCENE, help="path to a MuJoCo scene XML")
    ap.add_argument("--duration", type=float, default=120.0, help="sim seconds")
    ap.add_argument("--headless", action="store_true", help="no viewer, no pacing, no plots")
    ap.add_argument("--no-plots", action="store_true", help="run with viewer but skip plots")
    ap.add_argument("--walk", action="store_true",
                    help="terrain schedule: stand 2s then a steady 0.3 m/s "
                         "(default schedule stands for 15s then runs at 1.0 m/s)")
    ap.add_argument("--print-params", action="store_true",
                    help="dump every parameter and exit")
    add_arguments(ap)
    args = ap.parse_args()

    params = from_args(args)
    if args.print_params:
        print(describe(params))
        return
    changed = diff(params)
    if changed:
        print("parameter overrides:")
        for k, (was, now) in changed.items():
            print(f"  {k}: {was} -> {now}")

    cfg = SimConfig(
        scene=args.scene,
        duration_s=args.duration,
        headless=args.headless,
        params=params,
        cmd_schedule=WALK_CMD_SCHEDULE if args.walk else DEFAULT_CMD_SCHEDULE,
    )
    res = run(cfg)

    if args.headless or args.no_plots:
        print(f"ticks={res.n_ticks} sim_t={res.sim_time_s:.2f}s wall={res.wall_time_s:.2f}s "
              f"fell={res.fell} pelvis_z_final={res.pelvis_z[-1] if res.n_ticks else float('nan'):.3f} "
              f"mpc_solve_ms p50={_p(res.mpc_solve_time_ms, 50):.2f} "
              f"p99={_p(res.mpc_solve_time_ms, 99):.2f}")
        return

    plot_swing_foot_traj(res.t, res.foot.pos_now, res.foot.pos_des,
                         res.foot.vel_now, res.foot.vel_des, block=False)
    plot_mpc_result(res.t, res.mpc_force_world, res.tau_cmd, res.x_vec, block=False)
    plot_solve_time(list(res.mpc_solve_time_ms), list(res.mpc_update_time_ms),
                    cfg.mpc_dt, cfg.mpc_hz, block=True)


def _p(a, q: float) -> float:
    import numpy as np
    return float(np.percentile(a, q)) if len(a) else float("nan")


if __name__ == "__main__":
    main()
