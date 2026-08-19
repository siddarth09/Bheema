# BHEEMA-Parkour — Plan

Continuation of BHEEMA: from flat-ground convex-MPC walking to Atlas-style perceptive
parkour on the Unitree G1. This document is authoritative for scope and sequencing.

**Branch:** `parkour` (off `main` @ `81cf26e`)
**Decisions taken:** full Atlas-architecture replication, all phases in order · simulation only
(no hardware target) · developed as a branch of the existing bheema repo.

---

## Target architecture

Boston Dynamics' Atlas parkour is not online contact-implicit MPC. It is four cooperating
pieces, and this plan reproduces all four:

1. **Offline behavior library** — trajectory-optimized skills (box jump, gap jump, vault),
   each parameterized by obstacle geometry.
2. **Perception** — surfaces fitted to sensed geometry, matched against skill preconditions.
3. **Online MPC** — adapts the selected template to measured surfaces and live state.
4. **Whole-body control** — arms and torso carry angular momentum; legs are not enough.

Grounded in Boston Dynamics' own write-up of the Atlas parkour system (see
"Source-grounded corrections" below). Current BHEEMA has a partial (3) and nothing else.
The end state:

```
Perception (20 Hz)     depth → planar patches → convex surfaces → foothold scores / preconditions
Behavior FSM           select+parameterize: walk | step-up | box-jump | gap-jump | vault
Reference gen          terrain-relative CoM ref, contact schedule, swing over terrain
                       or: offline skill trajectory (Crocoddyl) resampled to current geometry
SRBD MPC (~40 Hz)      SO(3) variation-based, flight-phase capable, terrain-aware cones
WBC QP (200-500 Hz)    ddq/tau/F over full dynamics; contact, swing, momentum, posture tasks
MuJoCo (2000 Hz)
```

---

## Source-grounded corrections (from the Boston Dynamics Atlas parkour write-up)

Four things this plan had wrong or under-specified. All four *reduce* work.

### 1. A sparse annotated course map is a first-class input — not full reactive detection
This plan implied Phase 6 must recognise arbitrary obstacles and decide what to do. Atlas
does not do that. It is **given a high-level map ahead of time**: an approximate description
containing *obstacle templates and annotated actions* — where to go and which stunt to do
along the way. It is explicitly **not** an exact geometric match. Live perception only fills
in the details: "Atlas knows to look for a box to jump on, and if the box is moved 0.5 m to
the side it will find it there and jump there. If the box is moved too far away the system
won't find it and will come to a stop."

So the perception problem is **localise a known obstacle near where the map says it is**,
not classify an unknown one. Massively easier, and it makes the failure mode graceful
(stop) rather than catastrophic.

`TerrainScene.route` in `bheema/terrain.py` — the ordered list of surfaces each course
intends the robot to step on — is already exactly this annotated map. It was added for
benchmark scoring; it should become the actual planner input, with an `action` annotation
per entry (walk / step-up / jump / vault).

### 2. Rectangular faces, not general convex polygons
Atlas's perception "extracts surfaces using multi-plane segmentation" and tracks **detected
rectangular faces** of obstacles. Since our terrain is boxes by construction, fit
*rectangles* rather than general convex hulls: 4 parameters (centre, extents, yaw in-plane)
instead of a vertex list, far more robust to the depth smearing that inflates hulls at
edges, and it matches ground truth exactly.

### 3. ONE general-purpose controller, not a walk-mode / aerial-mode switch
Phase 4 as originally written added a flight mode alongside the walking MPC. Atlas uses
"a single, general purpose controller" for everything, with the behavior library supplying
the reference. So Phase 4's SO(3) formulation should **replace** the Euler-angle SRBD, not
sit beside it. Fewer code paths, no mode-transition bugs.

Corollary — the MPC horizon must **span behavior boundaries**: "knowing that a jump is
followed by a backflip, the controller can automatically create smooth transitions." The
current horizon is 1.5 gait periods (~1.15 s), which is fine for walking and probably too
short to see across a jump into the next move. Revisit horizon length at Phase 5, not
before.

### 4. Far fewer templates than planned
Phase 5 said "each skill solved over a grid of obstacle parameters, stored, and interpolated
at runtime." Atlas does not need the grid, because MPC covers the gap: "jumping off of a
52 cm platform isn't that different from a 40 cm one, and we can trust MPC to figure out
the details." Build **one template per skill** and let the controller adapt force, posture,
and *behavior timing*. Add a second template only when a gate fails.

### Confirmed as planned
- Depth camera → point cloud → **multi-plane segmentation** (not an elevation map). Atlas
  runs its time-of-flight camera at **15 fps** — use 15 Hz, not the 20 Hz assumed above.
- Offline trajectory optimization for the behavior library; online MPC adapts it.
- Persistent object tracking with pose estimation, surviving objects leaving the field of
  view (their visualisation fades tracked objects green → purple as they go out of view).
  Stronger than the frame-to-frame association assumed in Phase 3.
- IMU + joint positions + force sensors for balance; perception only for obstacles.

### Stated limits, worth inheriting as non-goals
"Attempting to transition to a backflip from a fast forward jogging motion wouldn't work."
There is a real trade-off between controller complexity and library size. Do not treat
arbitrary behavior sequencing as a goal.

---

## Why the current code can't get there

Verified against the code on `main`. These are the load-bearing blockers.

### Flat ground is hardcoded in five places
- `bheema/gait.py:48` — `body_pos = np.array([base_pos[0], base_pos[1], 0.0])`
- `bheema/gait.py:57` — nominal touchdown term forces `z = 0.0`
- `bheema/gait.py:126` — `foot_pos[2] = 0.0` right before the swing trajectory is built
- `bheema/com_traj.py:106` — `pos_traj_world[2, i] = z_pos_des_body`, a **world-frame** constant
- `bheema/gait.py:10` — `HEIGHT_SWING = 0.18`, a fixed clearance regardless of terrain

Consequence: on a 20 cm box the MPC commands the CoM to hold world z = 0.66 while the stance
foot is at z = 0.20. The QP fights the step-up instead of executing it.

Foothold heights *do* already propagate correctly once set — they flow into `Bc`/`Bd` through
`skew(r)` at `com_traj.py:229-241`. They are simply being zeroed upstream.

### The contact schedule is open-loop periodic
`bheema/gait.py:27-33` derives contacts from `mod(t / gait_period)` with no touchdown
detection anywhere in the loop. Over a gap or onto a box, early and late touchdown are the
norm. This is the highest-leverage single fix in Phase 1.

### Stance PD fights every non-nominal posture
`bheema/leg_controller.py:137-141` regresses each stance leg to
`q_des = [-0.3, 0, 0, 0.6, -0.3, 0]` at `Kp = 150`. That term is what currently stabilizes the
flat walk, but a crouch-and-extend for a jump, or a deep step-up, *is* a large deviation from
that pose. It must become a low-priority posture task inside a QP, not a fixed setpoint.

### No WBC, and the upper body is nailed down
`bheema/main.py:280-290` writes constants to waist yaw/roll/pitch and both shoulder pitches on
every sim step. Atlas parkour is substantially arm- and torso-driven angular momentum.

### The MPC structurally cannot represent flight
- `bheema/centroidal_mpc.py:112` — `fz_min = 23.0` is forced on every stance leg.
- `COST_MATRIX_Q` (`centroidal_mpc.py:13-18`) puts 3000 on z and 5000 on roll/pitch against a
  constant reference, so a ballistic phase is actively resisted by the cost even if the
  force bounds allow it (`gd` already integrates free fall correctly).

### The small-angle linearization caps body rotation
`com_traj.py:209-268` linearizes around small roll/pitch with a yaw-averaged `R_z`. Valid for
walking, invalid for a vault or any large-pitch maneuver. Phase 4 replaces this.

### One thing that is genuinely cheap
Terrain-aware friction cones stay **convex**: rotating each foot's pyramid into its contact
surface frame is still linear in the wrench. `_precompute_friction_and_cop_matrix`
(`centroidal_mpc.py:272`) stops being static, but its **sparsity pattern is unchanged** — which
is all `ca.conic` needs fixed at build time. It becomes a per-solve value update, not a rebuild.
Same trick applies to per-foot CoP limits on sloped surfaces.

---

## Phases

Each phase has an explicit gate. Do not start the next phase until the gate passes.

### Phase 0 — Terrain + oracle surface query
Small, and unblocks everything.

**Terrain is planar boxes only — no heightfields.** This mirrors the actual Atlas parkour
course (plywood boxes, platforms, ramps, beams, gaps) and it is a load-bearing simplification,
not just a shortcut: *all* terrain is piecewise planar, so every contact surface has an exact
constant normal and every foothold region is a convex polygon. See "Consequences" below.

- Terrain generator emitting MuJoCo scenes composed of `box` geoms, `include`ing
  `g1_with_hands.xml` — *not* `g1.xml`. `PinG1Model` builds from `g1_with_hands.xml`
  (`g1_config.py:8`) and expects a 50-element `q`, so including the 29-DoF model would
  silently break the Pinocchio/MuJoCo state sync. Scene set: flat (control), stairs,
  gap field, single platform (step-up/step-down), stepping stones, **inclined ramp**
  (a rotated box), balance beam (narrow box).
- **DONE** (`bheema/terrain.py`, `pixi run terrain`). Every terrain geom is in geom
  **group 0**, which doubles as the ray mask. The number is load-bearing and was chosen
  against the compiled model, not guessed: `g1_with_hands.xml` puts all 100 of its geoms
  in group 2 (visual) or 3 (collision) and none in 0 or 1, so group 0 uniquely identifies
  terrain. Group 3 would make a downward ray hit the robot's own legs; group 4+ is
  invisible by default because `mjv_defaultOption` leaves `geomgroup = [1,1,1,0,0,0]`.
- Generated scenes live in `unitree_g1/scenes/` (gitignored, deterministic). They must
  restate `<compiler meshdir="../assets"/>` **after** the include, because MuJoCo resolves
  `meshdir` against the top-level file's directory and the later compiler element wins.
  Scenes also set `offwidth/offheight` for Phase 3's offscreen depth rendering.
- `SurfaceQuery` interface with two backends behind one API (**not yet written**):
  - `oracle` — `mj_ray` against the sim, masked to the terrain geom group. Returns hit height,
    the exact face normal from the hit `geomid`'s frame, and the owning surface id.
  - `perceived` — plane segmentation from depth, added in Phase 3. Same return type.
- Developing the controller against the oracle is the only way to separate controller bugs
  from perception bugs. Keep both backends selectable for the life of the project.

**Ramps are not optional.** If every box is axis-aligned, every contact normal is exactly
`+z`, and all the terrain-aware friction-cone and CoP work in Phase 1 is never exercised —
it becomes untested dead code that silently breaks later. At least one tilted box from day one.

**Gate:** existing flat walk runs unchanged on the new scene loader; oracle surface query
matches known box geometry to < 1 mm in height and < 1e-6 in normal.

### Measured baseline (2026-08-18, pixi env: pinocchio 3.9.0 / mujoco 3.6.0)

Recorded before any Phase 1 change, so regressions are attributable:

- `scene_with_hands.xml`, standing: pelvis z holds 0.763 for 4 s. Note the upper-body
  posture writes (`main.py:279-290`) are **load-bearing** — omit them and it falls by t=2 s.
- `scene_with_hands.xml`, walking: 69.6 s without a fall, through the 1.0 m/s phase.
- `scene_platform_easy.xml`, 0.3 m/s: **15 consecutive clean steps** over 1.65 m, pelvis
  0.771–0.776 (±3 mm). Then at the platform edge the planner requested a foothold at
  z = 0.000 against a true surface of z = 0.150 — a 15.0 cm error — and collapsed over the
  next 3 steps while being pushed *backward* 0.36 m.
- Across all 19 touchdown plans in that run, `pos_touchdown_world[2]` took exactly one
  value: **0.0**. That is the whole Phase 1 problem in one number, and it is an information
  deficit, not a tuning or robustness deficit. Corollary: no gain tuning can fix it.
- MPC solve: ~1.4 ms (pixi) / 9.3 ms first solve cold, N = 48, budget 62.5 ms.

**Phase 1 gate metric, concretely:** the same diagnostic re-run must show the planned-vs-true
touchdown error at the platform edge go to ~0.0 cm, and the run must not fall.

### Phase 1 progress log (measured, 2026-08-18)

Done, all verified by `pixi run check` (8 asserts) staying green — flat ground is bit-identical
throughout (2.67 m over 16 s, standing 0.7634):

- `bheema/surface.py` — `SurfaceQuery` oracle. Heights exact to 0.001 mm on 56 faces; ramp
  normals 8/14/20 deg; floor `mjGEOM_PLANE` special-cased (its `size` is not half-extents).
- `gait.py` — foothold height, swing start height, and swing apex all from the terrain.
  `td_err_cm` is now **0.0 on all 8 scenes** (was equal to each obstacle's height).
- `com_traj.py` — CoM height reference is terrain-relative, from filtered stance-foot support
  height. Deliberately NOT from the terrain under the predicted future CoM: that reads ~0.35 m
  ahead and made the robot extend its legs before the step, losing 0.5 m of progress.
- `leg_controller.py` — stance posture gain scales down when the two footholds differ in
  height, measured from the TERRAIN under each foot (foot height is wrong: the swing foot is
  0.18 m up every step, which softened the gain on flat too).

**Three findings worth keeping:**

1. **Toe clearance, not ankle clearance.** The swing apex must clear terrain sampled under the
   *toe*, which leads the ankle frame by `FOOT_LX_FRONT = 0.12 m`. Measured on the 15 cm box:
   when the toe reached the leading edge the ankle was only s~0.29 into the swing at a
   commanded height of 0.123 m — 2.7 cm below the box top. The toe struck the vertical face.
   Enforce `z_traj(s) >= terrain(toe(s)) + margin` pointwise, and only where terrain rises
   above the interpolated path (applying the margin everywhere is unbounded as b(s) -> 0).
2. **Raising `HEIGHT_SWING` globally does not work.** Swept 0.14-0.40: 0.18 is near optimal,
   and >=0.32 falls on FLAT ground before reaching any obstacle. Without a WBC, a larger swing
   arc's reaction torque destabilises the torso. Per-swing clearance is the fix, not a bigger
   constant.
3. **Holding the touchdown pose does not work either.** `q_des = [-0.3,0,0,0.6,-0.3,0]` is not
   a drift anchor, it is what commands the bent-knee crouch. Snapshotting the pose at impact
   captures an extended reaching leg; the knee never flexes to accept load and it falls on flat
   at t=8.2 s.

### Classical approach: final result on the step-up (2026-08-19)

The step-up onto a 15 cm box was not achieved. Everything below was implemented, verified in
isolation, and measured on `scene_platform_easy` (obstacle edge x=1.60, top 0.15):

| configuration | platform | x_max | flat |
|---|---|---|---|
| legacy | fell@11.2s | 1.52 | ok, 2.67 m |
| legacy + event contact | fell@13.2s | 1.47 | ok, 1.85 m |
| wbc | fell@10.0s | 1.48 | ok, 3.01 m |
| wbc + event contact | fell@8.8s | 1.48 | fell@12.1s |
| wbc + event + force gating | fell@8.6s | 1.46 | fell@11.3s |

`x_max` never exceeds 1.52 against an edge at 1.60, with every ingredient in place:
terrain-aware footholds (td_err = 0.0 on all 8 scenes), pointwise toe clearance,
terrain-relative CoM reference, a whole-body QP with friction/CoP/torque limits inside the
solve, event-based contact timing, contact-force gating, and swept cost weights.

Component-level verification (each confirmed working):
- `surface.py`: heights to 0.001 mm on 56 faces, ramp normals 8.00/14.00/20.00 deg.
- `wbc.py`: double support Fz sums to 337.9 N vs 337.4 N true weight; single support gives
  Fz = 414 N / -0.00 N with swing wrench norm 9.4e-09; 20 deg slope friction ratio 0.383
  against mu = 0.8; solve p50 0.97 ms, p99 1.05 ms.
- `perception.py`: depth cloud 3.47 mm mean error vs oracle, p95 5.95 mm.

Two findings about the existing controller worth keeping:
1. The controller occupies a single narrow operating point. gait_hz 1.0 / 0.8 / 0.6 and duty
   0.55 / 0.50 all fall at t~3 s on FLAT ground. Swing time cannot be bought by slowing down.
2. Stability comes substantially from the Kp=150 joint PD, not from the MPC. Tracking the MPC
   wrench faithfully in the WBC removed that and the robot fell; an explicit joint PD on top
   of the QP torque (as in WBIC) was required to recover standing. The MPC's own first solve
   asks for Fz = 38 N / 436 N across the two feet against a 337 N robot.

Conclusion: the classical stack walks flat robustly (30 s, no fall, both controllers) and does
not traverse a 15 cm step. Parkour behaviours are not reachable from here without the SO(3) MPC
and an offline trajectory library, and the Phase 1 gate is unmet.

### Superseded: earlier reading of the same evidence

I could not make it climb the box within Phase 1's scope. What the measurements establish:

**The swing runs out of TIME, not clearance.** Measured on `platform_easy` after the toe-
clearance fix: the left foot sits at x=1.556, z=0.144 *in STANCE* (pos_des == pos_now, which
only happens in the stance branch) — hovering 14 cm up, 4.5 cm short of its x=1.651 target.
The swing did not complete the reach in 0.269 s and the clock-based schedule declared touchdown
anyway. A higher apex makes this WORSE, because it is more distance to cover in the same time.
That is the real reason the HEIGHT_SWING sweep looked the way it did.

**Swing time cannot be bought by slowing the gait.** Swept gait_hz 1.3 / 1.0 / 0.8 / 0.6 and
duty 0.65 / 0.55 / 0.50: everything except (1.3, 0.65) falls at t~3 s **on flat ground**. The
controller sits at a single narrow hand-tuned operating point and tolerates no change to it.

**Gating the MPC contact set on measured force is correct but does not help yet.** A foot in
mid-air cannot carry the 23 N `fz_min` floor, so gating it is right. But removing the phantom
support forces single-foot balance, which `FOOT_LX=0.12 / FOOT_LY=0.05` CoP limits cannot hold:
stairs went 13.3 s -> 10.8 s. Implemented behind `SimConfig.gate_contact_on_force`, default
**off**, to be enabled once a WBC exists.

**Conclusion: the missing whole-body controller is a prerequisite for the step-up, not an
enhancement.** The specific reasons, all visible in the above:
  * swing tracking is per-leg operational-space impedance with no dynamic consistency with the
    stance leg or torso, so a larger swing motion injects reaction torque nothing compensates;
  * stance posture is a fixed joint-space target rather than a task that can yield;
  * there is no angular-momentum regulation, so the arms and torso cannot help.
This is the same reason Atlas needs whole-body control for parkour. Phase 2 should be brought
forward and the step-up gate retried after it.

**What Phase 1 did achieve:** `td_err_cm = 0.0` on all 8 scenes, both stalls converted to
(much later) falls, survival +3-5 s on every terrain scene, and flat ground bit-identical
throughout. The terrain plumbing is correct and verified; the controller underneath it is not
strong enough to use it yet.

**Original diagnosis, retained for context.** Measured on
`platform_easy`: the foot is scheduled as STANCE while hanging in the air beside the box
(x=1.49, z=0.171, terrain beneath = 0.000), and `_compute_bounds`' `fz_min = 23.0` forces the
MPC to push 23-146 N through it. Pelvis falls 0.648 -> 0.41 with no support anywhere. This is
the open-loop `mod(t/period)` schedule, flagged as the highest-leverage single fix from the
start. Incremental posture/clearance work has reached its ceiling:

| scene | before Phase 1 | after |
|---|---|---|
| flat | ok, 2.67 m | ok, 2.67 m (unchanged, by design) |
| platform | fell@10.2s | fell@11.2s |
| ramp | STALLED | fell@14.1s |
| stairs | fell@8.8s | fell@13.3s |
| atlas_gym | STALLED | fell@13.6s |

Survival time improved substantially and the stalls are gone, but `x_max` has not moved past
the first obstacle on any scene. Event-based touchdown detection + moving the `fz_min` floor
off non-contacting feet is next.

### Phase 1 — Terrain-aware walking
- Remove all five flat-ground hardcodes; foothold z and surface normal from `SurfaceQuery`.
- Contact-surface-frame friction pyramids and CoP limits (per-solve value update).
- CoM z reference becomes terrain-relative: nominal height above the support-foothold plane.
- Swing clearance from the box profile between liftoff and touchdown, not a constant. With
  planar boxes this is a max over a short ray-cast fan along the swing arc, not a grid scan.
- **Foothold region constraints.** Because every surface is a convex polygon, "the foot lands
  fully on one surface" is a set of *linear* inequalities on the touchdown point. This makes
  Acosta & Posa (already cited in `README.md`) directly applicable — footstep position can
  become a decision variable constrained to a chosen surface rather than a Raibert output that
  gets projected and hoped over.
- **Event-based contact:** touchdown detection from foot force, contact-schedule adaptation for
  early/late touchdown.

**Gate:** 15 cm stairs up and down, a 10° ramp up and down, and a 4-stone stepping-stone
field, at 0.3 m/s, 60 s without a fall.

### Phase 2 — WBC replaces the per-leg controller
One QP at 200-500 Hz over `(ddq, tau, F_contact)`:
- Constraints: full floating-base dynamics, contact consistency (no-slip on stance),
  friction cones, torque limits, joint limits.
- Tasks, by priority: contact wrench tracking (from MPC) → swing foot pose → CoM / centroidal
  angular momentum → torso orientation → arm and waist posture.
- Pinocchio already provides `M`, `C`, `g`, frame Jacobians and `Jdot*dq`
  (`g1_config.py:242-260`). The QP is small; OSQP is fine.
- This is where `leg_controller.py:137`'s fixed `q_des` dies and the arms come alive.

**Gate:** Phase 1 terrain re-passed with the WBC, visible arm swing, and no fixed-posture
term anywhere in the control path.

### Phase 3 — Perception
- Head-mounted depth camera in the MJCF. Note `unitree_g1/g1.xml` currently has **no camera** —
  only IMU sites at lines 72 and 184.
- MuJoCo depth render at ~20 Hz → point cloud → **planar segmentation**, not a 2.5D elevation
  grid. Since ground truth is piecewise planar, extract planar patches (RANSAC or region
  growing on normals), fit a convex polygon to each, and track surfaces across frames by
  plane parameters. This is what Atlas actually does, and it is the natural consumer of the
  Phase 1 foothold-region constraints — same `SurfaceQuery` return type as the oracle.
- Foothold scoring over the extracted polygons: distance to polygon edge, plane inclination,
  patch support/confidence, and deviation from the Raibert nominal, maximized within leg
  reachability. Local roughness drops out — there is none.
- Watch for the failure modes that a grid map hides and a polygon map exposes: patches
  over-merged across a stair nose, polygons inflated past the true edge by depth smearing, and
  a missing patch producing *no* candidate rather than a bad one.
- Viewer overlay for the elevation map and scored candidates.
- Sim-only, so ground-truth pose from `g1_mujoco.py:update_pin_with_mujoco` stays. A
  floating-base EKF is explicitly **out of scope** — but the map must be built through a pose
  *interface*, so drift can be injected later to test robustness.

**Gate:** stepping stones and a gap field that are impossible without perception, with the
oracle backend disabled.

### Phase 4 — SO(3) MPC with flight phases
Removes the small-angle cap and makes ballistic motion representable.
- Replace Euler-angle SRBD with a **variation-based SO(3) formulation** (Ding, Zhou & Park,
  *Representation-Free MPC for Dynamic Motions in Quadrupeds*). Key property: handles large
  body rotations with no small-angle assumption while **still solving one convex QP per
  control step** — so OSQP, warm starting, and the ~1.4 ms solve budget all survive.
- Flight-phase support: drop the `fz_min` floor when both feet are swing, generate a ballistic
  CoM/orientation reference for the aerial segment, and re-anchor the reference at predicted
  touchdown.

**Gate:** in-place vertical hop and a 30 cm broad jump with a real flight phase, MPC solve
still within the control period.

### Phase 5 — Offline skill library
- **Crocoddyl** for the offline trajectory optimization (box jump, gap jump, step-vault,
  eventually 180° spin). Pinocchio-native DDP built for multi-contact with flight phases —
  worth the dependency versus hand-rolling in CasADi.
- Each skill solved over a grid of obstacle parameters (height, depth, approach speed), stored,
  and interpolated at runtime.
- Online: MPC tracks the template, adapting to measured surfaces and live state.

**Gate:** 30 cm box jump-up and jump-down, and a 40 cm gap jump, from an oracle-known obstacle.

### Phase 6 — Full pipeline
- Perception → surface segmentation → skill precondition matching → parameterization.
- Behavior FSM with approach/align phases, abort-and-recover transitions, fallback to walking.
- End-to-end run over a mixed course.

**Gate:** unrehearsed traverse of a mixed course (stairs → rough → gap → box) with no
oracle and no per-obstacle hand tuning.

---

## Cross-cutting

**Benchmark suite.** Build it in Phase 0 and run it every phase: fixed scene set × fixed
command schedule, scripted headless, reporting success/fall, distance, CoM tracking error,
foot-placement error vs. planned, MPC solve time distribution, WBC QP failure count. Every
gate above is a query against this. Regressions on earlier terrain are the main risk in
Phases 4-6.

**ROS 2 / mujoco_ros2_control — evaluated 2026-08-18, deferred to Phase 3.**
The sensor plugins are genuinely good and better than earlier notes claimed:
`mujoco_3d_lidar_plugin.cpp` publishes real `PointCloud2` (configurable azimuth/elevation
ranges, 2D resolution), `camera_plugin.cpp` publishes `Image` + `CameraInfo`, and
ros2_control's standard broadcasters cover IMU and force-torque. (Note
`external_wrench_plugin` *applies* disturbance wrenches; it does not read foot FT — that
comes from MuJoCo native `force`/`torque` sensors via `force_torque_sensor_broadcaster`.)
rviz and rosbag are real wins: recording a perception run once and replaying it offline is
the right way to develop plane segmentation.

Three reasons it is not the move *yet*:

1. **Interpreter conflict, reproduced.** `/opt/ros/jazzy` ships Pinocchio built against
   NumPy 1.x; bheema needs NumPy 2.x. `import pinocchio` with the ROS `PYTHONPATH` set
   segfaults (core dumped). `pinocchio` and `rclpy` cannot currently share one interpreter
   on this machine. The fix is the robostack env already scoped in HANUMAN's execution
   order — a discrete prerequisite task, not a side effect.
2. **ros2_control controllers are C++.** The MPC/WBC is Python + Pinocchio + CasADi/OSQP,
   so it cannot be a `controller_interface::ControllerInterface` without a rewrite. The
   alternative is a Python node running the 200-500 Hz WBC over DDS.
3. **Only one process can step the physics.** So "in-process control loop + ROS 2
   perception" is not available; adopting mujoco_ros2_control means the sim moves into the
   ROS 2 node and the control loop follows it. That also costs the run-to-run determinism
   the phase-gate benchmark suite depends on.

**Free insurance, take it now:** the `perceived` SurfaceQuery backend must take
`(point_cloud, sensor_pose)` as input — never a `MjModel`/`MjData` handle. Then the source
is swappable between an in-process depth render and a `PointCloud2` subscription with zero
change to the segmentation code, and this decision stays reversible at Phase 3.

Until then: single process, MuJoCo viewer overlays for visualization. `launch/` and
`package.xml` are dead weight in the current design.

**Housekeeping.** `bheema/__pycache__/*.pyc` is tracked in git and shows up dirty on every run;
add it to `.gitignore` early.

---

## Risks and kill criteria

| Risk | Mitigation / kill criterion |
|---|---|
| Event-based contact destabilizes the flat walk | Benchmark suite catches it in Phase 1; keep periodic schedule behind a flag |
| WBC QP infeasible during fast transitions | Soften low-priority tasks to costs; log failure count as a gate metric |
| SO(3) MPC breaks solve budget | Measure before committing Phase 5. Fallback: SO(3) only in aerial mode, Euler for walking |
| Crocoddyl skills don't transfer to MPC tracking | Track centroidal reference only, not whole-body; if it still fails, reduce skill set to jumps and drop vaults |
| Plane segmentation over-merges stair noses / inflates polygons past true edges | Oracle stays available; score edge distance conservatively (shrink polygons by a margin) before blaming the controller |
| All-axis-aligned boxes leave terrain-aware cones untested | Ramp in the scene set from Phase 0 |
| Full 180° flips | Accepted as likely out of reach for a solo sim project. Not a gate on any phase |

---

## References already in `research/`
`MIT CHEETAH.pdf` · `bipedal mpc.pdf` · `WBC.pdf` · `wbc_marcohutter.pdf` ·
`Tedrake_Footstep planning.pdf` · `Tedrake_EMB Optimization-Based Locomotion.pdf` ·
`aron ames bipedal gait design.pdf` · `RoMoCo.pdf`

To add: Ding/Zhou/Park representation-free MPC · Jenelten et al. perceptive locomotion and
foothold scoring · Crocoddyl paper · Atlas parkour technical talks.
