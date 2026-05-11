# BHEEMA — Bipedal Humanoid Equilibrium via Efficient Model-predictive Architecture


https://github.com/user-attachments/assets/3151054e-4f4d-4371-8789-fa6e95afd746

---

## RL Locomotion Policy for Unitree G1

A learned locomotion controller for the Unitree G1 humanoid (34.4 kg, 29 DOF), trained with Proximal Policy Optimization in massively parallel simulation. The policy runs at 50 Hz and outputs joint position targets for all 29 joints — legs, waist, and arms — tracked by actuator-level PD controllers at 1 kHz.

Trained on flat terrain with domain randomization (friction, mass, external pushes, sensor noise) for sim-to-real robustness. The robot walks, turns, strafes, and recovers from perturbations using a single 4-layer MLP with no explicit gait planner, trajectory optimizer, or contact scheduler.

---

### Pretrained Checkpoint

Download from [HuggingFace](http://huggingface.co/Siddarth09/bheema_locomotion):


## Architecture

```
Keyboard Teleop          →  velocity command [vx, vy, yaw_rate]
RL Policy (50 Hz)        →  29 joint position targets
PD Controller (1 kHz)    →  joint torques  →  MuJoCo physics (1 kHz)
```

The policy replaces the classical MPC + whole-body controller pipeline entirely. No Pinocchio, no QP solver, no Jacobians at runtime — just a forward pass through a small neural network.

---

## Policy Details

| | |
|---|---|
| **Algorithm** | PPO (Proximal Policy Optimization) |
| **Framework** | mjlab + RSL-RL + MuJoCo Warp |
| **Training envs** | 2048 parallel (GPU-accelerated) |
| **Training time** | ~7 hours on RTX 5060 (8 GB) |
| **Iterations** | 60,000 |
| **Network** | MLP: 99 → 512 → 256 → 128 → 29, ELU activations |
| **Policy rate** | 50 Hz |
| **Simulation rate** | 1 kHz |

---

## Observation Space (99 dimensions)

The policy observes only quantities available on real hardware — no privileged simulator state.

| Component | Dim | Source |
|---|---|---|
| Base linear velocity (body frame) | 3 | IMU + state estimation |
| Base angular velocity (body frame) | 3 | IMU gyroscope |
| Projected gravity | 3 | IMU orientation → R_body^T × [0,0,−1] |
| Joint positions (relative to default) | 29 | Joint encoders |
| Joint velocities | 29 | Joint encoders |
| Previous action | 29 | Internal buffer |
| Velocity command | 3 | Operator input [vx, vy, yaw_rate] |

Observation noise is injected during training: ±0.2 rad/s on angular velocity, ±0.05 on gravity projection, ±0.01 rad on joint positions, ±1.5 rad/s on joint velocities.

---

## Action Space (29 dimensions)

The policy outputs joint position offsets added to a default standing pose, scaled per joint:

```
q_target = q_default + action_scale × policy_output
τ = Kp × (q_target − q_measured) − Kd × dq_measured
```

| Joint Group | Joints | Action Scale (rad) | Kp (Nm/rad) | Kd (Nm·s/rad) |
|---|---|---|---|---|
| Hip pitch/yaw | 4 | 0.55 | 40.2 | 2.6 |
| Hip roll, knee | 4 | 0.35 | 99.1 | 6.3 |
| Ankle | 4 | 0.44 | 28.5 | 1.8 |
| Waist | 3 | 0.44–0.55 | 28.5–40.2 | 1.8–2.6 |
| Shoulder, elbow | 8 | 0.44 | 14.3 | 0.9 |
| Wrist | 6 | 0.07–0.44 | 14.3–16.8 | 0.9–1.1 |

Leg actuators use direct torque control (motor type). Upper body actuators use MuJoCo's built-in position servos.

---

## Reward Function

14 terms shape the walking behavior. Positive weights encourage desired behavior, negative weights penalize undesired behavior.

| Term | Weight | Purpose |
|---|---|---|
| track_linear_velocity | +2.0 | Track commanded XY velocity (Gaussian kernel) |
| track_angular_velocity | +2.0 | Track commanded yaw rate |
| air_time | +0.5 | Encourage proper foot swing (prevent shuffling) |
| foot_clearance | +0.5 | Lift feet during swing phase |
| upright | −1.0 | Keep torso vertical |
| pose | −0.5 | Stay near default joint configuration |
| body_ang_vel | −0.05 | Minimize roll/pitch angular velocity |
| angular_momentum | −0.02 | Keep angular momentum low |
| dof_pos_limits | −1.0 | Avoid joint limit violations |
| action_rate_l2 | −0.1 | Smooth actions (minimize ‖a_t − a_{t−1}‖²) |
| foot_swing_height | −0.25 | Don't kick too high |
| foot_slip | −0.1 | Minimize foot sliding during stance |
| soft_landing | −1e-5 | Gentle foot placement |
| self_collisions | −1.0 | Prevent limb-to-limb contact |

Velocity tracking uses a Gaussian kernel `exp(−‖error‖²/σ²)` instead of raw error to create a sharp gradient toward precise tracking.

---

## Domain Randomization

Applied during training for sim-to-real robustness:

| Parameter | Range |
|---|---|
| Ground friction | 0.5× – 1.5× nominal |
| Robot mass | ±10% |
| External pushes | ±0.5 m/s, every 5–15s |
| Initial joint offset | ±0.1 rad |
| Initial velocity | ±0.5 rad/s |
| Sensor noise | Per-channel (see observation table) |

---

## Training Results

| Metric | Value |
|---|---|
| Mean reward | 55.28 |
| Mean episode length | 1000 / 1000 (never falls) |
| Velocity tracking error (XY) | 3.62 m/s (curriculum pushed to 3 m/s commands) |
| Yaw tracking error | 0.76 rad/s |
| Action smoothness (std) | 0.38 |
| Fall rate | 0.0 |

Curriculum expanded velocity commands to vx ∈ [−2.0, 3.0] m/s, vy ∈ [−1.0, 1.0] m/s, ω_z ∈ [−0.7, 0.7] rad/s during training.

---

## Running

### Prerequisites

```bash
pip install mujoco torch pynput numpy
```

### Teleoperation

```bash
cd bheema_rl
python main_rl_teleop.py
```

| Key | Action |
|---|---|
| ↑ / ↓ | Forward / Backward |
| ← / → | Strafe Left / Right |
| Z / X | Turn Left / Right |
| SPACE | Emergency Stop |
| ESC | Quit |

### Training (requires mjlab)

```bash
git clone https://github.com/mujocolab/mjlab && cd mjlab
pip install -e ".[all]"
python -m mjlab.scripts.train Mjlab-Velocity-Flat-Unitree-G1 --env.scene.num-envs 2048
```

### Evaluation

```bash
python -m mjlab.scripts.play Mjlab-Velocity-Flat-Unitree-G1 \
    --checkpoint_file logs/rsl_rl/g1_velocity/<run>/model_59999.pt
```

---

## File Structure

```
bheema_rl/
├── rl_bridge.py         Policy inference + joint mapping + PD control
├── teleop.py            Arrow-key keyboard controller
└── main_rl_teleop.py    Simulation loop with live viewer
```

---

## Classical MPC Pipeline (System 2)

Bheema also includes a full model-predictive control pipeline based on the MIT Cheetah 3 convex MPC paper, which serves as the planning layer in the hybrid architecture:

```
centroidal_mpc.py        QP solver for optimal contact wrenches (OSQP)
com_traj.py              Reference trajectory with lateral sway
gait.py                  Phase-based gait scheduler + Raibert footstep planner
leg_controller.py        Jacobian transpose (stance) + impedance control (swing)
g1_config.py             Pinocchio model, FK, Jacobians, dynamics
g1_mujoco.py             MuJoCo interface + Pinocchio/MuJoCo state sync
```

---

## References

- Schulman et al., [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347), 2017
- Di Carlo et al., [Dynamic Locomotion in the MIT Cheetah 3 Through Convex MPC](https://dspace.mit.edu/bitstream/handle/1721.1/138000/convex_mpc_2fix.pdf), IROS 2018
- Rudin et al., [Learning to Walk in Minutes Using Massively Parallel Deep RL](https://arxiv.org/abs/2109.11978), CoRL 2022
- [mjlab](https://github.com/mujocolab/mjlab) — MuJoCo Warp + RSL-RL training framework
- [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) — Unitree G1 MJCF model