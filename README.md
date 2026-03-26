# BHEEMA: Humanoid Locomotion Framework

**Physics-Consistent Control Stack for the Unitree G1**

BHEEMA is a high-performance humanoid locomotion framework that integrates **Centroidal Model Predictive Control (MPC)** . Designed for the Unitree G1, the framework utilizes a modular architecture to bridge high-level trajectory planning with low-level joint torque physics, all verified within a high-fidelity **MuJoCo** simulation environment.

---

## Technical Architecture

### 1. Centroidal Model Predictive Control (MPC)
The "brain" of the locomotion stack solves an optimization problem over a 16-step lookahead horizon to determine optimal ground reaction forces.
* **State Space**: Operates on a 12-state vector (CoM position, RPY orientation, linear velocity, and angular velocity).
* **Wrench Optimization**: Commands full 6D wrenches ($F_{xyz}, \tau_{xyz}$) per foot, allowing for active balance and Center of Pressure (CoP) control.
* **Real-Time Performance**: Optimized using **CasADi** and **OSQP** with a vectorized Python backend to achieve solve times under **5ms**, fitting comfortably within the 18ms real-time control budget.
* **Torsional Friction & CoP**: Explicitly constrains foot-ground interaction to prevent "helicopter" yaw spins and ankle roll-over.

### 2. Contact Schedule & Trajectory Planning
Locomotion is modeled as a structured, periodic hybrid system.
* **Gait Logic**: Implements a structured bipedal walk with parameterized **Double Support** and **Single Support** phases (Nominal: 1.2Hz, 60% Duty Cycle).
* **Pure Kinematic Stepping**: Utilizes a modified Raibert Heuristic based on **Desired Velocity** drift to force aggressive strides and prevent "shuffling."
* **Swing Trajectory**: Generates 5th-order minimum-jerk curves with smooth Z-height clearance (Nominal: 12cm) to ensure zero-impact lift-off and landing.

### 3. operational Space Leg Controller
Translates high-level MPC wrenches into hardware-level joint torques.
* **Dynamic Transitioning**: Automatically switches between high-stiffness PD tracking in **Swing Phase** and Jacobian-Transpose force mapping in **Stance Phase**.
* **6D OSC**: Uses full spatial Jacobians (Local-World Aligned) to maintain foot orientation (flat-to-ground) and position simultaneously.
* **Gravity Compensation**: Leverages **Pinocchio** to calculate non-linear effects ($C \cdot dq + g$), ensuring the MPC only needs to solve for active balancing forces.

---

## Simulation & Telemetry

The framework includes a comprehensive "Post-Flight" analysis suite to verify gait physics:
* **Offline Previewer**: Visualizes the planned CoM path and footprints before launching the physics engine.
* **Real-Time Visualization**: Syncs MuJoCo sites to show targeted touchdown spots during live simulation.
* **Telemetry Analytics**: Generates high-fidelity plots of 6D contact forces, joint torque saturation, and CoM tracking errors to identify instabilities like "Death Spirals" or "Kinematic Singularities."

---

## Current Capabilities (Unitree G1)
* **Nominal Height**: 0.60m (Bent-knee configuration for maximum torque leverage).
* **Stance Width**: 0.28m (Wide base for lateral stability).
* **Velocity Tracking**: Stable forward locomotion up to 0.4 m/s.
* **Torque Limits**: Configurable up to God-Mode (500 Nm) for debugging or Hardware-Matched (139 Nm) for deployment.

---

## Dependencies
* **MuJoCo**: Physics simulation
* **Pinocchio**: Rigid-body dynamics and analytical Jacobians
* **CasADi / OSQP**: Numerical optimization for MPC
* **Matplotlib**: Real-time and post-flight telemetry

