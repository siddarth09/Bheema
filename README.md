# BHEEMA

**Humanoid Control + RL Locomotion Policy**

BHEEMA is a humanoid locomotion framework built around a **physics-consistent state representation**, an explicit **walking contact schedule**, and a modular control stack designed to support **model-based control (MPC / WBC)** as well as **learning-based policies**.

The project is developed and tested using **MuJoCo + ros2_control**, with Pinocchio used for kinematics and dynamics.

---

## State Representation (Centroidal State)

Humanoid locomotion control does not operate directly on the full joint state for planning and balance. Instead, BHEEMA uses a **centroidal state representation**, which captures the global motion and balance-relevant quantities of the robot while abstracting away joint-level complexity.

### What the state represents

At every control step, BHEEMA extracts a **centroidal state** consisting of:

* The **center of mass (COM) position** in the world frame
* The **base orientation** (roll, pitch, yaw)
* The **COM linear velocity** in the world frame
* The **base angular velocity** expressed in the world frame

This state captures:

* where the robot is,
* how it is oriented,
* how it is translating,
* how it is rotating.

These quantities are the ones that directly determine balance, momentum, and feasibility of contact forces.

### Why this state is used

This representation is standard in modern humanoid and legged-robot control because:

* It is **independent of the number of joints**
* It is **directly compatible with centroidal momentum dynamics**
* It allows control algorithms to reason about balance and motion **without solving full rigid-body dynamics online**

Joint positions and velocities are still used at lower levels of the controller, but they are intentionally **not part of the high-level planning state**.

---

## Contact Schedule (Walking Gait Logic)

Humanoid locomotion is inherently **hybrid**: the robot switches between different contact configurations as it walks.
BHEEMA models this explicitly using a **contact schedule**.

### What a contact schedule is

A contact schedule is a **time-indexed description of which feet are in contact with the ground**.

For a biped, this means:

* left foot contact: on or off
* right foot contact: on or off

At each discrete time step, the contact schedule specifies whether each foot is:

* in **stance** (able to apply forces), or
* in **swing** (must apply zero force).

### Why the contact schedule is explicit

The contact schedule determines:

* which ground reaction forces are allowed,
* which friction constraints are active,
* which torque channels exist in the dynamics.

Without an explicit contact schedule, force-based control problems become ill-posed, because the controller would be allowed to apply forces at feet that are not actually in contact.

### Walking structure

BHEEMA models walking using a **structured, phase-based gait**:

* **Double support**: both feet on the ground
* **Single support**: one foot on the ground, the other swinging
* Alternation of the stance foot between steps

This structure:

* avoids flight phases,
* ensures smooth load transfer,
* reflects how real humanoids walk.

The contact schedule is **parameterized**, not hard-coded.
Step duration, double-support fraction, and starting stance foot can be adjusted without changing the controller logic.

### Separation of responsibilities

The contact schedule:

* does **not** plan foot trajectories,
* does **not** compute forces,
* does **not** depend on joint kinematics.

It is purely a **logical signal** that gates constraints and force variables in higher-level controllers (e.g., centroidal MPC).
