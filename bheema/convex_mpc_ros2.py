#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

import numpy as np
import pinocchio as pin

from sensor_msgs.msg import JointState, Imu
from std_msgs.msg import Float64MultiArray
from tf2_ros import Buffer, TransformListener

# --- BHEEMA imports ---
from bheema.bheema.state import G1State
from bheema.bheema.centroidal_dynamics import CentroidalDynamics, CentroidalDiscreteModel
from bheema.bheema.centroidal_mpc import CentroidalMPC
from bheema.bheema.whole_body_controller import WholeBodyController
from bheema.bheema.contact_scheduler import ContactSchedule, WalkingGaitParams


class BheemaControlNode(Node):
    """
    Final integrated BHEEMA controller node (v1).

    - MPC: centroidal, slow (50Hz)
    - WBC: kinematic hierarchical, fast (200Hz)
    - Output: joint position targets
    """

    def __init__(self):
        super().__init__('bheema_controller')

        # --------------------------------------------------
        # Timing
        # --------------------------------------------------
        self.wbc_rate = 200.0
        self.dt = 1.0 / self.wbc_rate

        self.mpc_dt = 0.02
        self.mpc_horizon = 10
        self.mpc_decimation = int(self.mpc_dt / self.dt)
        self.mpc_tick = 0

        # --------------------------------------------------
        # Robot model & state
        # --------------------------------------------------
        self.state = G1State()
        self.model = self.state.model

        self.wbc = WholeBodyController(self.state)

        # --------------------------------------------------
        # Centroidal MPC
        # --------------------------------------------------
        mass = self.state.get_total_mass()
        inertia = self.state.get_centroidal_inertia_com()

        self.dynamics = CentroidalDynamics(mass, inertia)
        self.discrete_model = CentroidalDiscreteModel(self.mpc_dt)
        self.mpc = CentroidalMPC(self.mpc_horizon, self.mpc_dt)

        # --------------------------------------------------
        # Gait / contact scheduler
        # --------------------------------------------------
        self.gait_params = WalkingGaitParams(
            step_time=0.6,
            ds_fraction=0.2, # Reduced for clearer stepping
            start_with_left_stance=True
        )
        self.scheduler = ContactSchedule(self.gait_params)

        # --------------------------------------------------
        # Joint mapping (MUST match controller.yaml)
        # --------------------------------------------------
        self.leg_joints = [
            'left_hip_pitch_joint', 'left_hip_roll_joint', 'left_hip_yaw_joint',
            'left_knee_joint', 'left_ankle_pitch_joint', 'left_ankle_roll_joint',
            'right_hip_pitch_joint', 'right_hip_roll_joint', 'right_hip_yaw_joint',
            'right_knee_joint', 'right_ankle_pitch_joint', 'right_ankle_roll_joint'
        ]
        self.waist_joints = [
            'waist_yaw_joint', 'waist_roll_joint', 'waist_pitch_joint'
        ]

        self.all_joints = self.leg_joints + self.waist_joints
        self.joint_map = {name: i for i, name in enumerate(self.all_joints)}

        self.qj = np.zeros(len(self.all_joints))
        self.vj = np.zeros(len(self.all_joints))

        # --------------------------------------------------
        # Floating base state
        # --------------------------------------------------
        self.base_pos = np.array([0.0, 0.0, 0.73])
        self.prev_base_pos = self.base_pos.copy()

        self.base_quat = np.array([0.0, 0.0, 0.0, 1.0])  # x y z w
        self.base_lin_vel = np.zeros(3)
        self.base_ang_vel = np.zeros(3)

        # --------------------------------------------------
        # ROS interfaces
        # --------------------------------------------------
        self.leg_pub = self.create_publisher(
            Float64MultiArray,
            '/g1_leg_controller/commands',
            10
        )
        self.waist_pub = self.create_publisher(
            Float64MultiArray,
            '/g1_waist_controller/commands',
            10
        )

        self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_cb,
            10
        )

        self.create_subscription(
            Imu,
            '/imu_sensor_broadcaster/imu',
            self.imu_cb,
            10
        )

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # --------------------------------------------------
        # Control Variables
        # --------------------------------------------------
        self.f_mpc = np.zeros(6)
        self.sim_time = 0.0
        
        # Nominal Foot Positions (Cached for stepping in place)
        self.pl_nom = np.array([0.0, 0.15, 0.0]) # Approx G1 width
        self.pr_nom = np.array([0.0, -0.15, 0.0])

        self.timer = self.create_timer(self.dt, self.control_loop)
        self.get_logger().info("BHEEMA Control Node (v1) started")

    # ==================================================
    # Callbacks
    # ==================================================
    def joint_state_cb(self, msg):
        for i, name in enumerate(msg.name):
            if name in self.joint_map:
                idx = self.joint_map[name]
                self.qj[idx] = msg.position[i]
                self.vj[idx] = msg.velocity[i]

    def imu_cb(self, msg):
        self.base_quat = np.array([
            msg.orientation.x,
            msg.orientation.y,
            msg.orientation.z,
            msg.orientation.w
        ])

        # IMU ang vel is body frame → rotate to world
        w_body = np.array([
            msg.angular_velocity.x,
            msg.angular_velocity.y,
            msg.angular_velocity.z
        ])
        Rwb = pin.Quaternion(self.base_quat).toRotationMatrix()
        self.base_ang_vel = Rwb @ w_body

    def update_base_position(self):
        """Estimate base position via TF (Sim) or Odometry (Real)"""
        try:
            if self.tf_buffer.can_transform('world', 'pelvis', rclpy.time.Time()):
                t = self.tf_buffer.lookup_transform(
                    'world', 'pelvis', rclpy.time.Time()
                )
                self.base_pos = np.array([
                    t.transform.translation.x,
                    t.transform.translation.y,
                    t.transform.translation.z
                ])
        except Exception:
            pass

    # ==================================================
    # Main control loop
    # ==================================================
    def control_loop(self):
        # --------------------------------------------------
        # 1. State estimation
        # --------------------------------------------------
        self.update_base_position()

        # Simple finite difference for lin vel (noisy but works for V1)
        self.base_lin_vel = (self.base_pos - self.prev_base_pos) / self.dt
        self.prev_base_pos = self.base_pos.copy()

        q = np.concatenate([self.base_pos, self.base_quat, self.qj])
        v = np.concatenate([self.base_lin_vel, self.base_ang_vel, self.vj])

        self.state.update(q, v)
        x_cur = self.state.get_bheema_x()

        # --------------------------------------------------
        # 2. MPC (slow - 50Hz)
        # --------------------------------------------------
        if self.mpc_tick % self.mpc_decimation == 0:
            contact_plan = self.scheduler.build(
                self.sim_time,
                self.mpc_dt,
                self.mpc_horizon
            )

            # Reference: Station Keeping (Stand in place)
            x_ref = np.zeros((12, self.mpc_horizon))
            x_ref[0:2, :] = x_cur[0:2].reshape(2, 1) # Keep current XY
            x_ref[2, :] = 0.73  # Target Height

            rL, rR = self.state.get_foot_levers_world()
            Ac, Bc, gc = self.dynamics.continuous_matrices(rL, rR)
            Ad, Bd, gd = self.discrete_model.discretize(Ac, Bc, gc)

            try:
                # Assume LTI over horizon for speed
                u = self.mpc.build_and_solve(
                    [Ad] * self.mpc_horizon,
                    [Bd] * self.mpc_horizon,
                    [gd] * self.mpc_horizon,
                    x_cur,
                    x_ref,
                    contact_plan
                )
                self.f_mpc = u[:6]
            except Exception as e:
                self.get_logger().warn(f"MPC failed: {e}")

        self.mpc_tick += 1

        # --------------------------------------------------
        # 3. WBC (fast - 200Hz)
        # --------------------------------------------------
        cL, cR = self.scheduler._contact_at_time(self.sim_time)
        contact_table = np.array([[cL], [cR]])

        # -- Nominal Posture (Prevent Singularity) --
        q_nom = pin.neutral(self.model)
        for i, name in enumerate(self.model.names):
            if "knee" in name:
                idx = self.model.joints[i].idx_q
                if idx >= 0: q_nom[idx] = 0.3 # Bend knees
        q_nom[2] = 0.73

        # -- Swing Trajectory Generation (FILLED TODO) --
        swing_trajs = {}
        
        # Current Foot Positions
        pL_curr, pR_curr = self.state.get_foot_positions_world()
        
        # Simple Sine Wave Generator 
        swing_height = 0.08 # 8cm lift
        step_T = self.gait_params.step_time
        
        # Calculate Phase (0 to 1) within the step
        # Note: This is simplified. Ideally, get exact phase from scheduler.
        t_cycle = self.sim_time % step_T
        phase = t_cycle / step_T
        
        # If Left is swinging (contact == 0)
        if cL == 0:
            # Create a simple vertical lift sine wave
            # For stepping in place, x/y target is current x/y
            # We assume a bell curve or sine for Z
            z_lift = swing_height * np.sin(np.pi * phase) # Simple 0 -> 1 -> 0 arc
            
            # Target is current X,Y but modified Z
            # Ideally: Interpolate from Lift-off Pos to Touch-down Pos
            swing_trajs['left'] = {
                "pos": np.array([pL_curr[0], pL_curr[1], z_lift])
            }

        # If Right is swinging
        if cR == 0:
            z_lift = swing_height * np.sin(np.pi * phase)
            swing_trajs['right'] = {
                "pos": np.array([pR_curr[0], pR_curr[1], z_lift])
            }

        # Solve WBC
        q_des = self.wbc.compute_joint_targets(
            state=self.state,
            f_mpc=self.f_mpc,
            contact_table=contact_table,
            swing_trajs=swing_trajs,
            q_nominal=q_nom
        )

        # --------------------------------------------------
        # 4. Publish joint commands
        # --------------------------------------------------
        qj_des = q_des[7:]  # strip floating base

        leg_msg = Float64MultiArray()
        waist_msg = Float64MultiArray()

        leg_msg.data = qj_des[0:12].tolist()
        waist_msg.data = qj_des[12:15].tolist()

        self.leg_pub.publish(leg_msg)
        self.waist_pub.publish(waist_msg)

        self.sim_time += self.dt


def main(args=None):
    rclpy.init(args=args)
    node = BheemaControlNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()