import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState
from nav_msgs.msg import Odometry
import numpy as np
import pinocchio as pin

# Import Bheema Library
from bheema.state import G1State
from bheema.mpc import CentroidalMPC
from bheema.wbc import WholeBodyController
from bheema.gait_scheduler import BipedGaitScheduler
from bheema.hlip_planner import HLIPFootPlanner
from bheema.com_ref import ComReference
from bheema.centroidal_model import CentroidalDynamics, CentroidalDiscrete
from bheema.visualizer import Bheemaplotter

class BheemaControlNode(Node):
    def __init__(self):
        super().__init__('bheema_controller')

        # -------------------------------------------------------------------
        # 1. Configuration & Parameters
        # -------------------------------------------------------------------
        self.control_freq = 200.0  # Hz (WBC Loop)
        self.mpc_freq = 50.0       # Hz (MPC Loop)
        self.dt_wbc = 1.0 / self.control_freq
        self.dt_mpc = 1.0 / self.mpc_freq
        
        self.mpc_decimation = int(self.control_freq / self.mpc_freq)
        self.tick_count = 0

        # Debugging counters
        self.log_counter = 0

        # --- CONTROLLER DEFINITIONS ---
        self.leg_joint_names = [
            'left_hip_pitch_joint', 'left_hip_roll_joint', 'left_hip_yaw_joint',
            'left_knee_joint', 'left_ankle_pitch_joint', 'left_ankle_roll_joint',
            'right_hip_pitch_joint', 'right_hip_roll_joint', 'right_hip_yaw_joint',
            'right_knee_joint', 'right_ankle_pitch_joint', 'right_ankle_roll_joint'
        ]

        self.waist_joint_names = [
            'waist_yaw_joint', 'waist_roll_joint', 'waist_pitch_joint'
        ]

        # -------------------------------------------------------------------
        # 2. Initialize Bheema Modules
        # -------------------------------------------------------------------
        self.get_logger().info("Initializing Physics Modules...")
        
        self.state_est = G1State() 
        self.mass = self.state_est.get_mass() 
        self.com_height = 0.76
        self.plotter = Bheemaplotter(self.leg_joint_names)
        # Check Mass
        self.get_logger().info(f"Robot Mass: {self.mass:.2f} kg (Should be ~40-50kg)")

        self.N_horizon = 20

        self.mpc = CentroidalMPC(N=self.N_horizon, dt=self.dt_mpc, mass=self.mass)
        self.wbc = WholeBodyController(self.state_est)
        self.scheduler = BipedGaitScheduler(step_time=0.3, double_support_time=0.05)
        self.planner = HLIPFootPlanner(com_height=self.com_height)
        self.ref_gen = ComReference(self.N_horizon, self.dt_mpc)
        self.dyn = CentroidalDynamics(self.mass, np.eye(3))
        self.disc = CentroidalDiscrete(self.dt_mpc)

        self.x_curr = np.zeros((12, 1))
        self.u_mpc = np.zeros(6)
        self.last_support = None
        self.first_run = True
        
        self.planner.get_foot_positions()

        # -------------------------------------------------------------------
        # 3. ROS Interfaces
        # -------------------------------------------------------------------
        self.leg_pub = self.create_publisher(Float64MultiArray, '/g1_leg_controller/commands', 10)
        self.waist_pub = self.create_publisher(Float64MultiArray, '/g1_waist_controller/commands', 10)

        self.create_subscription(JointState, '/joint_states', self.joint_cb, 10)
        self.create_subscription(Odometry, '/simulator/floating_base_state', self.odom_cb, 10)

        self.timer = self.create_timer(self.dt_wbc, self.control_loop)
        
        self.get_logger().info("Bheema Controller Ready. Waiting for Odometry...")
        self.start_time = None
        self.ramp_duration = 3.0 # seconds
      
    # -----------------------------------------------------------------------
    # Callbacks
    # -----------------------------------------------------------------------
    def joint_cb(self, msg):
        if self.state_est.model.nq == 0: return

        pos_map = dict(zip(msg.name, msg.position))
        vel_map = dict(zip(msg.name, msg.velocity))

        q_actuated = []
        dq_actuated = []
        
        for i in range(2, self.state_est.model.njoints):
            name = self.state_est.model.names[i]
            if name in pos_map:
                q_actuated.append(pos_map[name])
                dq_actuated.append(vel_map[name])
            else:
                q_actuated.append(0.0)
                dq_actuated.append(0.0)

        if len(q_actuated) > 0:
            self.state_est.set_joint_state(np.array(q_actuated), np.array(dq_actuated))

    def odom_cb(self, msg):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        
        self.state_est.set_base_pose(
            np.array([p.x, p.y, p.z]),
            np.array([q.x, q.y, q.z, q.w])
        )

        v_lin_body = np.array([msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.linear.z])
        v_ang_body = np.array([msg.twist.twist.angular.x, msg.twist.twist.angular.y, msg.twist.twist.angular.z])

        quat = pin.Quaternion(q.w, q.x, q.y, q.z)
        R_b2w = quat.toRotationMatrix()
        v_lin_world = R_b2w @ v_lin_body
        v_ang_world = R_b2w @ v_ang_body

        self.state_est.set_base_velocity(v_lin_world, v_ang_world)

        if self.first_run:
            self.get_logger().info(f"Odom received! Height: {p.z:.3f}m")
            self.first_run = False

    # -----------------------------------------------------------------------
    # Main Control Loop
    # -----------------------------------------------------------------------
    def control_loop(self):
        if self.first_run: return

        # 1. Update State
        self.state_est.update()
        self.x_curr = self.state_est.get_centroidal_state()
        t = self.get_clock().now().nanoseconds / 1e9

        # 2. Run MPC
        if self.tick_count % self.mpc_decimation == 0:
            self.run_mpc(t)
        
        self.tick_count += 1
        self.log_counter += 1
        if self.start_time is None:
            self.start_time = t

        ramp = min(1.0, (t - self.start_time) / self.ramp_duration)
        self.u_mpc *= ramp

        # # 3. WBC
        # contact_now = self.scheduler.get_contact(t)
        contact_now = (1, 1) # Force Double Support
        swing_data = None
        swing_data = None
        if contact_now == (0, 1):
            swing_data = {'is_left': True, 'pos': self.planner.pL, 'vel': np.zeros(3)}
        elif contact_now == (1, 0):
            swing_data = {'is_left': False, 'pos': self.planner.pR, 'vel': np.zeros(3)}

        tau_full = self.wbc.compute_torques(
            q=self.state_est.q,
            dq=self.state_est.dq,
            f_mpc=self.u_mpc,
            swing_target=swing_data
        )

        # 4. Extract Torques & Log
        leg_torques = []
        tau_map = {}
        for i in range(2, self.state_est.model.njoints):
            name = self.state_est.model.names[i]
            idx_v = self.state_est.model.joints[i].idx_v
            if idx_v < len(tau_full):
                tau_map[name] = tau_full[idx_v]

        for name in self.leg_joint_names:
            leg_torques.append(tau_map.get(name, 0.0))

        # --- LOGGING (Every 0.5s) ---
        if self.log_counter % 100 == 0:
            max_tau = np.max(np.abs(leg_torques))
            fz_L = self.u_mpc[2]
            fz_R = self.u_mpc[5]
            self.get_logger().info(
                f"[WBC] Max Torque: {max_tau:.1f}Nm | MPC Forces (L/R): {fz_L:.0f} / {fz_R:.0f} N | Height: {self.x_curr[2,0]:.2f}m"
            )
            
            # Warn if torques are suspiciously zero
            if max_tau < 0.01:
                self.get_logger().warn("⚠️ Zero Torque Output! Check Jacobian/Mass or if Robot is Floating.")
        self.plotter.record(t, self.x_curr, self.u_mpc, leg_torques)
        # 5. Publish
        leg_msg = Float64MultiArray()
        leg_msg.data = leg_torques
        self.leg_pub.publish(leg_msg)

        waist_msg = Float64MultiArray()
        waist_msg.data = [0.0, 0.0, 0.0]
        self.waist_pub.publish(waist_msg)

    # -----------------------------------------------------------------------
    # MPC
    # -----------------------------------------------------------------------
    def run_mpc(self, t):
        contact_table = self.scheduler.get_contact_table(t, self.dt_mpc, self.N_horizon)
        
        sL, sR = contact_table[0,0], contact_table[1,0]
        curr_supp = "left" if (sL and not sR) else "right" if (sR and not sL) else None
        
        if curr_supp and curr_supp != self.last_support:
            self.planner.update_step(curr_supp, self.x_curr[0,0], self.x_curr[6,0])
        self.last_support = curr_supp

        v_cmd = np.array([0.0, 0.0, 0.0]) 
        x_ref = self.ref_gen.generate(self.x_curr, v_cmd, self.com_height)

        pL_plan, pR_plan = self.planner.pL, self.planner.pR
        Ad_seq, Bd_seq, gd_seq = [], [], []
        
        for k in range(self.N_horizon):
            pc = x_ref[0:3, k]
            rL, rR = pL_plan - pc, pR_plan - pc
            Ac, Bc, gc = self.dyn.continuous_matrices(rL, rR)
            Ad, Bd, gd = self.disc.discretize(Ac, Bc, gc)
            Ad_seq.append(Ad); Bd_seq.append(Bd); gd_seq.append(gd)

        try:
            self.u_mpc = self.mpc.solve(Ad_seq, Bd_seq, gd_seq, self.x_curr, x_ref, contact_table)
            
            # Log specific MPC events
            # self.get_logger().info(f"[MPC] Solved. Support: {curr_supp or 'Double'}")
            
        except Exception as e:
            self.get_logger().warn(f"MPC Failed: {e}")
            self.u_mpc = np.zeros(6)

def main(args=None):
    rclpy.init(args=args)
    node = BheemaControlNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()