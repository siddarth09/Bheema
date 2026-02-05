#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np
import pinocchio as pin
from sensor_msgs.msg import JointState
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64MultiArray

# --- BHEEMA imports ---
from bheema.state import G1State
from bheema.centroidal_dynamics import CentroidalDynamics, CentroidalDiscreteModel
from bheema.centroidal_mpc import CentroidalMPC
from bheema.whole_body_controller import WholeBodyController
from bheema.contact_scheduler import ContactSchedule, WalkingGaitParams

class BheemaControlNode(Node):
    def __init__(self):
        super().__init__('bheema_controller')

        # --- Timing ---
        self.wbc_rate = 200.0
        self.dt = 1.0 / self.wbc_rate
        self.sim_time = 0.0
        self.log_counter = 0

        # --- PHASES ---
        self.startup_mode = True       
        self.balance_mode = False      
        self.walking_mode = False      
        self.abort_mode   = False      

        self.startup_tick = 0
        self.startup_duration = 2.0
        self.balance_duration = 2.0    
        
        # [TUNING] NEUTRAL BALANCE
        self.target_height = 0.72       
        self.target_knee   = 0.5        
        self.target_hip    = -0.35      
        self.target_ankle  = -0.15      
        self.target_waist  = 0.0        
        
        # Offset Logic
        self.z_offset = 0.0
        self.offset_locked = False
        self.start_xy = np.zeros(2)

        # MPC settings
        self.mpc_dt = 0.02
        self.mpc_horizon = 10
        self.mpc_decimation = int(self.mpc_dt / self.dt)
        self.mpc_tick = 0

        # --- Robot Logic ---
        self.state = G1State()
        self.model = self.state.model
        self.idx_map = {name: self.model.joints[i].idx_q for i, name in enumerate(self.model.names)}

        # Initialize State
        q_init = pin.neutral(self.model)
        dq_init = np.zeros(self.model.nv)
        
        if 'left_knee_joint' in self.idx_map: q_init[self.idx_map['left_knee_joint']] = self.target_knee
        if 'right_knee_joint' in self.idx_map: q_init[self.idx_map['right_knee_joint']] = self.target_knee
        if 'left_hip_pitch_joint' in self.idx_map: q_init[self.idx_map['left_hip_pitch_joint']] = self.target_hip
        if 'right_hip_pitch_joint' in self.idx_map: q_init[self.idx_map['right_hip_pitch_joint']] = self.target_hip
        if 'left_ankle_pitch_joint' in self.idx_map: q_init[self.idx_map['left_ankle_pitch_joint']] = self.target_ankle
        if 'right_ankle_pitch_joint' in self.idx_map: q_init[self.idx_map['right_ankle_pitch_joint']] = self.target_ankle
        if 'waist_pitch_joint' in self.idx_map: q_init[self.idx_map['waist_pitch_joint']] = self.target_waist
        q_init[2] = self.target_height
        
        self.state.update(q_init, dq_init)

        # --- Controller ---
        self.wbc = WholeBodyController(self.state)
        self.wbc.w_pelvis_ori = 600.0   
        self.wbc.w_pelvis_height = 200.0
        self.wbc.w_foot = 200.0
        self.wbc.w_posture = 50.0 

        mass = self.state.get_total_mass()
        inertia = self.state.get_centroidal_inertia_com()
        self.dynamics = CentroidalDynamics(mass, inertia)
        self.discrete_model = CentroidalDiscreteModel(self.mpc_dt)
        self.mpc = CentroidalMPC(self.mpc_horizon, self.mpc_dt)

        self.gait_params = WalkingGaitParams(step_time=0.8, ds_fraction=0.4, start_with_left_stance=True)
        self.scheduler = ContactSchedule(self.gait_params)

        # --- ROS Setup ---
        self.leg_joints = [
            'left_hip_pitch_joint', 'left_hip_roll_joint', 'left_hip_yaw_joint',
            'left_knee_joint', 'left_ankle_pitch_joint', 'left_ankle_roll_joint',
            'right_hip_pitch_joint', 'right_hip_roll_joint', 'right_hip_yaw_joint',
            'right_knee_joint', 'right_ankle_pitch_joint', 'right_ankle_roll_joint'
        ]
        self.waist_joints = ['waist_yaw_joint', 'waist_roll_joint', 'waist_pitch_joint']
        self.active_joint_names = self.leg_joints + self.waist_joints
        
        self.pin_q_indices = []
        self.pin_v_indices = []
        for name in self.active_joint_names:
            if self.model.existJointName(name):
                joint_id = self.model.getJointId(name)
                self.pin_q_indices.append(self.model.joints[joint_id].idx_q)
                self.pin_v_indices.append(self.model.joints[joint_id].idx_v)
        
        self.qj_ros = np.zeros(len(self.active_joint_names))
        self.vj_ros = np.zeros(len(self.active_joint_names))
        self.ros_name_to_idx = {name: i for i, name in enumerate(self.active_joint_names)}

        self.base_pos = np.array([0.0, 0.0, self.target_height])
        self.base_quat = np.array([0.0, 0.0, 0.0, 1.0])
        self.base_lin_vel = np.zeros(3)
        self.base_ang_vel = np.zeros(3)

        self.leg_pub = self.create_publisher(Float64MultiArray, '/g1_leg_controller/commands', 10)
        self.waist_pub = self.create_publisher(Float64MultiArray, '/g1_waist_controller/commands', 10)
        
        self.create_subscription(JointState, '/joint_states', self.joint_state_cb, 10)
        self.create_subscription(Odometry, '/simulator/floating_base_state', self.odom_cb, 10)

        self.f_mpc = np.zeros(6)
        self.timer = self.create_timer(self.dt, self.control_loop)
        self.get_logger().info("BHEEMA Control Node (Aggressive Offset) started")

    def joint_state_cb(self, msg):
        for i, name in enumerate(msg.name):
            if name in self.ros_name_to_idx:
                idx = self.ros_name_to_idx[name]
                self.qj_ros[idx] = msg.position[i]
                self.vj_ros[idx] = msg.velocity[i]

    def odom_cb(self, msg):
        raw_z = msg.pose.pose.position.z
        
        # [FIX] Aggressive Offset Logic
        # We don't lock it immediately. We keep checking for the high value.
        if not self.offset_locked:
            if raw_z > 1.2:
                # We found the Sky Spawn! Force the known XML offset.
                self.z_offset = 0.793
                self.offset_locked = True
                self.get_logger().info(f"OFFSET LOCKED: Sky Spawn detected. Forcing {self.z_offset}")
            elif self.sim_time > 0.5:
                # If 0.5s passed and no Sky Spawn, assume normal spawn
                self.z_offset = raw_z - self.target_height
                self.offset_locked = True
                self.get_logger().info(f"OFFSET LOCKED: Normal Spawn. Calibrated {self.z_offset:.3f}")

        # Safety Fallback: Even if not locked, apply temp correction if high
        temp_offset = self.z_offset
        if temp_offset == 0.0 and raw_z > 1.2:
            temp_offset = 0.793

        self.base_pos = np.array([
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            raw_z - temp_offset
        ])

        self.base_quat = np.array([
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w
        ])

        v_body = np.array([msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.linear.z])
        w_body = np.array([msg.twist.twist.angular.x, msg.twist.twist.angular.y, msg.twist.twist.angular.z])
        Rwb = pin.Quaternion(self.base_quat).toRotationMatrix()
        self.base_lin_vel = Rwb @ v_body
        self.base_ang_vel = Rwb @ w_body

    def log_diagnostics(self, x_cur, q_cmd):
        if self.log_counter % 50 != 0: return

        phase = "WALKING"
        if self.abort_mode: phase = "ABORTED"
        elif self.startup_mode: phase = "STARTUP"
        elif self.balance_mode: phase = "BALANCING"

        h_cur = x_cur[2].item()
        rpy = x_cur[3:6].flatten()

        print(f"\n[{phase}] T={self.sim_time:.2f}s | H={h_cur:.3f}")
        print(f"  > R/P: {rpy[0]:.3f} / {rpy[1]:.3f}")

    def control_loop(self):
        if self.abort_mode: return

        q_full = pin.neutral(self.model)
        v_full = np.zeros(self.model.nv)
        q_full[0:3] = self.base_pos
        q_full[3:7] = self.base_quat
        v_full[0:3] = self.base_lin_vel
        v_full[3:6] = self.base_ang_vel
        
        for i in range(len(self.active_joint_names)):
            q_full[self.pin_q_indices[i]] = self.qj_ros[i]
            v_full[self.pin_v_indices[i]] = self.vj_ros[i]

        self.state.update(q_full, v_full)
        x_cur = self.state.get_bheema_x()

        # --- SAFETY CHECK ---
        rpy = x_cur[3:6].flatten()
        if abs(rpy[1]) > 0.4: 
            print(f">>> CRITICAL FAILURE: TIPPING DETECTED (Pitch={rpy[1]:.2f}). ABORTING. <<<")
            self.abort_mode = True
            return

        # ==========================================================
        # PHASE 1: STARTUP
        # ==========================================================
        if self.startup_mode:
            self.startup_tick += 1
            progress = min(1.0, (self.startup_tick * self.dt) / self.startup_duration)
            
            curr_knee  = 0.0 + (self.target_knee  - 0.0) * progress
            curr_hip   = 0.0 + (self.target_hip   - 0.0) * progress
            curr_ankle = 0.0 + (self.target_ankle - 0.0) * progress
            curr_waist = 0.0 + (self.target_waist - 0.0) * progress

            q_des = np.zeros(len(self.active_joint_names))
            for i, name in enumerate(self.active_joint_names):
                if 'knee' in name: q_des[i] = curr_knee
                elif 'hip_pitch' in name: q_des[i] = curr_hip
                elif 'ankle_pitch' in name: q_des[i] = curr_ankle
                elif 'waist_pitch' in name: q_des[i] = curr_waist

            self.publish_joints(q_des)
            self.log_diagnostics(x_cur, q_des)

            if progress >= 1.0:
                print(">>> STARTUP DONE. BALANCING... <<<")
                self.startup_mode = False
                self.balance_mode = True
                self.sim_time = 0.0  
            
            self.log_counter += 1
            return

        # ==========================================================
        # PHASE 2: ACTIVE BALANCE
        # ==========================================================
        if self.balance_mode:
            self.start_xy = x_cur[0:2].flatten()
            cL, cR = 1, 1 
            
            if self.mpc_tick % self.mpc_decimation == 0:
                contact_plan = self.scheduler.build(self.sim_time, self.mpc_dt, self.mpc_horizon)
                x_ref = np.zeros((12, self.mpc_horizon))
                x_ref[0, :] = self.start_xy[0] 
                x_ref[1, :] = self.start_xy[1]
                x_ref[2, :] = self.target_height
                
                rL, rR = self.state.get_foot_levers_world()
                Ac, Bc, gc = self.dynamics.continuous_matrices(rL, rR)
                Ad, Bd, gd = self.discrete_model.discretize(Ac, Bc, gc)
                try:
                    u = self.mpc.build_and_solve([Ad]*self.mpc_horizon, [Bd]*self.mpc_horizon, [gd]*self.mpc_horizon, x_cur, x_ref, contact_plan)
                    self.f_mpc = u[:6]
                except: pass
            
            if self.sim_time > self.balance_duration:
                if abs(rpy[0]) < 0.1 and abs(rpy[1]) < 0.1:
                    print(f">>> BALANCE STABLE. WALKING START. <<<")
                    self.balance_mode = False
                    self.walking_mode = True
                    self.sim_time = 0.0 
                else:
                    print(f">>> UNSTABLE (Pitch={rpy[1]:.2f}). HOLDING BALANCE... <<<")
                    self.sim_time = 0.0 

        else:
            # PHASE 3: WALKING
            cL, cR = self.scheduler._contact_at_time(self.sim_time)
            if self.mpc_tick % self.mpc_decimation == 0:
                contact_plan = self.scheduler.build(self.sim_time, self.mpc_dt, self.mpc_horizon)
                x_ref = np.zeros((12, self.mpc_horizon))
                x_ref[0, :] = self.start_xy[0] 
                x_ref[1, :] = self.start_xy[1]
                x_ref[2, :] = self.target_height
                
                rL, rR = self.state.get_foot_levers_world()
                Ac, Bc, gc = self.dynamics.continuous_matrices(rL, rR)
                Ad, Bd, gd = self.discrete_model.discretize(Ac, Bc, gc)
                try:
                    u = self.mpc.build_and_solve([Ad]*self.mpc_horizon, [Bd]*self.mpc_horizon, [gd]*self.mpc_horizon, x_cur, x_ref, contact_plan)
                    self.f_mpc = u[:6]
                except: pass

        self.mpc_tick += 1

        # WBC
        contact_table = np.array([[cL], [cR]])
        
        q_nom = pin.neutral(self.model)
        if 'left_knee_joint' in self.idx_map: q_nom[self.idx_map['left_knee_joint']] = self.target_knee
        if 'right_knee_joint' in self.idx_map: q_nom[self.idx_map['right_knee_joint']] = self.target_knee
        if 'left_hip_pitch_joint' in self.idx_map: q_nom[self.idx_map['left_hip_pitch_joint']] = self.target_hip
        if 'right_hip_pitch_joint' in self.idx_map: q_nom[self.idx_map['right_hip_pitch_joint']] = self.target_hip
        if 'left_ankle_pitch_joint' in self.idx_map: q_nom[self.idx_map['left_ankle_pitch_joint']] = self.target_ankle
        if 'right_ankle_pitch_joint' in self.idx_map: q_nom[self.idx_map['right_ankle_pitch_joint']] = self.target_ankle
        if 'waist_pitch_joint' in self.idx_map: q_nom[self.idx_map['waist_pitch_joint']] = self.target_waist
        q_nom[2] = self.target_height

        swing_trajs = {}
        if self.walking_mode:
            swing_h = 0.08
            t_cyc = self.sim_time % self.gait_params.step_time
            phase = t_cyc / self.gait_params.step_time
            
            target_y_L = self.start_xy[1] + 0.15
            target_y_R = self.start_xy[1] - 0.15

            if cL == 0:
                z_lift = swing_h * np.sin(np.pi * phase)
                swing_trajs['left'] = {"pos": np.array([self.start_xy[0], target_y_L, z_lift])}
            if cR == 0:
                z_lift = swing_h * np.sin(np.pi * phase)
                swing_trajs['right'] = {"pos": np.array([self.start_xy[0], target_y_R, z_lift])}

        q_des_full = self.wbc.compute_joint_targets(
            state=self.state, f_mpc=self.f_mpc, 
            contact_table=contact_table, swing_trajs=swing_trajs, q_nominal=q_nom
        )

        q_active_des = [q_des_full[self.pin_q_indices[i]] for i in range(len(self.active_joint_names))]
        self.publish_joints(q_active_des)
        
        self.log_diagnostics(x_cur, q_active_des)
        self.log_counter += 1
        self.sim_time += self.dt

    def publish_joints(self, q_des):
        leg_msg = Float64MultiArray()
        waist_msg = Float64MultiArray()
        leg_msg.data = list(q_des[0:12])
        waist_msg.data = list(q_des[12:15])
        self.leg_pub.publish(leg_msg)
        self.waist_pub.publish(waist_msg)

def main(args=None):
    rclpy.init(args=args)
    node = BheemaControlNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()