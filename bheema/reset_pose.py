import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState # NEW: Need to know where joints are
import time
import numpy as np

class SoftWiggleNode(Node):
    def __init__(self):
        super().__init__('soft_wiggle_node')
        self.pub_legs = self.create_publisher(Float64MultiArray, '/g1_leg_controller/commands', 10)
        
        self.leg_joint_names = [
            'left_hip_pitch_joint', 'left_hip_roll_joint', 'left_hip_yaw_joint',
            'left_knee_joint', 'left_ankle_pitch_joint', 'left_ankle_roll_joint',
            'right_hip_pitch_joint', 'right_hip_roll_joint', 'right_hip_yaw_joint',
            'right_knee_joint', 'right_ankle_pitch_joint', 'right_ankle_roll_joint'
        ]
        
        # State tracking
        self.current_q = {}
        self.create_subscription(JointState, '/joint_states', self.state_cb, 10)
        
        self.get_logger().info("Waiting for joint states...")
        while rclpy.ok() and not self.current_q:
            rclpy.spin_once(self)

        self.run_test()

    def state_cb(self, msg):
        for i, name in enumerate(msg.name):
            self.current_q[name] = msg.position[i]

    def run_test(self):
        for i, name in enumerate(self.leg_joint_names):
            self.get_logger().info(f"Subtle wiggle: {name}")
            self.apply_soft_wiggle(i, name)

    def apply_soft_wiggle(self, target_idx, name):
        msg = Float64MultiArray()
        q_start = self.current_q[name]
        start_t = time.time()
        
        # PD Gains for the wiggle (matches your G1 mass)
        kp = 40.0 
        kd = 2.0
        
        while (time.time() - start_t) < 1.5:
            t = time.time() - start_t
            # Very small oscillation: 0.05 radians
            q_target = q_start + 0.05 * np.sin(2 * np.pi * 2.0 * t)
            
            # Simple PD to compute torque
            tau = kp * (q_target - self.current_q[name])
            
            cmd = [0.0] * 12
            cmd[target_idx] = tau
            msg.data = cmd
            self.pub_legs.publish(msg)
            time.sleep(0.01)
            rclpy.spin_once(self, timeout_sec=0)

def main():
    rclpy.init()
    SoftWiggleNode()
    rclpy.shutdown()

if __name__ == '__main__':
    main()