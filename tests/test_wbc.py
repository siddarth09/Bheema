import numpy as np
import pinocchio as pin
import unittest

# Import your modules
from bheema.state import G1State
from bheema.wbc import WholeBodyController

class TestWBC(unittest.TestCase):

    def setUp(self):
        print("\n[Setup] Initializing G1 State and WBC...")
        self.state = G1State()
        self.wbc = WholeBodyController(self.state)
        
        self.model = self.state.model
        self.nq = self.model.nq
        self.nv = self.model.nv
        
        # Helper to create a valid state with BENT KNEES
        self.q0 = pin.neutral(self.model)
        self.dq0 = np.zeros(self.nv)
        
        # --- CRITICAL FIX: Bend Knees to avoid Singularity ---
        # We need to find the knee joint indices. 
        # Since we don't know exact IDs, we iterate names.
        for i, name in enumerate(self.model.names):
            if "knee" in name:
                # Joint index in configuration vector q
                idx_q = self.model.joints[i].idx_q
                if idx_q < len(self.q0):
                    self.q0[idx_q] = 0.5  # Bend knees by 0.5 rad

        # Update internal state once
        self.state.q = self.q0.copy()
        self.state.dq = self.dq0.copy()
        self.state.update()

    def test_standing_torques(self):
        print("[Test] Standing Posture...")
        
        f_mpc = np.zeros(6) 
        
        # Verify inputs to prevent Segfaults
        self.assertEqual(self.q0.shape[0], self.nq, "q vector wrong size")
        self.assertEqual(self.dq0.shape[0], self.nv, "dq vector wrong size")

        tau = self.wbc.compute_torques(self.q0, self.dq0, f_mpc, swing_target=None)

        self.assertEqual(tau.shape[0], self.nv)
        
        # Gravity should create torque on bent knees
        gravity_torque = np.linalg.norm(tau[6:])
        print(f"    Joint Torques (Gravity Comp): {gravity_torque:.2f}")
        self.assertTrue(gravity_torque > 0.1, "WBC should produce gravity compensation torque!")

    def test_reaction_force_mapping(self):
        print("[Test] Reaction Force Mapping...")

        # 1. Baseline (Gravity only)
        tau_0 = self.wbc.compute_torques(self.q0, self.dq0, np.zeros(6))

        # 2. Apply 100N upwards on Left Foot
        # Force vector: [fx_L, fy_L, fz_L, fx_R, fy_R, fz_R]
        f_mpc = np.array([0, 0, 100.0, 0, 0, 0]) 
        
        tau_100 = self.wbc.compute_torques(self.q0, self.dq0, f_mpc)

        # 3. Difference
        tau_diff = tau_100 - tau_0
        diff_norm = np.linalg.norm(tau_diff[6:])
        
        print(f"    Torque difference from 100N lift: {diff_norm:.2f} Nm")

        # Now that knees are bent, this MUST be > 0
        self.assertTrue(diff_norm > 1.0, 
            "Applying force to bent leg should produce joint torques (Jacobian Transpose).")

if __name__ == '__main__':
    unittest.main()