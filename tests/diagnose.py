import pinocchio as pin
import numpy as np
import sys
from bheema.state import G1State

def test_pinocchio_stability():
    print("=== Pinocchio Stability Diagnostic ===")
    
    # 1. Load State
    try:
        print("[1] Loading G1State...")
        state = G1State()
        model = state.model
        data = state.data
        print(f"    Model Loaded. nq={model.nq}, nv={model.nv}")
        print(f"    Frames: Base={state.base_frame}, LF={state.left_foot_frame}")
    except Exception as e:
        print(f"!!! Failed to load state: {e}")
        return

    # 2. Setup Vectors (Enforce Safe Types)
    print("[2] Setting up vectors...")
    q = pin.neutral(model)
    v = np.zeros(model.nv)
    a = np.zeros(model.nv)
    
    # Enforce contiguity (The Fix)
    q = np.ascontiguousarray(q, dtype=np.float64)
    v = np.ascontiguousarray(v, dtype=np.float64)
    a = np.ascontiguousarray(a, dtype=np.float64)

    # 3. Test Kinematics
    print("[3] Running Forward Kinematics...")
    try:
        pin.forwardKinematics(model, data, q, v)
        pin.updateFramePlacements(model, data)
        print("    PK passed.")
    except Exception as e:
        print(f"!!! FK Failed: {e}")

    # 4. Test RNEA (Inverse Dynamics) - Common Crash Point
    print("[4] Running RNEA (Gravity)...")
    try:
        tau = pin.rnea(model, data, q, v, a)
        print(f"    RNEA passed. Tau norm: {np.linalg.norm(tau):.4f}")
    except Exception as e:
        print(f"!!! RNEA Failed: {e}")

    # 5. Test Jacobians - Common Crash Point
    print("[5] Computing Frame Jacobian...")
    try:
        pin.computeJointJacobians(model, data, q)
        J = pin.computeFrameJacobian(model, data, q, state.left_foot_frame, pin.LOCAL_WORLD_ALIGNED)
        print(f"    Jacobian passed. Shape: {J.shape}")
    except Exception as e:
        print(f"!!! Jacobian Failed: {e}")

    print("=== Diagnostic Complete: If you see this, Pinocchio is fine. ===")

if __name__ == "__main__":
    test_pinocchio_stability()