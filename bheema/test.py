# test_wbc.py
import numpy as np
import pinocchio as pin
from state import G1State
from whole_body_controller import WholeBodyController

def print_norm(name, v):
    print(f"{name:20s} | norm = {np.linalg.norm(v):.6f}")

def assert_small(name, v, tol=1e-3):
    n = np.linalg.norm(v)
    print_norm(name, v)
    assert n < tol, f"{name} too large: {n}"

# ---------------------------------------------------------
def build_state():
    robot = G1State()
    q = pin.neutral(robot.model)
    dq = np.zeros(robot.nv)

    # [CRITICAL FIX] Bend knees to avoid singularity (Straight leg = singular)
    # Finding knee indices by name (assuming standard G1 naming)
    for i, name in enumerate(robot.model.names):
        if "knee" in name:
            idx = robot.model.joints[i].idx_q
            if idx >= 0:
                q[idx] = 0.3  # Bend knee slightly

    # Adjust height to match bent knees (approximate)
    q[2] = 0.73 
    
    robot.update(q, dq)
    return robot, q.copy()

def build_wbc(state):
    return WholeBodyController(state)

# ---------------------------------------------------------
def test_posture_only():
    print("\n=== TEST 1: Posture regulation ===")
    state, q_nominal = build_state()
    wbc = build_wbc(state)
    
    # Run controller
    q_des = wbc.compute_joint_targets(
        state, np.zeros(6), np.ones((2, 1)), {}, q_nominal
    )
    
    dq = pin.difference(state.model, q_nominal, q_des)
    assert_small("posture error", dq)

def test_pelvis_orientation():
    print("\n=== TEST 2: Pelvis orientation ===")
    state, q_nominal = build_state()
    
    # Tilt the robot state
    q_tilted = q_nominal.copy()
    q_tilted[3:7] = pin.Quaternion(pin.rpy.rpyToMatrix(0.1, -0.1, 0.0)).coeffs()
    state.update(q_tilted, np.zeros(state.nv))
    
    wbc = build_wbc(state)
    q_des = wbc.compute_joint_targets(
        state, np.zeros(6), np.ones((2, 1)), {}, q_nominal
    )
    
    dq = pin.difference(state.model, q_tilted, q_des)
    print_norm("pelvis correction", dq)
    assert np.linalg.norm(dq) > 1e-4

def test_single_swing():
    print("\n=== TEST 3: Swing foot ===")
    state, q_nominal = build_state()
    wbc = build_wbc(state)
    contact = np.array([[0], [1]]) # Left swing
    pL, _ = state.get_foot_positions_world()
    
    swing_trajs = {"left": {"pos": pL + np.array([0.0, 0.0, 0.05])}}
    
    q_des = wbc.compute_joint_targets(
        state, np.zeros(6), contact, swing_trajs, q_nominal
    )
    
    dq = pin.difference(state.model, q_nominal, q_des)
    print_norm("swing motion", dq)
    assert np.linalg.norm(dq) > 1e-4

def test_contact_switch():
    print("\n=== TEST 4: Contact switch ===")
    state, q_nominal = build_state()
    wbc = build_wbc(state)
    for contact in [np.array([[0],[1]]), np.array([[1],[1]])]:
        q_des = wbc.compute_joint_targets(
            state, np.zeros(6), contact, {}, q_nominal
        )
        dq = pin.difference(state.model, q_nominal, q_des)
        print(f"contact: {contact.flatten()}, dq norm: {np.linalg.norm(dq):.6f}")

def test_posture_does_not_affect_swing():
    print("\n=== TEST 5: Swing convergence ===")
    state, q_nominal = build_state()
    wbc = build_wbc(state)

    contact = np.array([[0], [1]])
    pL, _ = state.get_foot_positions_world()

    # Target: 5cm up
    p_des = pL + np.array([0, 0, 0.05])
    swing_trajs = {"left": {"pos": p_des}}

    # Initial Error
    e_before = p_des - pL

    # Compute Control
    q_des = wbc.compute_joint_targets(
        state, np.zeros(6), contact, swing_trajs, q_nominal
    )

    # Check Result using Kinematics on q_des
    # Note: We must create a NEW data object or be careful not to corrupt state.data
    # wbc.data contains the kinematics of q_des because compute_joint_targets doesn't update it to q_des at the end
    # We must manually compute FK for q_des
    data_check = state.model.createData()
    pin.forwardKinematics(state.model, data_check, q_des)
    pin.updateFramePlacements(state.model, data_check)

    p_new = data_check.oMf[wbc.left_foot_frame].translation
    e_after = p_des - p_new

    print(f"Error Before: {np.linalg.norm(e_before):.6f}")
    print(f"Error After : {np.linalg.norm(e_after):.6f}")

    # Check if error decreased
    assert np.linalg.norm(e_after) < np.linalg.norm(e_before), "Controller failed to reduce error"

if __name__ == "__main__":
    test_posture_only()
    test_pelvis_orientation()
    test_single_swing()
    test_contact_switch()
    test_posture_does_not_affect_swing()
    print("\nAll WBC tests passed.")