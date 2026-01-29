import pinocchio as pin
import numpy as np
from pinocchio.robot_wrapper import RobotWrapper
from pathlib import Path

# -----------------------------
# Paths
# -----------------------------
BHEEMA_PATH = Path("/home/sid/projects25/src/bheema/unitree_g1/g1_with_hands.xml")


def quat_to_rpy(quat_xyzw: np.ndarray) -> np.ndarray:
    """Quaternion (x,y,z,w) -> roll,pitch,yaw (Pinocchio ZYX convention)."""
    q = pin.Quaternion(quat_xyzw)
    R = q.toRotationMatrix()
    rpy = pin.rpy.matrixToRpy(R)
    return np.asarray(rpy).reshape(3,)


class G1State:
    """
    Pinocchio-backed state interface for BHEEMA.

    Provides:
      - generalized state (q, dq)
      - COM position/velocity
      - base orientation (rpy)
      - base angular velocity in world frame
      - foot positions and lever arms (foot - COM)
      - total mass
      - centroidal inertia about COM (Ig.inertia)
    """

    def __init__(self, mjcf_path: Path = BHEEMA_PATH):
        self.robot = RobotWrapper.BuildFromMJCF(
            str(mjcf_path),
            root_joint=pin.JointModelFreeFlyer()
        )

        self.model = self.robot.model
        self.data = self.model.createData()

        # Frame IDs (must exist in MJCF/Pinocchio model)
        self.base_frame = self.model.getFrameId("pelvis")
        self.left_foot_frame = self.model.getFrameId("left_foot")
        self.right_foot_frame = self.model.getFrameId("right_foot")

        # Dimensions
        self.nq = self.model.nq
        self.nv = self.model.nv

        # Stored state
        self.q = pin.neutral(self.model)
        self.dq = np.zeros(self.nv)

        # Cached quantities (updated each update())
        self._updated = False

    # -----------------------------
    # Core update
    # -----------------------------
    def update(self, q: np.ndarray, dq: np.ndarray) -> None:
        """
        Update Pinocchio internal buffers at (q, dq).
        Must be called before any getters each control tick.
        """
        self.q = np.asarray(q).copy()
        self.dq = np.asarray(dq).copy()

        pin.forwardKinematics(self.model, self.data, self.q, self.dq)
        pin.updateFramePlacements(self.model, self.data)
        pin.computeAllTerms(self.model, self.data, self.q, self.dq)
        pin.centerOfMass(self.model, self.data, self.q, self.dq)
        pin.ccrba(self.model, self.data, self.q, self.dq)  # ensures data.Ig is valid

        self._updated = True

    def _require_updated(self):
        if not self._updated:
            raise RuntimeError("G1State.update(q,dq) must be called before reading state.")

    # -----------------------------
    # Generalized state
    # -----------------------------
    def get_generalized_state(self):
        self._require_updated()
        return self.q.copy(), self.dq.copy()

    # -----------------------------
    # Physical parameters
    # -----------------------------
    def get_total_mass(self) -> float:
        self._require_updated()
        return float(self.data.mass[0])


    def get_centroidal_inertia_com(self) -> np.ndarray:
        """
        Centroidal inertia about COM (3x3).
        NOTE: Pinocchio stores centroidal momentum/inertia in a particular convention;
        for our simplified centroidal MPC we will treat this as the inertia in WORLD alignment
        as long as omega is world-aligned and r is world.
        """
        self._require_updated()
        return np.asarray(self.data.Ig.inertia).copy()

    # -----------------------------
    # Centroidal state pieces
    # -----------------------------
    def get_com_state_world(self):
        """COM position and velocity in WORLD frame."""
        self._require_updated()
        p_com = np.asarray(self.data.com[0]).copy()
        v_com = np.asarray(self.data.vcom[0]).copy()
        return p_com, v_com

    def get_base_orientation_rpy(self):
        """Base orientation (r,p,y) from free-flyer quaternion, in radians."""
        self._require_updated()
        quat_xyzw = self.q[3:7]
        return quat_to_rpy(quat_xyzw)

    def get_base_angular_velocity_world(self):
        """
        Base angular velocity in WORLD frame.
        For free-flyer, dq[3:6] is the base angular velocity in BODY frame.
        """
        self._require_updated()
        omega_body = self.dq[3:6]
        oMb = self.data.oMf[self.base_frame]
        R_wb = np.asarray(oMb.rotation)
        omega_world = R_wb @ omega_body
        return omega_world

    # -----------------------------
    # Feet
    # -----------------------------
    def get_foot_positions_world(self):
        self._require_updated()
        oMfL = self.data.oMf[self.left_foot_frame]
        oMfR = self.data.oMf[self.right_foot_frame]
        return np.asarray(oMfL.translation).copy(), np.asarray(oMfR.translation).copy()

    def get_foot_levers_world(self):
        self._require_updated()
        p_com, _ = self.get_com_state_world()
        pL, pR = self.get_foot_positions_world()
        return (pL - p_com), (pR - p_com)

    
    # -----------------------------
    def get_bheema_x(self) -> np.ndarray:
        """
        Returns BHEEMA centroidal MPC state in the LOCKED order:

          x = [ p_com(3),
                rpy(3),
                v_com(3),
                omega_world(3) ]  (12x1)

        This must match your centroidal discrete model.
        """
        self._require_updated()
        p_com, v_com = self.get_com_state_world()
        rpy = self.get_base_orientation_rpy()
        omega = self.get_base_angular_velocity_world()

        x = np.concatenate([p_com, rpy, v_com, omega])
        return x.reshape(12, 1)


# -----------------------------
# Quick local sanity test
# -----------------------------
def main():
    robot = G1State()

    q = pin.neutral(robot.model)
    dq = np.zeros(robot.model.nv)
    q[2] = 0.75

    robot.update(q, dq)

    print("\n--- Dimensions ---")
    print("nq:", robot.nq, "nv:", robot.nv)

    print("\n--- Physical params ---")
    print("mass:", robot.get_total_mass())
    print("Ig inertia:\n", robot.get_centroidal_inertia_com())

    x = robot.get_bheema_x()
    print("\n--- BHEEMA x (12) ---")
    print(x.flatten())

    pL, pR = robot.get_foot_positions_world()
    rL, rR = robot.get_foot_levers_world()
    p_com, _ = robot.get_com_state_world()

    print("\n--- Feet ---")
    print("pL:", pL)
    print("pR:", pR)
    print("rL:", rL)
    print("rR:", rR)
    print("\nCheck p_com + rL:", p_com + rL)
    print("Check p_com + rR:", p_com + rR)


if __name__ == "__main__":
    main()
