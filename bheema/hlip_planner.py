import numpy as np


class HLIPFootPlanner:
    """
    HLIP-based foot placement planner for biped walking.

    Uses 1D Linear Inverted Pendulum Model in sagittal plane.

    State inputs:
        x_com  : COM x position
        v_com  : COM x velocity

    Output:
        Updated foot lever arms rL, rR (foot position - COM)
    """

    def __init__(self, com_height: float, g: float = 9.81):

        self.h = com_height
        self.g = g

        # Natural frequency
        self.omega = np.sqrt(g / com_height)

        # Nominal lateral stance width
        self.step_width = 0.2

        # Internal foot world positions
        self.pL = np.array([0.0,  self.step_width/2, 0.0])
        self.pR = np.array([0.0, -self.step_width/2, 0.0])

        self.initialized = False

    # ---------------------------------------------------------
    # Capture Point
    # ---------------------------------------------------------

    def compute_capture_point(self, x_com, v_com):
        """
        Instantaneous capture point for LIP:

            x_cp = x + v/omega
        """
        return x_com + v_com / self.omega

    # ---------------------------------------------------------
    # Update Step
    # ---------------------------------------------------------

    def update_step(self, support_leg, x_com, v_com):
        """
        Called at support switch.

        Moves swing foot to capture location.
        """

        x_cp = self.compute_capture_point(x_com, v_com)

        # Add small forward bias for progression
        alpha = 0.6  # damping gain
        step_length = alpha * (x_cp - x_com)
        
        # Clip step length to avoid unrealistic jumps
        step_length = np.clip(step_length, -0.4, 0.4)

        new_foot_x = x_com + step_length

        if support_leg == "left":
            # move right foot
            self.pR[0] = new_foot_x
        elif support_leg == "right":
            # move left foot
            self.pL[0] = new_foot_x

        return self.get_foot_levers(x_com)

    # ---------------------------------------------------------
    # Get Lever Arms
    # ---------------------------------------------------------

    def get_foot_levers(self, x_com):
        """
        Returns lever arms r = p_foot - p_com
        """

        p_com = np.array([x_com, 0.0, self.h])

        rL = self.pL - p_com
        rR = self.pR - p_com

        return rL, rR

    # ---------------------------------------------------------
    # Initialization helper
    # ---------------------------------------------------------

    def get_foot_positions(self):
        """
        Initial lever arms assuming COM at x=0
        """
        p_com = np.array([0.0, 0.0, self.h])
        rL = self.pL - p_com
        rR = self.pR - p_com
        return rL, rR
