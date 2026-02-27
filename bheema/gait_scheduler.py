import numpy as np


class BipedGaitScheduler:
    """
    Simple time-based biped walking scheduler.

    Contact state:
        1 = contact
        0 = swing

    contact_table shape:
        (2, N)
        row 0 → Left foot
        row 1 → Right foot
    """

    def __init__(
        self,
        step_time: float,
        double_support_time: float
    ):
        """
        step_time: duration of single support phase
        double_support_time: duration of double support
        """

        self.step_time = step_time
        self.ds_time = double_support_time

        # Full cycle duration
        self.cycle_time = (
            self.ds_time
            + self.step_time
            + self.ds_time
            + self.step_time
        )

    # --------------------------------------------------
    # Determine instantaneous contact state
    # --------------------------------------------------

    def get_contact(self, t: float):
        """
        Returns:
            (left_contact, right_contact)
        """

        t_mod = t % self.cycle_time

        # Phase 1: Double Support
        if t_mod < self.ds_time:
            return 1, 1

        # Phase 2: Left Support (right foot swing)
        elif t_mod < self.ds_time + self.step_time:
            return 1, 0

        # Phase 3: Double Support
        elif t_mod < self.ds_time + self.step_time + self.ds_time:
            return 1, 1

        # Phase 4: Right Support (left foot swing)
        else:
            return 0, 1

    # --------------------------------------------------
    # Generate MPC Horizon Contact Table
    # --------------------------------------------------

    def get_contact_table(self, t0: float, dt: float, N: int):
        """
        Build contact table for horizon.

        Returns:
            contact_table (2, N)
        """

        table = np.zeros((2, N), dtype=int)

        for k in range(N):
            t = t0 + k * dt
            left, right = self.get_contact(t)
            table[0, k] = left
            table[1, k] = right

        return table
