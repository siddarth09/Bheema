import numpy as np
from bheema.gait_scheduler import BipedGaitScheduler


def test_scheduler():

    print("\n=== TEST: Biped Gait Scheduler ===")

    scheduler = BipedGaitScheduler(
        step_time=0.5,
        double_support_time=0.2
    )

    dt = 0.1
    N = 10

    t0 = 0.0

    table = scheduler.get_contact_table(t0, dt, N)

    print("Contact Table:")
    print(table)

    # Ensure only valid values
    assert np.all((table == 0) | (table == 1))

    # Ensure at least one support leg always exists
    for k in range(N):
        assert table[:, k].sum() >= 1

    print("✔ Scheduler passed.")


if __name__ == "__main__":
    test_scheduler()
