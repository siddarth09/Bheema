import numpy as np
from bheema.hlip_planner import HLIPFootPlanner


def test_capture_point():

    planner = HLIPFootPlanner(com_height=1.0)

    x = 0.0
    v = 0.5

    cp = planner.compute_capture_point(x, v)

    print("Capture point:", cp)

    assert cp > 0.0
    print("✔ HLIP capture test passed.")


if __name__ == "__main__":
    test_capture_point()
