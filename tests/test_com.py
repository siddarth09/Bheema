import numpy as np
from bheema.com_ref import ComReference


def main():

    N = 10
    dt = 0.02

    ref = ComReference(N, dt)

    x0 = np.zeros((12,1))
    x0[2] = 1.0  # initial COM height

    v_des = np.array([0.3, 0.0, 0.0])
    z_des = 1.0

    x_ref = ref.generate(
        x0=x0,
        v_des_world=v_des,
        z_des=z_des,
        yaw_rate_des=0.0
    )

    print("x_ref shape:", x_ref.shape)
    print("First step:", x_ref[:,0])
    print("Last step:", x_ref[:,-1])


if __name__ == "__main__":
    main()
