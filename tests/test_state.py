import numpy as np
from bheema.state import G1State
import pinocchio as pin

state = G1State()

# Fake base pose
state.set_base_pose(
    np.array([0,0,0.75]),
    np.array([0,0,0,1])
)

# Fake zero velocity
state.set_base_velocity(
    np.zeros(3),
    np.zeros(3)
)

# Fake joints
state.set_joint_state(
    np.zeros(state.nv - 6),
    np.zeros(state.nv - 6)
)

state.update()

print("COM:", state.get_com())
print("Centroidal x:", state.get_centroidal_state().flatten())
