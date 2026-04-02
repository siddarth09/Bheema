import mujoco
import mujoco.viewer
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# 1. Load your specific G1 XML
# Make sure this matches the filename you use in main.py
XML_PATH = "/home/sid/projects25/src/bheema/unitree_g1/scene_with_hands.xml" 
model = mujoco.MjModel.from_xml_path(XML_PATH)
data = mujoco.MjData(model)

# 2. Start the passive MuJoCo viewer (runs in the background)
viewer = mujoco.viewer.launch_passive(model, data)

# 3. Setup the Matplotlib GUI Slider
fig, ax = plt.subplots(figsize=(6, 2.5))
fig.canvas.manager.set_window_title('Bheema Height & Kinematics Tuner')
plt.subplots_adjust(bottom=0.4)
ax_slider = plt.axes([0.2, 0.2, 0.6, 0.15])

# Theta represents how much the knees are bent. 
# 0.0 = Locked straight, 1.0 = Deep squat
slider = Slider(ax_slider, 'Squat Depth\n(Theta)', 0.0, 1.0, valinit=0.3)

def set_joint_qpos(name, value):
    """Helper to safely set joint angles by name"""
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if joint_id != -1:
        qpos_adr = model.jnt_qposadr[joint_id]
        data.qpos[qpos_adr] = value

def update(val):
    theta = slider.val
    
    # A. Calculate the new Pelvis Z-Height
    # The G1's thigh and calf are roughly 0.3m each. 
    # The height drops based on the cosine of the bend angle.
    z_drop = 0.6 * (1.0 - np.cos(theta))
    nominal_straight_z = 0.793 
    current_z = nominal_straight_z - z_drop
    
    # B. Apply the Kinematic "Z-Fold" pattern to the joints
    joints = [
        ("left_hip_pitch_joint", -theta),
        ("left_knee_joint", 2 * theta),
        ("left_ankle_pitch_joint", -theta),
        ("right_hip_pitch_joint", -theta),
        ("right_knee_joint", 2 * theta),
        ("right_ankle_pitch_joint", -theta)
    ]
    
    for name, pos in joints:
        set_joint_qpos(name, pos)
        
    # C. Update the floating base (pelvis) height so feet stay on the ground
    data.qpos[2] = current_z
    
    # D. Push updates to the MuJoCo viewer instantly
    mujoco.mj_kinematics(model, data) # Updates positions without running physics
    viewer.sync()
    
    # Update the window text so you know exactly what NOMINAL_Z to use in main.py
    fig.suptitle(f"Pelvis Height (NOMINAL_Z): {current_z:.3f} m", fontsize=14, fontweight='bold')

# Hook up the slider and initialize the first frame
slider.on_changed(update)
update(0.3)

print("\n--- BHEEMA KINEMATICS TUNER ---")
print("1. Click the Matplotlib window.")
print("2. Drag the slider left and right.")
print("3. Watch the MuJoCo viewer to see how NOMINAL_Z affects the knee bend.")
print("Close the Matplotlib window to exit.\n")

plt.show()

# Cleanup
viewer.close()