"""
Bheema — RL Policy Teleop
===========================
Controls:
    ↑ / ↓       Forward / Backward
    ← / →       Strafe Left / Right
    Z / X       Turn Left / Right
    C / V       Raise / Lower CoM
    SPACE       Emergency Stop
    ESC         Quit
"""
import os
os.environ["MPLBACKEND"] = "TkAgg"
import time
import mujoco as mj
import mujoco.viewer as mjv
import numpy as np

from teleop import Teleop
from rl_bridge import RLBridge
from pathlib import Path
from huggingface_hub import hf_hub_download

# ─── Config ───
_ROOT = Path(__file__).parent.parent  # bheema_rl/ → bheema/ → project root
MJCF_PATH = str(_ROOT / "unitree_g1" / "scene_with_hands.xml")

# Download checkpoint from HuggingFace (cached after first download)
CHECKPOINT = hf_hub_download(
    repo_id="Siddarth09/bheema_locomotion",
    filename="model_59999.pt",
)


SIM_HZ = 1000
SIM_DT = 1.0 / SIM_HZ
POLICY_HZ = 50.0
RUN_LENGTH_S = 300.0
RENDER_HZ = 60.0
SIM_STEPS = int(RUN_LENGTH_S * SIM_HZ)

# ─── Load model ───
print("Loading MuJoCo model...")
model = mj.MjModel.from_xml_path(MJCF_PATH)
data = mj.MjData(model)
model.opt.timestep = SIM_DT

base_bid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "pelvis")

# ─── Initialize from keyframe ───
key_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_KEY, "stand")
if key_id != -1:
    mj.mj_resetDataKeyframe(model, data, key_id)
    print(f"Loaded keyframe 'stand': base z = {data.qpos[2]:.3f}")
else:
    print("WARNING: 'stand' keyframe not found, using defaults")
    data.qpos[2] = 0.74

mj.mj_forward(model, data)
print(f"Initial base height: {data.qpos[2]:.3f}")

# ─── Load policy ───
print("Loading RL policy...")
rl = RLBridge(CHECKPOINT, model, policy_hz=POLICY_HZ)

# ─── Start teleop ───
print("Starting teleop...")
teleop = Teleop(max_vx=1.5, max_vy=0.5, max_yaw_rate=0.6, ramp_rate=2.0)
teleop.start()

# ─── Warmup: hold standing pose with control active ───
print("Warming up (holding standing pose)...")
for _ in range(int(0.5 * SIM_HZ)):
    rl.step(data, [0.0, 0.0, 0.0], data.time)
    rl.apply_control(data)
    mj.mj_step(model, data)
print(f"After warmup: base z = {data.qpos[2]:.3f}")

# ─── Main Loop ───
print(f"\nRunning RL teleop — Arrow keys to walk, ESC to quit\n")
sim_start = time.perf_counter()

with mjv.launch_passive(model, data) as viewer:
    viewer.cam.type = mj.mjtCamera.mjCAMERA_TRACKING
    viewer.cam.trackbodyid = base_bid
    viewer.cam.distance = 3.0
    viewer.cam.elevation = -20
    viewer.cam.azimuth = 90
    viewer.opt.flags[mj.mjtVisFlag.mjVIS_CONTACTPOINT] = True

    for k in range(SIM_STEPS):
        if not viewer.is_running() or not teleop.is_running():
            break

        time_now = float(data.time)

        # Read teleop
        vx, vy, _, yaw_rate = teleop.get_cmd()

        # RL policy step (rate-limited internally)
        rl.step(data, [vx, vy, yaw_rate], time_now)

        # Apply control at SIM_HZ
        rl.apply_control(data)

        # Physics
        mj.mj_step(model, data)

        # Render
        if k % int(SIM_HZ / RENDER_HZ) == 0:
            viewer.sync()

            if k % int(SIM_HZ / 2) == 0:
                print(f"\r  t={time_now:6.1f}s  "
                      f"vx={vx:+.2f}  vy={vy:+.2f}  yaw={yaw_rate:+.2f}  "
                      f"z={data.qpos[2]:.3f}  ",
                      end="", flush=True)

            wall_elapsed = time.perf_counter() - sim_start
            sim_ahead = time_now - wall_elapsed
            if sim_ahead > 0:
                time.sleep(sim_ahead)

teleop.stop()
print(f"\n\nDone. Simulated {data.time:.1f}s")