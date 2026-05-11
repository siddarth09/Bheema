"""
Bheema RL — Custom G1 Flat Terrain Locomotion
==============================================

Forks mjlab's built-in G1 velocity environment and customizes
the reward weights to shape the walking behavior we want.

This is the correct approach: don't rebuild the entire env from scratch.
The observations, actions, terminations, sensors, and domain randomization
in mjlab's velocity task are battle-tested and sim-to-real proven.
What makes YOUR policy unique is the REWARD FUNCTION.

Train:
    python -m mjlab.scripts.train Bheema-Velocity-Flat-G1 --env.scene.num-envs 2048

Play:
    python -m mjlab.scripts.play Bheema-Velocity-Flat-G1-Play --checkpoint_file logs/<run>/model_*.pt
"""

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner
from mjlab.tasks.velocity.config.g1.env_cfgs import unitree_g1_flat_env_cfg
from mjlab.tasks.velocity.config.g1.rl_cfg import unitree_g1_ppo_runner_cfg


def bheema_g1_flat_env_cfg(play: bool = False):
    """
    Start from mjlab's G1 flat config, then override reward weights.
    We customize the REWARD WEIGHTS to shape walking behavior.
    """
    cfg = unitree_g1_flat_env_cfg(play=play)

    # =================================================================
    # REWARD TUNING — This is where your locomotion knowledge lives
    # =================================================================
    #
    # Each reward term already exists in the base config.
    # We adjust the WEIGHTS to prioritize what we care about.
    #
    # Positive weights = "do more of this"
    # Negative weights = "do less of this"
    #
    # The base config weights are tuned for general-purpose walking.
    # Below we tune for stable, efficient bipedal locomotion.
    # =================================================================

    # --- VELOCITY TRACKING (the primary task) ---
    # Higher weight = more aggressive tracking, less energy-efficient
    # Lower weight = lazier tracking, smoother motion
    if "track_lin_vel_xy" in cfg.rewards:
        cfg.rewards["track_lin_vel_xy"].weight = 1.5  # Default is ~1.0
    
    if "track_ang_vel_z" in cfg.rewards:
        cfg.rewards["track_ang_vel_z"].weight = 0.8

    # --- ENERGY / SMOOTHNESS ---
    # These prevent the robot from learning jerky, motor-burning gaits
    if "joint_torques" in cfg.rewards:
        cfg.rewards["joint_torques"].weight = -1e-4
    
    if "action_rate" in cfg.rewards:
        cfg.rewards["action_rate"].weight = -0.01  # Smooth actions
    
    if "action_acc" in cfg.rewards:
        cfg.rewards["action_acc"].weight = -0.005  # Even smoother

    # --- BODY STABILITY ---
    # Keep the torso upright and at the right height
    if "upright" in cfg.rewards:
        cfg.rewards["upright"].weight = -1.0  # Penalize tilting

    if "body_ang_vel" in cfg.rewards:
        cfg.rewards["body_ang_vel"].weight = -0.05  # Don't spin wildly

    # --- GAIT QUALITY ---
    # Foot clearance: lift feet properly during swing (no shuffling)
    if "foot_clearance" in cfg.rewards:
        cfg.rewards["foot_clearance"].weight = 0.5

    # Air time: encourage proper swing duration
    if "air_time" in cfg.rewards:
        cfg.rewards["air_time"].weight = 0.5  # Base has 0.0, we enable it

    # Foot slip: don't slide feet on ground during stance
    if "foot_slip" in cfg.rewards:
        cfg.rewards["foot_slip"].weight = -0.1

    # --- POSTURE ---
    # Stay close to default joint angles (prevents weird stances)
    if "pose" in cfg.rewards:
        cfg.rewards["pose"].weight = -0.5

    # --- SELF COLLISION ---
    if "self_collisions" in cfg.rewards:
        cfg.rewards["self_collisions"].weight = -1.0

    # --- ANGULAR MOMENTUM ---
    # Penalize excessive angular momentum (keeps motion clean)
    if "angular_momentum" in cfg.rewards:
        cfg.rewards["angular_momentum"].weight = -0.02

    # --- JOINT LIMITS ---
    if "joint_limits" in cfg.rewards:
        cfg.rewards["joint_limits"].weight = -1.0

    return cfg


# =================================================================
# REGISTER THE TASK
# =================================================================

register_mjlab_task(
    task_id="Bheema-Velocity-Flat-G1",
    env_cfg=bheema_g1_flat_env_cfg(),
    play_env_cfg=bheema_g1_flat_env_cfg(play=True),
    rl_cfg=unitree_g1_ppo_runner_cfg(),
    runner_cls=VelocityOnPolicyRunner,
)