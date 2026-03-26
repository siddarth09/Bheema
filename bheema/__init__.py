"""
Bheema — Nonlinear MPC bipedal locomotion for Unitree G1.

Architecture (based on Galliker et al. 2022, "Bipedal Locomotion with NMPC"):
  - Centroidal NMPC  (CasADi + IPOPT) at ~20 Hz   → 6D contact wrenches
  - Whole-Body Controller (Pinocchio RNEA) at 1 kHz → joint torques
  - MuJoCo simulation at 1 kHz
"""
