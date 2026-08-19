"""
Depth sensing.

    cam = DepthCamera(model, data)
    cloud = cam.point_cloud()          # (N, 3) in world frame
    pose  = cam.pose()                 # 4x4 sensor-to-world

Renders the `head_depth` camera, unprojects to a point cloud, and optionally records
(cloud, pose) pairs to an npz for offline segmentation work. Nothing here touches the
control loop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import mujoco as mj
import numpy as np

from bheema.terrain import TERRAIN_GEOM_GROUP

CAMERA_NAME = "head_depth"


@dataclass(frozen=True)
class DepthCameraParams:
    """Time-of-flight depth camera, matched to the Atlas parkour sensor rate."""

    camera: str = CAMERA_NAME
    width: int = 320
    height: int = 240
    rate_hz: float = 15.0
    z_near: float = 0.15          # m, drop returns closer than this
    z_far: float = 4.0            # m, drop returns beyond this
    stride: int = 2               # pixel decimation before unprojection
    noise_std: float = 0.0        # m, gaussian range noise, proportional to depth^2
    dropout: float = 0.0          # fraction of returns discarded at random
    # Render terrain only, excluding the robot's own links. A real stack self-filters
    # using the robot model; masking the render is the cheap equivalent. Set False to keep
    # self-returns and exercise a real self-filter.
    exclude_self: bool = True


class DepthCamera:
    """Offscreen depth rendering and unprojection for a body-mounted camera."""

    def __init__(self, model, data, params: DepthCameraParams | None = None,
                 rng: np.random.Generator | None = None):
        self.model = model
        self.data = data
        self.p = params or DepthCameraParams()
        self.rng = rng or np.random.default_rng(0)

        self.cam_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_CAMERA, self.p.camera)
        if self.cam_id < 0:
            raise KeyError(f"camera {self.p.camera!r} not in the model")

        self._renderer = mj.Renderer(model, height=self.p.height, width=self.p.width)
        self._renderer.enable_depth_rendering()
        self._rays = self._pixel_rays()

        self._scene_option = None
        if self.p.exclude_self:
            opt = mj.MjvOption()
            mj.mjv_defaultOption(opt)
            opt.geomgroup[:] = 0
            opt.geomgroup[TERRAIN_GEOM_GROUP] = 1
            self._scene_option = opt

    def close(self) -> None:
        self._renderer.close()

    # -- geometry ----------------------------------------------------------------

    def _pixel_rays(self) -> np.ndarray:
        """Unit rays through each (strided) pixel, in camera frame.

        MuJoCo's camera looks along -z with +y up. fovy is the vertical field of view in
        degrees; the horizontal extent follows from the aspect ratio.
        """
        p = self.p
        fovy = np.deg2rad(float(self.model.cam_fovy[self.cam_id]))
        f = 0.5 * p.height / np.tan(0.5 * fovy)          # focal length in pixels
        cx, cy = 0.5 * p.width, 0.5 * p.height

        v, u = np.mgrid[0:p.height:p.stride, 0:p.width:p.stride]
        x = (u.ravel() + 0.5 - cx) / f
        y = -(v.ravel() + 0.5 - cy) / f
        rays = np.stack([x, y, -np.ones_like(x)], axis=1)
        return rays / np.linalg.norm(rays, axis=1, keepdims=True)

    def pose(self) -> np.ndarray:
        """4x4 homogeneous sensor-to-world transform."""
        T = np.eye(4)
        T[:3, :3] = np.asarray(self.data.cam_xmat[self.cam_id]).reshape(3, 3)
        T[:3, 3] = np.asarray(self.data.cam_xpos[self.cam_id])
        return T

    # -- sensing -----------------------------------------------------------------

    def depth_image(self) -> np.ndarray:
        """(height, width) metric depth along the camera z axis."""
        if self._scene_option is None:
            self._renderer.update_scene(self.data, camera=self.cam_id)
        else:
            self._renderer.update_scene(self.data, camera=self.cam_id,
                                        scene_option=self._scene_option)
        return np.asarray(self._renderer.render())

    def point_cloud(self, in_world: bool = True) -> np.ndarray:
        """(N, 3) points. Invalid, too-near and too-far returns are removed."""
        p = self.p
        depth = self.depth_image()[::p.stride, ::p.stride].ravel()

        if p.noise_std > 0.0:
            depth = depth + self.rng.normal(0.0, p.noise_std * depth**2, depth.shape)

        keep = np.isfinite(depth) & (depth > p.z_near) & (depth < p.z_far)
        if p.dropout > 0.0:
            keep &= self.rng.random(depth.shape) >= p.dropout
        if not keep.any():
            return np.zeros((0, 3))

        # Depth is measured along -z, and each ray's z component is -1 before
        # normalisation, so range = depth / |ray_z|.
        rays = self._rays[keep]
        pts = rays * (depth[keep] / np.abs(rays[:, 2]))[:, None]

        if not in_world:
            return pts
        T = self.pose()
        return pts @ T[:3, :3].T + T[:3, 3]


@dataclass
class CloudRecording:
    """Sequence of (time, cloud, pose) samples."""

    times: list[float] = field(default_factory=list)
    clouds: list[np.ndarray] = field(default_factory=list)
    poses: list[np.ndarray] = field(default_factory=list)

    def add(self, t: float, cloud: np.ndarray, pose: np.ndarray) -> None:
        self.times.append(float(t))
        self.clouds.append(np.asarray(cloud, dtype=np.float32))
        self.poses.append(np.asarray(pose, dtype=np.float64))

    def __len__(self) -> int:
        return len(self.times)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            times=np.asarray(self.times),
            poses=np.asarray(self.poses),
            counts=np.asarray([len(c) for c in self.clouds]),
            points=np.concatenate(self.clouds) if self.clouds else np.zeros((0, 3), np.float32),
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> "CloudRecording":
        z = np.load(path)
        rec = cls()
        offset = 0
        for t, pose, n in zip(z["times"], z["poses"], z["counts"]):
            rec.add(float(t), z["points"][offset:offset + n], pose)
            offset += int(n)
        return rec


def record(scene: str, duration_s: float = 12.0, walk: bool = True,
           out: str | Path = "recordings/depth.npz",
           params: DepthCameraParams | None = None) -> Path:
    """Walk a scene and record the depth stream.

    Runs the normal controller, so the camera moves as the robot does. Recording is
    read-only with respect to the control loop.
    """
    from bheema.sim import DEFAULT_CMD_SCHEDULE, WALK_CMD_SCHEDULE, SimConfig, run

    cfg = SimConfig(scene=scene, duration_s=duration_s, headless=True, verbose=False,
                    cmd_schedule=WALK_CMD_SCHEDULE if walk else DEFAULT_CMD_SCHEDULE)
    res = run(cfg)

    model = mj.MjModel.from_xml_path(scene)
    data = mj.MjData(model)
    cam = DepthCamera(model, data, params)
    rec = CloudRecording()
    step = max(1, int(round(cfg.ctrl_hz / cam.p.rate_hz)))

    for i in range(0, res.n_ticks, step):
        data.qpos[:] = res.q_log[i]
        mj.mj_forward(model, data)
        rec.add(res.t[i], cam.point_cloud(), cam.pose())
    cam.close()
    return rec.save(out)


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="record a depth stream while walking")
    ap.add_argument("--scene", default="unitree_g1/scenes/scene_platform_easy.xml")
    ap.add_argument("--duration", type=float, default=12.0)
    ap.add_argument("--out", default="recordings/depth.npz")
    ap.add_argument("--width", type=int, default=320)
    ap.add_argument("--height", type=int, default=240)
    ap.add_argument("--rate-hz", type=float, default=15.0)
    ap.add_argument("--noise-std", type=float, default=0.0)
    ap.add_argument("--dropout", type=float, default=0.0)
    args = ap.parse_args()

    p = DepthCameraParams(width=args.width, height=args.height, rate_hz=args.rate_hz,
                          noise_std=args.noise_std, dropout=args.dropout)
    path = record(args.scene, args.duration, out=args.out, params=p)
    rec = CloudRecording.load(path)
    pts = sum(len(c) for c in rec.clouds)
    print(f"wrote {path}: {len(rec)} frames, {pts} points "
          f"({pts / max(len(rec), 1):.0f} per frame)")


if __name__ == "__main__":
    main()
