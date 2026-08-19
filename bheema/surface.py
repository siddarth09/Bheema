"""
Terrain surface queries.

The controller needs three things about the ground that it currently does not ask for:

  1. the height under a candidate foothold,
  2. the contact normal there (for the friction cone and CoP limits), and
  3. the extent of the surface, so "the foot lands fully on one face" can be written as
     linear constraints on the touchdown point.

`SurfaceQuery` provides all three behind one interface with two backends:

  * `SurfaceQuery`          -- oracle. Ray-casts against the MuJoCo model. Exact, ~free.
  * `PerceivedSurfaceQuery` -- multi-plane segmentation of a depth point cloud.

Developing the controller against the oracle is the only way to tell a controller bug from
a perception bug, so both stay selectable for the life of the project.

Surfaces are represented as **rectangles**, not general convex polygons: centre, two
in-plane axes, two half-extents. Terrain is boxes by construction, so a rectangle is exact,
and it is what Atlas's perception tracks ("detected rectangular faces"). Four parameters is
also far more robust to fit from noisy depth than a vertex list.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import mujoco as mj
import numpy as np

from bheema.terrain import TERRAIN_GEOM_GROUP

NGROUP = getattr(mj, "mjNGROUP", 6)
UP = np.array([0.0, 0.0, 1.0])


@dataclass(frozen=True)
class Surface:
    """One planar, rectangular, walkable face."""

    id: int                      # MuJoCo geom id; stable within a scene
    name: str
    normal: np.ndarray           # (3,) unit, pointing away from the material (upward-ish)
    offset: float                # plane is {x : normal . x = offset}
    center: np.ndarray           # (3,) centre of the rectangular face
    axes: np.ndarray             # (3, 2) orthonormal in-plane axes
    half_extents: np.ndarray     # (2,) along axes[:, 0] and axes[:, 1]
    is_plane: bool = False       # infinite ground plane (the floor geom)

    def height_at(self, x: float, y: float) -> float:
        """z where the vertical line through (x, y) meets this plane."""
        nx, ny, nz = self.normal
        if abs(nz) < 1e-9:
            return float("nan")          # vertical face: no unique height
        return float((self.offset - nx * x - ny * y) / nz)

    def local_xy(self, x: float, y: float) -> np.ndarray:
        """In-plane coordinates of (x, y) relative to the face centre."""
        p = np.array([x, y, self.height_at(x, y)])
        return self.axes.T @ (p - self.center)

    def contains(self, x: float, y: float, margin: float = 0.0) -> bool:
        """Is (x, y) inside the face, shrunk by `margin` on every side?

        `margin` is how the foot's own half-extent (and, later, perception uncertainty)
        gets accounted for: pass the foot half-length and a foothold that "contains" the
        point is one the whole foot fits on.
        """
        if self.is_plane:
            return True
        u = np.abs(self.local_xy(x, y))
        lim = self.half_extents - margin
        return bool(np.all(lim > 0) and np.all(u <= lim))

    def edge_distance(self, x: float, y: float) -> float:
        """Signed distance to the nearest face edge; positive inside.

        Primary term in foothold scoring.
        """
        if self.is_plane:
            return float("inf")
        return float(np.min(self.half_extents - np.abs(self.local_xy(x, y))))

    @property
    def inclination_deg(self) -> float:
        return float(np.degrees(np.arccos(np.clip(abs(self.normal[2]), -1.0, 1.0))))

    def corners(self) -> np.ndarray:
        """(4, 3) world-frame corners, for drawing the face in the viewer."""
        a, b = self.axes[:, 0] * self.half_extents[0], self.axes[:, 1] * self.half_extents[1]
        return np.array([self.center - a - b, self.center + a - b,
                         self.center + a + b, self.center - a + b])


@dataclass
class SurfaceHit:
    valid: bool
    z: float
    normal: np.ndarray
    surface: Surface | None = None

    @classmethod
    def miss(cls) -> "SurfaceHit":
        return cls(False, float("nan"), UP.copy(), None)


class SurfaceQuery:
    """Oracle backend: exact terrain from the MuJoCo model via ray casting.

    Rays are masked to TERRAIN_GEOM_GROUP. The robot's collision geoms are group 3, so an
    unmasked downward ray from above the robot would hit its legs.
    """

    def __init__(self, model, data, ray_from_z: float = 3.0):
        self.model = model
        self.data = data
        self.ray_from_z = ray_from_z
        self._mask = np.zeros(NGROUP, dtype=np.uint8)
        self._mask[TERRAIN_GEOM_GROUP] = 1
        self._gid = np.zeros(1, dtype=np.int32)
        self._down = np.array([0.0, 0.0, -1.0])
        # Terrain geoms are static, so their extracted faces never change. Cache by geom id.
        self._surfaces: dict[int, Surface] = {}

    # -- public API ---------------------------------------------------------------

    def at(self, x: float, y: float, from_z: float | None = None) -> SurfaceHit:
        """Downward ray at (x, y): height, contact normal, and the owning surface."""
        origin = np.array([x, y, self.ray_from_z if from_z is None else from_z])
        dist = mj.mj_ray(self.model, self.data, origin, self._down,
                         self._mask, 1, -1, self._gid)
        if dist < 0:
            return SurfaceHit.miss()
        gid = int(self._gid[0])
        z = float(origin[2] - dist)
        hit_point = np.array([x, y, z])
        return SurfaceHit(True, z, self._face_normal(gid, hit_point), self.surface_of(gid))

    def height(self, x: float, y: float) -> float:
        """Just the height. NaN on a miss -- callers must not silently treat that as 0."""
        return self.at(x, y).z

    def max_height_between(self, p0, p1, n: int = 9) -> tuple[float, SurfaceHit]:
        """Highest terrain strictly between two footholds.

        This is what swing clearance should be measured against, replacing the fixed
        HEIGHT_SWING constant: the foot has to clear whatever is in the way, which on a box
        course is the box's own top edge.
        """
        p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
        best_z, best = -np.inf, SurfaceHit.miss()
        for s in np.linspace(0.0, 1.0, n + 2)[1:-1]:      # exclude the endpoints
            q = p0 + s * (p1 - p0)
            hit = self.at(q[0], q[1])
            if hit.valid and hit.z > best_z:
                best_z, best = hit.z, hit
        return (best_z if np.isfinite(best_z) else float("nan")), best

    def surfaces_near(self, x: float, y: float, radius: float = 0.6,
                      n: int = 7) -> list[Surface]:
        """Distinct surfaces found on a grid around (x, y). Foothold-search candidates."""
        found: dict[int, Surface] = {}
        for gx in np.linspace(x - radius, x + radius, n):
            for gy in np.linspace(y - radius, y + radius, n):
                hit = self.at(gx, gy)
                if hit.valid and hit.surface is not None:
                    found[hit.surface.id] = hit.surface
        return list(found.values())

    # -- geometry ----------------------------------------------------------------

    def surface_of(self, gid: int) -> Surface:
        if gid not in self._surfaces:
            self._surfaces[gid] = self._extract(gid)
        return self._surfaces[gid]

    def _extract(self, gid: int) -> Surface:
        m, d = self.model, self.data
        name = mj.mj_id2name(m, mj.mjtObj.mjOBJ_GEOM, gid) or f"geom_{gid}"
        c = np.array(d.geom_xpos[gid], dtype=float)
        R = np.array(d.geom_xmat[gid], dtype=float).reshape(3, 3)

        if m.geom_type[gid] == mj.mjtGeom.mjGEOM_PLANE:
            # `size` for a plane is (x_halfwidth, y_halfwidth, grid_spacing) and is 0 for an
            # infinite plane -- it is NOT a half-extent. Treating it as one would compute a
            # 0.05 m rectangle and reject every foothold on flat ground.
            n = R[:, 2]
            return Surface(gid, name, n, float(n @ c), c, R[:, :2],
                           np.array([np.inf, np.inf]), is_plane=True)

        h = np.array(m.geom_size[gid], dtype=float)

        # Walkable face = the one whose outward normal points most upward. Handles rotated
        # boxes (ramps) with no special case.
        best, axis, sign = -np.inf, 2, 1.0
        for k in range(3):
            for s in (-1.0, 1.0):
                up = (s * R[:, k]) @ UP
                if up > best:
                    best, axis, sign = up, k, s

        n = sign * R[:, axis]
        center = c + sign * h[axis] * R[:, axis]
        others = [k for k in range(3) if k != axis]
        axes = np.column_stack([R[:, others[0]], R[:, others[1]]])
        return Surface(gid, name, n, float(n @ center), center, axes,
                       np.array([h[others[0]], h[others[1]]]))

    def _face_normal(self, gid: int, p_world: np.ndarray) -> np.ndarray:
        """Outward normal of the face the ray actually struck.

        Taking `geom_xmat[:, 2]` here is a real trap: that is the +z face's normal, so a ray
        clipping a vertical side returns a normal that is 90 degrees wrong. It then feeds the
        friction pyramid and presents as a controller bug.

        The face is chosen by |local coordinate| / half_extent, not raw |local coordinate|:
        on a box that is long in one axis -- e.g. the 0.80 x 0.60 x 0.125 platform -- the raw
        comparison picks the wrong face.
        """
        m, d = self.model, self.data
        if m.geom_type[gid] == mj.mjtGeom.mjGEOM_PLANE:
            return np.array(d.geom_xmat[gid], dtype=float).reshape(3, 3)[:, 2]

        c = np.array(d.geom_xpos[gid], dtype=float)
        R = np.array(d.geom_xmat[gid], dtype=float).reshape(3, 3)
        h = np.array(m.geom_size[gid], dtype=float)
        p_local = R.T @ (p_world - c)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.abs(p_local) / np.where(h > 1e-12, h, np.inf)
        k = int(np.argmax(ratio))
        return float(np.sign(p_local[k]) or 1.0) * R[:, k]


class PerceivedSurfaceQuery:
    """Perception backend: multi-plane segmentation of a depth point cloud.

    Deliberately takes `(point_cloud, sensor_pose)` rather than a MuJoCo handle, so the
    cloud can come from an in-process depth render, a recorded file, or a ROS 2
    `PointCloud2` subscription without any change to the segmentation or to callers.

    Pipeline: per-point normals by local PCA; region grow on normal similarity and plane
    offset (offset is required -- stair treads share a normal); fit a rectangle per patch;
    shrink by a safety margin, since depth smearing at discontinuities inflates faces past
    their true edge; track by (n, d) across frames.
    """

    def __init__(self, margin: float = 0.02):
        self.margin = margin
        self.surfaces: list[Surface] = []

    def update(self, point_cloud: np.ndarray, sensor_pose: np.ndarray) -> None:
        raise NotImplementedError

    def at(self, x: float, y: float) -> SurfaceHit:
        """Highest tracked surface whose (shrunk) rectangle contains (x, y)."""
        best = SurfaceHit.miss()
        for s in self.surfaces:
            if s.contains(x, y, margin=self.margin):
                z = s.height_at(x, y)
                if not best.valid or z > best.z:
                    best = SurfaceHit(True, z, s.normal, s)
        return best


def make_surface_query(kind: str, model=None, data=None, **kw):
    """Factory so callers name a backend instead of importing a class."""
    if kind == "oracle":
        if model is None or data is None:
            raise ValueError("oracle backend needs (model, data)")
        return SurfaceQuery(model, data, **kw)
    if kind == "perceived":
        return PerceivedSurfaceQuery(**kw)
    raise KeyError(f"unknown backend {kind!r}; use 'oracle' or 'perceived'")
