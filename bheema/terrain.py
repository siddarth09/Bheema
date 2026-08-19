"""
Planar box terrain generation for BHEEMA-Parkour.

Every terrain element is a MuJoCo `box` geom. No heightfields. This mirrors the real
Atlas parkour course (plywood boxes, platforms, ramps, beams, gaps) and buys two
properties the controller depends on:

  1. Every contact surface has an exact, constant normal -- read off the geom frame
     instead of finite-differencing a height grid.
  2. Every foothold region is a convex polygon, so "the foot lands fully on one
     surface" is a set of *linear* inequalities on the touchdown point.

All terrain geoms are placed in geom group TERRAIN_GEOM_GROUP so that ray queries can
mask the robot out. The group number is not arbitrary -- verified against the compiled
model, g1_with_hands.xml puts every one of its geoms in group 2 (visual mesh) or 3
(collision mesh), and nothing in 0 or 1:

  * group 3 is wrong: a downward ray from above the robot would hit its own legs.
  * group 4+ is wrong: mjv_defaultOption leaves geomgroup = [1,1,1,0,0,0], so terrain
    in group 4 is *invisible* in the viewer and in offscreen renders unless every
    consumer remembers to flip opt.geomgroup[4]. Easy to forget, confusing to debug.
  * group 0 is correct: free of robot geoms, so it uniquely identifies terrain, and
    visible by default. It is also what the original scene_with_hands.xml floor used
    (no group attribute => 0).

Scenes are written to unitree_g1/scenes/. MuJoCo resolves `meshdir` relative to the
top-level model file's directory rather than the file that declares it, so a generated
scene in a subdirectory must restate `meshdir` -- and must do so *after* the include,
since the later compiler element wins. Verified: without the override the compiler
looks for 'assets/unitree_g1/pelvis.STL' and fails.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------------

# Robot geoms occupy groups 2 and 3 only. Terrain takes group 0: unique to terrain
# (so it doubles as a ray mask) and visible by default. See the module docstring.
TERRAIN_GEOM_GROUP = 0

SCENES_DIR = Path(__file__).parent.parent / "unitree_g1" / "scenes"
ROBOT_XML = "../g1_with_hands.xml"
MESHDIR = "../assets"

LEVELS = {"easy": 0.0, "medium": 0.5, "atlas": 1.0}

# G1 reference dimensions (34.4 kg, 29 DoF). Nominal CoM height is 0.66 m in a
# bent-knee stance and the leg is roughly 0.70 m hip-to-foot, which is why the
NOMINAL_COM_HEIGHT = 0.66


def lerp(a: float, b: float, t: float) -> float:
    return float(a + (b - a) * t)


# --------------------------------------------------------------------------------
# Primitives
# --------------------------------------------------------------------------------

@dataclass(frozen=True)
class Box:
    """One planar terrain element. `size` is MuJoCo half-extents."""
    name: str
    size: tuple[float, float, float]
    pos: tuple[float, float, float]
    euler: tuple[float, float, float] = (0.0, 0.0, 0.0)
    material: str = "plywood"

    def to_xml(self, indent: str = "    ") -> str:
        sx, sy, sz = self.size
        px, py, pz = self.pos
        attrs = [
            f'name="{self.name}"',
            'type="box"',
            f'size="{sx:.6f} {sy:.6f} {sz:.6f}"',
            f'pos="{px:.6f} {py:.6f} {pz:.6f}"',
            f'group="{TERRAIN_GEOM_GROUP}"',
            f'material="{self.material}"',
        ]
        if any(abs(e) > 1e-12 for e in self.euler):
            ex, ey, ez = self.euler
            attrs.append(f'euler="{ex:.6f} {ey:.6f} {ez:.6f}"')
        return f"{indent}<geom {' '.join(attrs)}/>"

    @classmethod
    def from_top_face(
        cls,
        name: str,
        center_xy: tuple[float, float],
        extent_xy: tuple[float, float],
        top_z: float,
        thickness: float = 0.0,
        material: str = "plywood",
    ) -> "Box":
        """
        Build a box from the quantity that actually matters for locomotion -- the
        height of its walkable top face -- rather than from its center.

        A box whose top face sits at `top_z` is extended down to (and below) z=0 by
        default so it is never floating and never leaves a lip the swing foot can
        catch on.
        """
        half_z = 0.5 * (top_z + (thickness if thickness > 0.0 else 0.10))
        cx, cy = center_xy
        ex, ey = extent_xy
        return cls(
            name=name,
            size=(0.5 * ex, 0.5 * ey, half_z),
            pos=(cx, cy, top_z - half_z),
            material=material,
        )


@dataclass
class TerrainScene:
    """A named collection of terrain boxes plus the robot spawn point."""
    name: str
    description: str
    boxes: list[Box] = field(default_factory=list)
    spawn_xy: tuple[float, float] = (0.0, 0.0)
    # perception can be scored against this, and the benchmark suite uses it to
    # measure progress along the course rather than raw x distance.
    route: list[str] = field(default_factory=list)

    def top_face_z(self, name: str) -> float:
        """Height of a named element's top face. Only valid for unrotated boxes."""
        for b in self.boxes:
            if b.name == name:
                if any(abs(e) > 1e-12 for e in b.euler):
                    raise ValueError(f"{name} is rotated; top face is not a single z")
                return b.pos[2] + b.size[2]
        raise KeyError(name)

    def to_xml(self) -> str:
        geoms = "\n".join(b.to_xml() for b in self.boxes)
        return f"""<mujoco model="bheema_parkour_{self.name}">
  <!-- GENERATED by bheema/terrain.py -- do not edit by hand.
       {self.description}
       Terrain geoms are group {TERRAIN_GEOM_GROUP} (ray mask); robot geoms are 2 and 3. -->
  <include file="{ROBOT_XML}"/>
  <!-- Must follow the include: meshdir resolves against this file's directory. -->
  <compiler angle="radian" meshdir="{MESHDIR}"/>

  <statistic center="2 0 0.6" extent="3"/>

  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.25 0.25 0.25" specular="0.4 0.4 0.4"/>
    <rgba haze="0.15 0.25 0.35 1"/>
    <!-- offwidth/offheight size the offscreen framebuffer used for depth rendering -->
    <global azimuth="140" elevation="-20" offwidth="1280" offheight="720"/>
    <map znear="0.01"/>
  </visual>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0"
      width="512" height="3072"/>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge"
      rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3" markrgb="0.8 0.8 0.8"
      width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true"
      texrepeat="5 5" reflectance="0.2"/>
    <material name="plywood" rgba="0.78 0.63 0.42 1"/>
    <material name="plywood_dark" rgba="0.55 0.42 0.28 1"/>
    <material name="concrete" rgba="0.62 0.62 0.60 1"/>
    <material name="platform" rgba="0.22 0.22 0.24 1"/>
    <material name="mat" rgba="0.20 0.38 0.62 1"/>
  </asset>

  <worldbody>
    <geom name="floor" size="0 0 0.05" type="plane" material="groundplane"
      group="{TERRAIN_GEOM_GROUP}" friction="1.5 0.1 0.1"/>
    <site name="target_com" type="sphere" size="0.05" rgba="0 1 0 0.5"/>
    <site name="target_foot_l" type="sphere" size="0.03" rgba="1 0 0 0.5"/>
    <site name="target_foot_r" type="sphere" size="0.03" rgba="0 0 1 0.5"/>

{geoms}
  </worldbody>
</mujoco>
"""

    def write(self, out_dir: Path | None = None) -> Path:
        out_dir = SCENES_DIR if out_dir is None else Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"scene_{self.name}.xml"
        path.write_text(self.to_xml())
        return path


# --------------------------------------------------------------------------------
# Scene builders
#
# Each takes a difficulty in [0, 1] and returns a TerrainScene. Element heights
# interpolate from a gentle, walkable setting to one requiring flight phases.
# --------------------------------------------------------------------------------

def flat(t: float = 0.0) -> TerrainScene:
    """Control scene. No boxes at all -- the flat-walk regression baseline."""
    return TerrainScene(
        name="flat",
        description="Control: bare floor plane, no terrain elements.",
        boxes=[],
        route=["floor"],
    )


def stairs(t: float = 0.0, n_steps: int = 6) -> TerrainScene:
    """Ascending flight, then a landing, then descending. Tests step-up and step-down."""
    rise = lerp(0.10, 0.20, t)
    run = lerp(0.42, 0.30, t)      # shorter tread as it gets harder
    width = lerp(1.20, 0.70, t)
    x0 = 1.20

    boxes, route = [], []
    for i in range(n_steps):
        name = f"step_up_{i}"
        boxes.append(Box.from_top_face(
            name, (x0 + run * (i + 0.5), 0.0), (run, width),
            top_z=rise * (i + 1), material="plywood" if i % 2 else "plywood_dark",
        ))
        route.append(name)

    landing_len = 0.90
    x_landing = x0 + run * n_steps + 0.5 * landing_len
    boxes.append(Box.from_top_face(
        "landing", (x_landing, 0.0), (landing_len, width),
        top_z=rise * n_steps, material="concrete",
    ))
    route.append("landing")

    x_down = x0 + run * n_steps + landing_len
    for i in range(n_steps - 1):
        # The final tread is the floor itself, so there is no box for it.
        name = f"step_down_{i}"
        boxes.append(Box.from_top_face(
            name, (x_down + run * (i + 0.5), 0.0), (run, width),
            top_z=rise * (n_steps - i - 1),
            material="plywood" if i % 2 else "plywood_dark",
        ))
        route.append(name)

    return TerrainScene(
        name="stairs",
        description=f"{n_steps} steps up at {rise*100:.0f}cm rise / {run*100:.0f}cm run, "
                    f"landing, then {n_steps} down.",
        boxes=[b for b in boxes if b.size[2] > 1e-6],
        route=route,
    )


def ramp(t: float = 0.0) -> TerrainScene:
    """
    Inclined plywood: up-slope, flat top, down-slope.

    This scene exists specifically to exercise the terrain-aware friction pyramid and
    CoP limits. On axis-aligned boxes every contact normal is exactly +z and that code
    path never runs a non-trivial case.
    """
    angle = np.deg2rad(lerp(8.0, 20.0, t))
    slope_len = 1.30
    width = lerp(1.20, 0.80, t)
    thickness = 0.08
    top_z = slope_len * np.sin(angle)
    run = slope_len * np.cos(angle)
    x0 = 1.20

    # A rotated box's top face passes through its center offset along the face normal.
    # Rotating about -y tilts the +x end upward.
    def slope(name: str, x_start: float, sign: float) -> Box:
        cx = x_start + 0.5 * run
        cz = 0.5 * top_z - 0.5 * thickness * np.cos(angle)
        return Box(
            name=name,
            size=(0.5 * slope_len, 0.5 * width, 0.5 * thickness),
            pos=(cx, 0.0, cz),
            euler=(0.0, -sign * angle, 0.0),
            material="plywood",
        )

    flat_len = 0.80
    boxes = [
        slope("ramp_up", x0, +1.0),
        Box.from_top_face("ramp_top", (x0 + run + 0.5 * flat_len, 0.0),
                          (flat_len, width), top_z=top_z, material="plywood_dark"),
        slope("ramp_down", x0 + run + flat_len, -1.0),
    ]
    return TerrainScene(
        name="ramp",
        description=f"{np.rad2deg(angle):.0f}deg ramp up, {flat_len*100:.0f}cm flat top, "
                    f"ramp down. Peak {top_z*100:.0f}cm.",
        boxes=boxes,
        route=["ramp_up", "ramp_top", "ramp_down"],
    )


def platform(t: float = 0.0) -> TerrainScene:
    """Single raised platform: one step-up, traverse, one step-down."""
    height = lerp(0.15, 0.45, t)
    length = lerp(1.60, 1.20, t)
    width = lerp(1.20, 0.80, t)
    return TerrainScene(
        name="platform",
        description=f"Single {height*100:.0f}cm platform, {length:.1f}m long.",
        boxes=[Box.from_top_face("platform", (1.60 + 0.5 * length, 0.0),
                                 (length, width), top_z=height, material="plywood")],
        route=["platform"],
    )


def gap_field(t: float = 0.0, n_gaps: int = 4) -> TerrainScene:
    """Level pads separated by gaps. Tests stride adaptation, then real jumps."""
    height = lerp(0.12, 0.35, t)
    gap = lerp(0.15, 0.45, t)
    pad = lerp(0.70, 0.50, t)
    width = lerp(1.10, 0.70, t)

    boxes, route = [], []
    x = 1.20
    for i in range(n_gaps + 1):
        name = f"pad_{i}"
        boxes.append(Box.from_top_face(
            name, (x + 0.5 * pad, 0.0), (pad, width),
            top_z=height, material="plywood" if i % 2 else "concrete",
        ))
        route.append(name)
        x += pad + gap
    return TerrainScene(
        name="gap_field",
        description=f"{n_gaps} gaps of {gap*100:.0f}cm between {pad*100:.0f}cm pads "
                    f"at {height*100:.0f}cm.",
        boxes=boxes,
        route=route,
    )


def stepping_stones(t: float = 0.0, n_stones: int = 6) -> TerrainScene:
    """
    Discrete alternating stones. Impossible without perception-driven foothold
    selection.
    """
    height = lerp(0.12, 0.30, t)
    stone = lerp(0.30, 0.20, t)          # square stone side length
    # Stride must exceed the stone side at every difficulty, or the stones abut and
    # the scene degenerates into a continuous path with nothing to plan around.
    stride = lerp(0.38, 0.46, t)
    sway = lerp(0.13, 0.19, t)           # lateral offset, alternating

    boxes, route = [], []
    for i in range(n_stones):
        name = f"stone_{i}"
        y = sway if i % 2 == 0 else -sway
        boxes.append(Box.from_top_face(
            name, (1.20 + stride * i, y), (stone, stone),
            top_z=height, material="concrete" if i % 2 else "plywood",
        ))
        route.append(name)
    return TerrainScene(
        name="stepping_stones",
        description=f"{n_stones} {stone*100:.0f}cm stones at {stride*100:.0f}cm stride "
                    f"({(stride-stone)*100:.0f}cm gap), +/-{sway*100:.0f}cm sway, "
                    f"{height*100:.0f}cm high.",
        boxes=boxes,
        route=route,
    )


def beam(t: float = 0.0) -> TerrainScene:
    """Narrow balance beam. Squeezes the lateral CoP margin toward zero."""
    height = lerp(0.15, 0.35, t)
    bw = lerp(0.34, 0.18, t)
    length = lerp(1.80, 2.40, t)
    approach = 0.60
    boxes = [
        Box.from_top_face("beam_entry", (1.20 + 0.5 * approach, 0.0),
                          (approach, 0.90), top_z=height, material="concrete"),
        Box.from_top_face("beam", (1.20 + approach + 0.5 * length, 0.0),
                          (length, bw), top_z=height, material="plywood_dark"),
        Box.from_top_face("beam_exit", (1.20 + approach + length + 0.5 * approach, 0.0),
                          (approach, 0.90), top_z=height, material="concrete"),
    ]
    return TerrainScene(
        name="beam",
        description=f"{bw*100:.0f}cm wide beam, {length:.1f}m long, at {height*100:.0f}cm. "
                    f"Foot half-width is 5cm (FOOT_LY).",
        boxes=boxes,
        route=["beam_entry", "beam", "beam_exit"],
    )


def atlas_gym(t: float = 0.0) -> TerrainScene:
    """
    The Boston Dynamics parkour room, laid out as a traversable course along +x.

    Reading the reference footage left-to-right: an approach mat, angled plywood,
    stacked plywood boxes, the long dark raised platform with a plank laid on it, and
    a low concrete block on the far side.

    Sequenced so each element demands something different: ramp (non-vertical contact
    normal) -> two step-ups -> gap jump at height -> narrow plank -> step down. The
    gap is between box_b and the platform, so it is a genuine at-height gap rather
    than a disguised step-up.

    Heights are G1-scaled, not Atlas-scaled -- see NOMINAL_COM_HEIGHT.
    """
    h_a = lerp(0.15, 0.40, t)        # first plywood box
    h_b = lerp(0.25, 0.55, t)        # taller box beside it
    h_plat = lerp(0.20, 0.50, t)     # long raised platform, just below box_b
    h_block = lerp(0.15, 0.40, t)    # concrete block on the far side
    gap = lerp(0.15, 0.45, t)        # box_b -> platform gap
    plank_w = lerp(0.34, 0.20, t)
    ramp_deg = lerp(8.0, 18.0, t)

    boxes, route = [], []

    # Approach mat -- top face FLUSH with the floor (z=0), not raised. A raised mat is a
    # 2cm lip 20cm from spawn: too small for the controller to treat as a step, too big to
    # ignore, and it sits inside the approach distance the gait needs to stabilise. It made
    # the robot trip and enter a limit cycle before reaching any real obstacle.
    # Being coplanar with the floor is also the correct thing for perception -- a patch at
    # the same height *should* merge with the ground plane.
    boxes.append(Box(name="mat", size=(0.90, 0.80, 0.010),
                     pos=(1.10, 0.0, -0.010), material="mat"))
    route.append("mat")

    # Angled plywood rising to box_a. Rotating a box about +y by -angle tilts its top
    # face normal toward -x, i.e. ascending in +x.
    angle = np.deg2rad(ramp_deg)
    slope_len = h_a / np.sin(angle)
    run = slope_len * np.cos(angle)
    thickness = 0.06
    x = 2.10
    boxes.append(Box(
        name="ramp_up",
        size=(0.5 * slope_len, 0.45, 0.5 * thickness),
        pos=(x + 0.5 * run, 0.0, 0.5 * h_a - 0.5 * thickness * np.cos(angle)),
        euler=(0.0, -angle, 0.0),
        material="plywood",
    ))
    route.append("ramp_up")
    x += run

    # Stacked plywood boxes: two step-ups off the top of the ramp.
    boxes.append(Box.from_top_face("box_a", (x + 0.40, 0.0), (0.80, 0.90),
                                   top_z=h_a, material="plywood"))
    route.append("box_a")
    x += 0.80
    boxes.append(Box.from_top_face("box_b", (x + 0.35, 0.0), (0.70, 0.90),
                                   top_z=h_b, material="plywood_dark"))
    route.append("box_b")
    x += 0.70 + gap

    # Long dark raised platform ("the table"), sitting just below box_b so crossing
    # the gap is a small step down as well as a reach.
    plat_len = 2.40
    boxes.append(Box.from_top_face("platform", (x + 0.5 * plat_len, 0.0),
                                   (plat_len, 0.95), top_z=h_plat, material="platform"))
    route.append("platform")
    # The plank rests on the platform, so it is a genuinely thin box rather than one
    # extended down to the floor. It narrows the far half of the traverse.
    boxes.append(Box(name="plank", size=(0.50, 0.5 * plank_w, 0.020),
                     pos=(x + 1.70, 0.0, h_plat + 0.020), material="plywood"))
    route.append("plank")
    x += plat_len + gap

    # Concrete block on the far side: step down and off onto the floor.
    boxes.append(Box.from_top_face("block", (x + 0.35, 0.0), (0.70, 0.80),
                                   top_z=h_block, material="concrete"))
    route.append("block")

    return TerrainScene(
        name="atlas_gym",
        description=(
            f"Atlas parkour room as a course: mat -> {ramp_deg:.0f}deg ramp -> "
            f"{h_a*100:.0f}cm box -> {h_b*100:.0f}cm box -> {gap*100:.0f}cm gap -> "
            f"{h_plat*100:.0f}cm platform + {plank_w*100:.0f}cm plank -> "
            f"{gap*100:.0f}cm gap -> {h_block*100:.0f}cm block."
        ),
        boxes=boxes,
        route=route,
    )


SCENES = {
    "flat": flat,
    "stairs": stairs,
    "ramp": ramp,
    "platform": platform,
    "gap_field": gap_field,
    "stepping_stones": stepping_stones,
    "beam": beam,
    "atlas_gym": atlas_gym,
}


def build(name: str, difficulty: str | float = "easy") -> TerrainScene:
    """Build a scene by name. `difficulty` is a LEVELS key or a raw float in [0, 1]."""
    if name not in SCENES:
        raise KeyError(f"unknown scene {name!r}; choose from {sorted(SCENES)}")
    t = LEVELS[difficulty] if isinstance(difficulty, str) else float(difficulty)
    if not 0.0 <= t <= 1.0:
        raise ValueError(f"difficulty must be in [0, 1], got {t}")
    scene = SCENES[name](t)
    # Tag the difficulty into the filename so benchmark runs are self-describing.
    label = difficulty if isinstance(difficulty, str) else f"t{t:.2f}"
    scene.name = f"{scene.name}_{label}"
    return scene


def write_all(difficulty: str | float = "easy", out_dir: Path | None = None) -> list[Path]:
    return [build(n, difficulty).write(out_dir) for n in SCENES]


# --------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--scene", default="atlas_gym", choices=sorted(SCENES) + ["all"])
    ap.add_argument("--difficulty", default="easy",
                    help="easy | medium | atlas, or a float in [0, 1]")
    ap.add_argument("--out-dir", default=None, type=Path)
    ap.add_argument("--view", action="store_true", help="open the scene in the viewer")
    ap.add_argument("--list", action="store_true", help="describe every scene and exit")
    args = ap.parse_args()

    try:
        difficulty: str | float = float(args.difficulty)
    except ValueError:
        difficulty = args.difficulty

    if args.list:
        for name in SCENES:
            for lvl in LEVELS:
                s = build(name, lvl)
                print(f"{name:<16} {lvl:<7} {len(s.boxes):>2} boxes  {s.description}")
            print()
        return

    if args.scene == "all":
        for p in write_all(difficulty, args.out_dir):
            print(f"wrote {p}")
        return

    scene = build(args.scene, difficulty)
    path = scene.write(args.out_dir)
    print(f"wrote {path}\n{scene.description}")

    if args.view:
        import mujoco as mj
        import mujoco.viewer as mjv
        model = mj.MjModel.from_xml_path(str(path))
        data = mj.MjData(model)
        mj.mj_forward(model, data)
        mjv.launch(model, data)


if __name__ == "__main__":
    main()
