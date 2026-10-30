"""Convert a CMU mocap BVH into an Animation clip of wire joint angles, the inverse of the renderers' bone map."""

from __future__ import annotations

import argparse
import json
import math
import sys
import xml.etree.ElementTree as ET
from collections.abc import Mapping, Sequence
from pathlib import Path

import attrs
import numpy as np

from .profile import JOINT_NAMES

DEFAULT_FPS = 20.0
DEFAULT_BONES = ("LeftUpLeg", "LeftLeg", "RightUpLeg", "RightLeg", "LowerBack", "Spine", "Spine1")
FEET = {"l": "LeftFoot", "r": "RightFoot"}
SPLAY_JOINTS = ("l_y_hip", "l_p_hip", "r_y_hip", "r_p_hip")
ROOT = "Hips"
DAE_TO_BVH = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])  # human assets stand Z up facing +x, BVH is Y up facing +z
SOLVER_ITERATIONS = 12
SOLVER_STEP = 1e-5
_EXAMPLE = "python3 -m task_generator.simulators.human.bvh 22_14.bvh --skin <human asset>/meshes/<name>.dae -o arena_simulation_setup/assets/Common/Animation/squat --from 0.72 --to 1.75"


@attrs.frozen
class Joint:
    """A BVH joint: its offset from its parent at rest, its channels in file order and the names of its children."""

    name: str
    parent: str | None
    offset: tuple[float, float, float]
    channels: tuple[str, ...]
    children: tuple[str, ...]


@attrs.frozen
class BindPose:
    """A human asset's skeleton at bind, in BVH axes: each joint's orientation and position."""

    orientations: dict[str, np.ndarray]
    positions: dict[str, np.ndarray]


@attrs.frozen
class Motion:
    """A parsed BVH: joints by name in file order, frame time and the per-frame channel values."""

    joints: dict[str, Joint]
    frame_time: float
    values: np.ndarray


@attrs.frozen
class BoneTarget:
    """One wire joint's share of one bone's local rotation, as in bone_map.json."""

    joint: str
    axis: tuple[float, float, float]
    gain: float


def parse(text: str) -> Motion:
    """Parse the HIERARCHY and MOTION sections of a BVH file."""
    tokens = text.split()
    pos = 0
    joints: dict[str, Joint] = {}
    order: list[str] = []
    children: dict[str, list[str]] = {}
    channels: dict[str, tuple[str, ...]] = {}
    offsets: dict[str, tuple[float, float, float]] = {}
    parents: dict[str, str | None] = {}
    stack: list[str] = []
    current: str | None = None
    while tokens[pos] != "MOTION":
        tok = tokens[pos]
        if tok in ("ROOT", "JOINT"):
            current = tokens[pos + 1]
            parents[current] = stack[-1] if stack else None
            if stack:
                children[stack[-1]].append(current)
            children[current] = []
            order.append(current)
            pos += 2
        elif tok == "End":
            current = None
            pos += 2
        elif tok == "{":
            stack.append(current if current is not None else "")
            pos += 1
        elif tok == "}":
            stack.pop()
            pos += 1
        elif tok == "OFFSET":
            if current is not None and current not in offsets:
                offsets[current] = (float(tokens[pos + 1]), float(tokens[pos + 2]), float(tokens[pos + 3]))
            pos += 4
        elif tok == "CHANNELS":
            n = int(tokens[pos + 1])
            assert current is not None, "CHANNELS outside a joint"
            channels[current] = tuple(tokens[pos + 2 : pos + 2 + n])
            pos += 2 + n
        else:
            pos += 1
    frames = int(tokens[pos + 2])
    frame_time = float(tokens[pos + 5])
    values = np.array(tokens[pos + 6 :], dtype=float)
    width = sum(len(channels.get(name, ())) for name in order)
    if values.size != frames * width:
        raise ValueError(f"MOTION holds {values.size} values, {frames} frames of {width} channels need {frames * width}")
    for name in order:
        joints[name] = Joint(name=name, parent=parents[name], offset=offsets.get(name, (0.0, 0.0, 0.0)), channels=channels.get(name, ()), children=tuple(c for c in children[name] if c))
    return Motion(joints=joints, frame_time=frame_time, values=values.reshape(frames, width))


def _axis_rotation(axis: Sequence[float] | np.ndarray, angle: float) -> np.ndarray:
    x, y, z = np.asarray(axis, dtype=float) / np.linalg.norm(axis)
    c, s = math.cos(angle), math.sin(angle)
    k = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return np.eye(3) + s * k + (1.0 - c) * (k @ k)


_CHANNEL_AXES = {"Xrotation": (1.0, 0.0, 0.0), "Yrotation": (0.0, 1.0, 0.0), "Zrotation": (0.0, 0.0, 1.0)}


def local_rotations(motion: Motion) -> dict[str, np.ndarray]:
    """Per-joint local rotation matrices (frames, 3, 3), the rotation channels composed in file order."""
    out: dict[str, np.ndarray] = {}
    col = 0
    n = motion.values.shape[0]
    for name, joint in motion.joints.items():
        rot = np.repeat(np.eye(3)[None], n, axis=0)
        for k, channel in enumerate(joint.channels):
            if channel in _CHANNEL_AXES:
                angles = np.radians(motion.values[:, col + k])
                rot = rot @ np.stack([_axis_rotation(_CHANNEL_AXES[channel], a) for a in angles])
        col += len(joint.channels)
        out[name] = rot
    return out


def _heading(rot: np.ndarray) -> np.ndarray:
    """Rotation about the vertical (BVH +y) that turns +z onto the horizontal part of rot's +z."""
    forward = rot[:, 2]
    yaw = math.atan2(forward[0], forward[2])
    return _axis_rotation((0.0, 1.0, 0.0), yaw)


def upright_worlds(motion: Motion, rotations: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Per-joint world rotations (frames, 3, 3) with the root's heading taken out, its tilt kept."""
    worlds: dict[str, np.ndarray] = {}
    for name, joint in motion.joints.items():
        if joint.parent is None:
            worlds[name] = np.stack([_heading(r).T @ r for r in rotations[name]])
        else:
            worlds[name] = worlds[joint.parent] @ rotations[name]
    return worlds


def _align(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """The smallest rotation turning direction source onto direction target, none when either has no length."""
    if np.linalg.norm(source) < 1e-9 or np.linalg.norm(target) < 1e-9:
        return np.eye(3)
    a, b = source / np.linalg.norm(source), target / np.linalg.norm(target)
    axis = np.cross(a, b)
    s = float(np.linalg.norm(axis))
    if s < 1e-9:
        return np.eye(3)
    return _axis_rotation(axis, math.atan2(s, float(np.dot(a, b))))


def _hang_legs(bind: BindPose) -> BindPose:
    """The pose with each thigh and shank swung to hang straight down from its hip and each foot keeping its orientation, the wire's leg zero."""
    orientations, positions = dict(bind.orientations), dict(bind.positions)
    down = np.array([0.0, -1.0, 0.0])
    for side in ("Left", "Right"):
        names = [f"{side}{bone}" for bone in ("UpLeg", "Leg", "Foot")]
        if not all(name in positions for name in names):
            continue
        turn = np.eye(3)
        for bone, child in zip(names, names[1:], strict=False):
            reach = turn @ (bind.positions[child] - bind.positions[bone])
            turn = _align(reach, down) @ turn
            orientations[bone] = turn @ bind.orientations[bone]
            positions[child] = positions[bone] + float(np.linalg.norm(reach)) * down
    return BindPose(orientations=orientations, positions=positions)


def bind_pose(dae: Path) -> BindPose:
    """The skin joints' bind orientations and positions in BVH axes, read off the inverse bind matrices of a human asset's COLLADA, legs hung straight down."""
    root = ET.parse(dae).getroot()
    ns = root.tag[: root.tag.index("}") + 1]
    by_id = {element.get("id"): element for element in root.iter() if element.get("id")}
    orientations: dict[str, np.ndarray] = {}
    positions: dict[str, np.ndarray] = {}
    for skin in root.iter(f"{ns}skin"):
        inputs = {element.get("semantic"): (element.get("source") or "").lstrip("#") for element in skin.iterfind(f"{ns}joints/{ns}input")}
        names = (by_id[inputs["JOINT"]].findtext(f"{ns}Name_array") or "").split()
        inverse_binds = np.array((by_id[inputs["INV_BIND_MATRIX"]].findtext(f"{ns}float_array") or "").split(), dtype=float).reshape(-1, 4, 4)
        for name, inverse_bind in zip(names, inverse_binds, strict=True):
            bind = np.linalg.inv(inverse_bind)
            u, _, vt = np.linalg.svd(DAE_TO_BVH @ bind[:3, :3])
            orientations.setdefault(name, u @ vt)
            positions.setdefault(name, DAE_TO_BVH @ bind[:3, 3])
    if not orientations:
        raise ValueError(f"{dae}: no skin controller")
    return _hang_legs(BindPose(orientations=orientations, positions=positions))


def load_bone_map(path: Path) -> dict[str, list[BoneTarget]]:
    """bone_map.json as wire targets grouped by bone, in the order the renderers compose them."""
    by_bone: dict[str, list[BoneTarget]] = {}
    for joint, targets in json.loads(path.read_text()).items():
        for target in targets:
            by_bone.setdefault(target["bone"], []).append(BoneTarget(joint=joint, axis=tuple(target["axis"]), gain=float(target["sign"]) * float(target["scale"])))
    return by_bone


def _compose(targets: Sequence[BoneTarget], angles: np.ndarray) -> np.ndarray:
    rot = np.eye(3)
    for target, angle in zip(targets, angles, strict=True):
        rot = rot @ _axis_rotation(target.axis, target.gain * angle)
    return rot


def _log(rot: np.ndarray) -> np.ndarray:
    """Rotation vector (axis times angle) of a rotation matrix."""
    skew = 0.5 * np.array([rot[2, 1] - rot[1, 2], rot[0, 2] - rot[2, 0], rot[1, 0] - rot[0, 1]])
    angle = math.atan2(float(np.linalg.norm(skew)), 0.5 * (float(np.trace(rot)) - 1.0))
    norm = float(np.linalg.norm(skew))
    if norm < 1e-12:
        if angle < 1e-6:
            return np.zeros(3)
        axis = np.sqrt(np.clip((np.diag(rot) + 1.0) / 2.0, 0.0, None))
        return angle * axis
    return angle * skew / norm


def _residual(targets: Sequence[BoneTarget], angles: np.ndarray, goal: np.ndarray) -> np.ndarray:
    return _log(goal.T @ _compose(targets, angles))


def decompose(targets: Sequence[BoneTarget], goal: np.ndarray, start: np.ndarray) -> np.ndarray:
    """Angles about the targets' axes whose composition comes closest to goal, by Gauss-Newton from start."""
    angles = np.array(start, dtype=float)
    for _ in range(SOLVER_ITERATIONS):
        r = _residual(targets, angles, goal)
        jac = np.empty((3, len(targets)))
        for k in range(len(targets)):
            step = np.zeros(len(targets))
            step[k] = SOLVER_STEP
            jac[:, k] = (_residual(targets, angles + step, goal) - r) / SOLVER_STEP
        delta, *_ = np.linalg.lstsq(jac, -r, rcond=None)
        angles += delta
        if float(np.abs(delta).max()) < 1e-9:
            break
    return angles


def bone_targets(motion: Motion, bind: BindPose, bones: Sequence[str]) -> dict[str, np.ndarray]:
    """Per bone, the world orientation (frames, 3, 3) the asset's bone takes to point where the mocap bone points."""
    worlds = upright_worlds(motion, local_rotations(motion))
    out: dict[str, np.ndarray] = {}
    for bone in bones:
        child = next((c for c in motion.joints[bone].children if c in bind.positions), None)
        align = np.eye(3) if child is None else _align(bind.positions[child] - bind.positions[bone], np.array(motion.joints[child].offset))
        out[bone] = worlds[bone] @ align @ bind.orientations[bone]
    return out


def wire_frames(motion: Motion, bone_map: Mapping[str, Sequence[BoneTarget]], bind: BindPose, bones: Sequence[str] = DEFAULT_BONES, fps: float = DEFAULT_FPS, start_s: float = 0.0, stop_s: float | None = None, parallel: bool = False) -> list[dict]:
    """Clip frames ({"t", "angles"} over every wire joint) resampled to fps, the given bones solved and every other joint at 0.

    parallel holds the hips' yaw and abduction at 0. A foot left out keeps its sole flat: its ankle is -(hip + knee).
    """
    targets = bone_targets(motion, bind, bones)
    n = motion.values.shape[0]
    duration = (n - 1) * motion.frame_time
    stop = duration if stop_s is None else min(stop_s, duration)
    times = np.arange(start_s, stop + 1e-9, 1.0 / fps)
    frames = [{"t": float(t - start_s), "angles": dict.fromkeys(JOINT_NAMES, 0.0)} for t in times]
    for bone in bones:
        parent = motion.joints[bone].parent
        assert parent is not None, f"{bone} is the root"
        parent_bind = bind.orientations[parent]
        previous = np.zeros(len(bone_map[bone]))
        for frame, t in zip(frames, times, strict=True):
            k = min(int(round(t / motion.frame_time)), n - 1)
            parent_world = targets[parent][k] if parent in targets else parent_bind
            goal = bind.orientations[bone].T @ parent_bind @ parent_world.T @ targets[bone][k]
            previous = decompose(bone_map[bone], goal, previous)
            for target, angle in zip(bone_map[bone], previous, strict=True):
                frame["angles"][target.joint] = float(angle)
    if parallel:
        for frame in frames:
            frame["angles"].update(dict.fromkeys(SPLAY_JOINTS, 0.0))
    for side, foot in FEET.items():
        if foot not in bones:
            for frame in frames:
                angles = frame["angles"]
                angles[f"{side}_ankle"] = -(angles[f"{side}_r_hip"] + angles[f"{side}_knee"])
    return frames


def root_height(motion: Motion) -> np.ndarray:
    """The root's vertical position per frame (BVH +y), in file units."""
    root = motion.joints[ROOT]
    return motion.values[:, list(root.channels).index("Yposition")]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python3 -m task_generator.simulators.human.bvh",
        description="Convert a CMU mocap BVH (the cgspeed conversion, the cmu_mb rig the human assets carry) into an Animation clip of wire joint angles. The root is held upright and its tilt folded into the thighs and the lower back. Joints of bones outside --bones stay at 0, the neutral stance.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"example:\n  {_EXAMPLE}\n\nThe clip lands as <OUT>/<OUT name>.npz plus annotation.yaml. Check it with `arena asset find animation <name>`, publish it with `arena asset push animation <name>`.",
    )
    parser.add_argument("bvh", type=Path, metavar="CLIP.bvh", help="CMU BVH in the cmu_mb skeleton (Hips, LHipJoint, LeftUpLeg, ...)")
    parser.add_argument("-o", "--output", type=Path, required=True, metavar="DIR", help="Animation asset directory to write, its name is the clip name")
    parser.add_argument("--skin", type=Path, required=True, metavar="HUMAN.dae", help="skinned mesh of any human asset (meshes/<name>.dae), whose bind frames the bone map's axes are given in")
    parser.add_argument("--bone-map", type=Path, metavar="bone_map.json", help="wire-to-bone map, default the one arena_people_msgs installs")
    parser.add_argument("--bones", default=",".join(DEFAULT_BONES), metavar="A,B,...", help=f"bones whose rotation is converted, default {','.join(DEFAULT_BONES)} (legs and spine, whose neutral stance is the bind pose; feet left out keep their soles flat)")
    parser.add_argument("--fps", type=float, default=DEFAULT_FPS, metavar="F", help=f"frame rate of the clip in Hz, default {DEFAULT_FPS}")
    parser.add_argument("--parallel", action="store_true", help=f"keep the legs parallel: {', '.join(SPLAY_JOINTS)} held at 0, only the sagittal leg motion taken from the mocap")
    parser.add_argument("--from", dest="start", type=float, default=0.0, metavar="S", help="first second of the BVH taken, default 0")
    parser.add_argument("--to", dest="stop", type=float, metavar="S", help="last second of the BVH taken, default its end")
    parser.add_argument("--desc", default="", help="annotation desc, default the BVH name")
    parser.add_argument("--note", default="", help="annotation note, e.g. the source and its attribution")
    return parser


def _default_bone_map() -> Path:
    from ament_index_python.packages import get_package_share_directory

    return Path(get_package_share_directory("arena_people_msgs")) / "bone_map.json"


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    from arena_simulation_setup.tree.assets.Animation import write_clip

    motion = parse(args.bvh.read_text())
    bone_map = load_bone_map(args.bone_map or _default_bone_map())
    bones = [b for b in args.bones.split(",") if b]
    unknown = [b for b in bones if b not in bone_map or b not in motion.joints]
    if unknown:
        print(f"bones {unknown} are not in both the bone map and the BVH, pick from {sorted(set(bone_map) & set(motion.joints))}", file=sys.stderr)
        return 2
    frames = wire_frames(motion, bone_map, bind_pose(args.skin), bones, args.fps, args.start, args.stop, args.parallel)
    meta = {"desc": args.desc or args.bvh.stem, "fps": args.fps, "loop": False, "note": args.note or f"converted from {args.bvh.name}"}
    print(write_clip(args.output, frames, meta))
    return 0


if __name__ == "__main__":
    sys.exit(main())
