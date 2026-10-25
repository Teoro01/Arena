"""Articulated objects: the links of an object's meshes, the joints between them, their named states and the link poses at given joint values."""

from __future__ import annotations

import enum
import itertools
import math
import re
import typing
from collections.abc import Iterable, Mapping, Sequence

import attrs
import numpy as np

from arena_simulation_setup.utils.geometry import Orientation, Pose, Position

Side = tuple[float, float]
Box = tuple[Side, Side, Side]
Values = dict[str, float]

NAME = re.compile(r"^[a-z0-9_]+$")
STATE_TOLERANCE = 1e-3
SWEEP_SAMPLES = 4096
SWEEP_STEPS = 9
DEFAULT_ANGULAR_VELOCITY = 1.0
DEFAULT_LINEAR_VELOCITY = 0.25
RESERVED_STATES = frozenset({"state", "target", "moving", "blocked"})


class JointType(enum.StrEnum):
    REVOLUTE = "revolute"
    PRISMATIC = "prismatic"
    CONTINUOUS = "continuous"
    FIXED = "fixed"


def _name(kind: str, value: object) -> str:
    if not isinstance(value, str) or not NAME.match(value):
        raise ValueError(f"{kind} name must match [a-z0-9_]+, got {value!r}")
    return value


def _vector(value: object) -> Position:
    if isinstance(value, Position):
        return value
    if isinstance(value, str) or not isinstance(value, Sequence) or len(value) != 3:
        raise ValueError(f"expected [x, y, z], got {value!r}")
    return Position(*(float(component) for component in value))


def _axis(value: object) -> Position:
    axis = _vector(value)
    if axis.norm() < 1e-9:
        raise ValueError(f"joint axis must not be zero, got {list(axis)}")
    return axis.normalized()


def _limit(value: object) -> Side | None:
    if value is None:
        return None
    if isinstance(value, str) or not isinstance(value, Sequence) or len(value) != 2:
        raise ValueError(f"joint limit must be [lower, upper], got {value!r}")
    lower, upper = (float(bound) for bound in value)
    if not (math.isfinite(lower) and math.isfinite(upper)) or lower > upper:
        raise ValueError(f"joint limit needs finite lower <= upper, got {value!r}")
    return (lower, upper)


def rotation(axis: Position, angle: float) -> Orientation:
    """Rotation by angle (radians) about a unit axis."""
    half = angle / 2.0
    s = math.sin(half)
    return Orientation(w=math.cos(half), x=axis.x * s, y=axis.y * s, z=axis.z * s)


def matrix(pose: Pose) -> np.ndarray:
    """4x4 homogeneous transform of a pose."""
    w, x, y, z = pose.orientation
    result = np.eye(4)
    result[:3, :3] = [
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ]
    result[:3, 3] = list(pose.position)
    return result


@attrs.define(kw_only=True)
class Joint:
    """A joint moving its child link relative to its parent, origin and axis in the model frame at all joint values 0."""

    name: str = attrs.field(converter=lambda value: _name("joint", value))
    type: JointType = attrs.field(converter=JointType)
    parent: str = attrs.field(converter=lambda value: _name("link", value))
    child: str = attrs.field(converter=lambda value: _name("link", value))
    origin: Position = attrs.field(factory=lambda: Position(0.0, 0.0, 0.0), converter=_vector)
    axis: Position = attrs.field(factory=lambda: Position(0.0, 0.0, 1.0), converter=_axis)
    limit: Side | None = attrs.field(default=None, converter=_limit)
    velocity: float | None = attrs.field(default=None, converter=attrs.converters.optional(float))

    def __attrs_post_init__(self) -> None:
        if self.parent == self.child:
            raise ValueError(f"joint {self.name} connects link {self.child} to itself")
        if self.type in (JointType.REVOLUTE, JointType.PRISMATIC) and self.limit is None:
            raise ValueError(f"{self.type} joint {self.name} needs a limit: [lower, upper]")
        if self.type in (JointType.CONTINUOUS, JointType.FIXED) and self.limit is not None:
            raise ValueError(f"{self.type} joint {self.name} takes no limit")
        if self.velocity is not None and not self.velocity > 0.0:
            raise ValueError(f"joint {self.name} velocity must be > 0, got {self.velocity}")

    @property
    def movable(self) -> bool:
        return self.type is not JointType.FIXED

    @property
    def speed(self) -> float:
        """Kinematic speed in rad/s or m/s."""
        if self.velocity is not None:
            return self.velocity
        return DEFAULT_LINEAR_VELOCITY if self.type is JointType.PRISMATIC else DEFAULT_ANGULAR_VELOCITY

    @property
    def bounds(self) -> Side:
        """Reachable values, one full turn for a continuous joint and 0 for a fixed one."""
        if self.limit is not None:
            return self.limit
        if self.type is JointType.CONTINUOUS:
            return (-math.pi, math.pi)
        return (0.0, 0.0)

    def clamp(self, value: float) -> float:
        if self.limit is None:
            return math.remainder(value, 2 * math.pi) if self.type is JointType.CONTINUOUS else 0.0
        return min(max(value, self.limit[0]), self.limit[1])

    def motion(self, value: float) -> Pose:
        """Model-frame motion of the child at this joint value, relative to its parent."""
        if self.type in (JointType.REVOLUTE, JointType.CONTINUOUS):
            turn = rotation(self.axis, value)
            return Pose(position=self.origin - turn * self.origin, orientation=turn)
        if self.type is JointType.PRISMATIC:
            return Pose(position=self.axis * value, orientation=Orientation.identity())
        return Pose(position=Position(0.0, 0.0, 0.0), orientation=Orientation.identity())

    @classmethod
    def parse(cls, value: Joint | Mapping[str, typing.Any]) -> Joint:
        if isinstance(value, Joint):
            return value
        if not isinstance(value, Mapping):
            raise ValueError(f"joint must be a mapping, got {value!r}")
        fields = {field.name for field in attrs.fields(cls)}
        if unknown := sorted(set(value) - fields):
            raise ValueError(f"joint {value.get('name')!r} has unknown keys {unknown}, known: {sorted(fields)}")
        if missing := sorted({"name", "type", "parent", "child"} - set(value)):
            raise ValueError(f"joint {value.get('name')!r} is missing {missing}")
        try:
            return cls(**value)
        except ValueError as e:
            raise ValueError(f"joint {value.get('name')!r}: {e}") from None

    def unstructure(self) -> dict[str, object]:
        result: dict[str, object] = {
            "name": self.name,
            "type": str(self.type),
            "parent": self.parent,
            "child": self.child,
            "origin": [round(component, 6) + 0.0 for component in self.origin],
            "axis": [round(component, 6) + 0.0 for component in self.axis],
        }
        if self.limit is not None:
            result["limit"] = [round(bound, 6) + 0.0 for bound in self.limit]
        if self.velocity is not None:
            result["velocity"] = self.velocity
        return result


def _links(value: object) -> dict[str, list[str]]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError(f"links must map each link name to its source node names, got {value!r}")
    links: dict[str, list[str]] = {}
    for link, nodes in value.items():
        if nodes is not None and (isinstance(nodes, str) or not isinstance(nodes, Iterable)):
            raise ValueError(f"link {link!r} takes a list of source node names, got {nodes!r}")
        links[_name("link", link)] = [str(node) for node in nodes or ()]
    return links


def _joints(value: object) -> list[Joint]:
    if not isinstance(value, Sequence) or isinstance(value, str):
        raise ValueError(f"joints must be a list, got {value!r}")
    return [Joint.parse(joint) for joint in value]


def _states(value: object) -> dict[str, Values]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"states must map each state name to joint values, got {value!r}")
    states: dict[str, Values] = {}
    for state, values in value.items():
        if not isinstance(values, Mapping) or not values:
            raise ValueError(f"state {state!r} must map joint names to values, got {values!r}")
        states[_name("state", state)] = {_name("joint", joint): float(position) for joint, position in values.items()}
    return states


def _box(value: object) -> Box | None:
    if value is None:
        return None
    sides = [tuple(float(bound) for bound in side) for side in typing.cast("Iterable[Iterable[float]]", value)]
    if len(sides) != 3 or any(len(side) != 2 or side[0] > side[1] for side in sides):
        raise ValueError(f"swept must be [[x_min, x_max], [y_min, y_max], [z_min, z_max]], got {value!r}")
    return typing.cast(Box, tuple(sides))


@attrs.define(kw_only=True)
class Articulation:
    """Links, joints and named states of an articulated object, in the model frame at all joint values 0."""

    links: dict[str, list[str]] = attrs.field(converter=_links)
    joints: list[Joint] = attrs.field(converter=_joints)
    states: dict[str, Values] = attrs.field(factory=dict, converter=_states)
    swept: Box | None = attrs.field(default=None, converter=_box)

    def __attrs_post_init__(self) -> None:
        names = [joint.name for joint in self.joints]
        if duplicates := sorted({name for name in names if names.count(name) > 1}):
            raise ValueError(f"joint names must be unique, duplicated: {', '.join(duplicates)}")
        if not any(joint.movable for joint in self.joints):
            raise ValueError("an articulation needs at least one joint that is not fixed")
        for joint in self.joints:
            for link in (joint.parent, joint.child):
                if link not in self.links:
                    raise ValueError(f"joint {joint.name} names link {link!r}, which links does not list (links: {', '.join(self.links)})")
        children = [joint.child for joint in self.joints]
        if twice := sorted({link for link in children if children.count(link) > 1}):
            raise ValueError(f"a link has one parent joint, {', '.join(twice)} has more")
        roots = [link for link in self.links if link not in children]
        if len(roots) != 1:
            raise ValueError(f"an articulation has exactly one root link (no parent joint), found {', '.join(roots) or 'none'}")
        if len(self.order()) != len(self.joints):
            raise ValueError("joints form a cycle")
        if unreached := sorted(set(self.links) - {roots[0], *children}):
            raise ValueError(f"links {', '.join(unreached)} hang on no joint")
        nodes = [node for link_nodes in self.links.values() for node in link_nodes]
        if repeated := sorted({node for node in nodes if nodes.count(node) > 1}):
            raise ValueError(f"source nodes belong to one link each, {', '.join(repeated)} is listed twice")
        by_name = self.joint_map
        for state, values in self.states.items():
            if state in RESERVED_STATES or state in by_name:
                raise ValueError(f"state name {state!r} is taken, avoid {', '.join(sorted(RESERVED_STATES))} and the joint names")
            for joint_name, value in values.items():
                joint = by_name.get(joint_name)
                if joint is None:
                    raise ValueError(f"state {state} names joint {joint_name!r}, joints: {', '.join(by_name)}")
                if not joint.movable:
                    raise ValueError(f"state {state} sets fixed joint {joint_name}")
                if joint.limit is not None and not joint.limit[0] - STATE_TOLERANCE <= value <= joint.limit[1] + STATE_TOLERANCE:
                    raise ValueError(f"state {state} sets {joint_name} to {value}, outside its limit {list(joint.limit)}")

    @property
    def joint_map(self) -> dict[str, Joint]:
        return {joint.name: joint for joint in self.joints}

    @property
    def root(self) -> str:
        children = {joint.child for joint in self.joints}
        return next(link for link in self.links if link not in children)

    @property
    def movable(self) -> list[Joint]:
        return [joint for joint in self.joints if joint.movable]

    def order(self) -> list[Joint]:
        """Joints parents first, the joints a cycle cuts off left out."""
        children = {joint.child for joint in self.joints}
        reached = {link for link in self.links if link not in children}
        ordered: list[Joint] = []
        pending = list(self.joints)
        while pending:
            ready = [joint for joint in pending if joint.parent in reached]
            if not ready:
                break
            for joint in ready:
                ordered.append(joint)
                reached.add(joint.child)
                pending.remove(joint)
        return ordered

    def initial(self) -> Values:
        """The `closed` state where there is one, else every joint at 0 clamped into its limit."""
        values = {joint.name: joint.clamp(0.0) for joint in self.movable}
        values.update(self.states.get("closed", {}))
        return values

    def resolve(self, spec: str | Mapping[str, float | str] | None, base: Mapping[str, float] | None = None) -> Values:
        """Joint values of a state name or of joint values (each a number or a state name naming that joint), unnamed joints keep base or the initial values."""
        values = dict(base) if base is not None else self.initial()
        if spec is None:
            return values
        if isinstance(spec, str):
            if spec not in self.states:
                raise ValueError(f"unknown state {spec!r}, states: {', '.join(self.states) or 'none'}")
            values.update(self.states[spec])
            return values
        joints = self.joint_map
        for name, value in spec.items():
            joint = joints.get(name)
            if joint is None or not joint.movable:
                raise ValueError(f"unknown movable joint {name!r}, joints: {', '.join(joint.name for joint in self.movable)}")
            if isinstance(value, str):
                if name not in self.states.get(value, {}):
                    raise ValueError(f"state {value!r} sets no value for joint {name}")
                value = self.states[value][name]
            values[name] = joint.clamp(float(value))
        return values

    def state_of(self, values: Mapping[str, float], tolerance: float = STATE_TOLERANCE) -> str:
        """The first state whose joint values all match, empty between states."""
        for state, targets in self.states.items():
            if all(abs(values.get(joint, 0.0) - target) <= tolerance for joint, target in targets.items()):
                return state
        return ""

    def link_poses(self, values: Mapping[str, float]) -> dict[str, Pose]:
        """Model-frame pose of each link's geometry at the joint values, unnamed joints at 0."""
        poses = {self.root: Pose(position=Position(0.0, 0.0, 0.0), orientation=Orientation.identity())}
        for joint in self.order():
            poses[joint.child] = poses[joint.parent] * joint.motion(values.get(joint.name, 0.0))
        return poses

    def transformed(self, transform: np.ndarray) -> Articulation:
        """This articulation carried through a rigid transform with uniform scale, as the build applies to stand a model on the floor."""
        transform = np.asarray(transform, dtype=np.float64)
        linear = transform[:3, :3]
        scale = float(np.cbrt(np.linalg.det(linear)))
        if not scale > 0.0 or not np.allclose(linear.T @ linear, scale * scale * np.eye(3), atol=1e-6 * max(1.0, scale * scale)):
            raise ValueError(f"articulations follow rigid transforms with uniform scale only, got {transform.tolist()}")
        joints = []
        for joint in self.joints:
            origin = linear @ np.array(list(joint.origin)) + transform[:3, 3]
            axis = linear @ np.array(list(joint.axis))
            linear_joint = joint.type is JointType.PRISMATIC
            joints.append(
                attrs.evolve(
                    joint,
                    origin=Position(*origin),
                    axis=Position(*axis),
                    limit=None if joint.limit is None else tuple(bound * scale for bound in joint.limit) if linear_joint else joint.limit,
                    velocity=None if joint.velocity is None else joint.velocity * scale if linear_joint else joint.velocity,
                )
            )
        prismatic = {joint.name for joint in self.joints if joint.type is JointType.PRISMATIC}
        states = {state: {joint: value * scale if joint in prismatic else value for joint, value in values.items()} for state, values in self.states.items()}
        return Articulation(links=self.links, joints=joints, states=states)

    def configurations(self, samples: int = SWEEP_SAMPLES, steps: int = SWEEP_STEPS) -> Iterable[Values]:
        """Joint values covering every joint's reachable range: the full grid of steps per joint when it fits in samples, else every joint's bounds with the others at 0 plus seeded random draws."""
        movable = self.movable
        grids = [np.linspace(*joint.bounds, steps) for joint in movable]
        if steps ** len(movable) <= samples:
            for combination in itertools.product(*grids):
                yield dict(zip((joint.name for joint in movable), combination, strict=True))
            return
        for joint, grid in zip(movable, grids, strict=True):
            for value in grid:
                yield {joint.name: float(value)}
        rng = np.random.default_rng(0)
        for _ in range(samples):
            yield {joint.name: float(rng.uniform(*joint.bounds)) for joint in movable}

    def sweep(self, link_boxes: Mapping[str, Box], samples: int = SWEEP_SAMPLES) -> Box:
        """Bounding box of the link boxes over every configuration, links without a box left out."""
        corners = {link: np.array(list(itertools.product(*box))) for link, box in link_boxes.items()}
        low = np.full(3, np.inf)
        high = np.full(3, -np.inf)
        for values in self.configurations(samples):
            for link, pose in self.link_poses(values).items():
                if link not in corners:
                    continue
                transform = matrix(pose)
                points = corners[link] @ transform[:3, :3].T + transform[:3, 3]
                low = np.minimum(low, points.min(axis=0))
                high = np.maximum(high, points.max(axis=0))
        if not np.all(np.isfinite(low)):
            raise ValueError("no link has a box to sweep")
        return typing.cast(Box, tuple((round(float(lo), 4) + 0.0, round(float(hi), 4) + 0.0) for lo, hi in zip(low, high, strict=True)))

    @classmethod
    def parse(cls, value: Articulation | Mapping[str, typing.Any]) -> Articulation:
        if isinstance(value, Articulation):
            return value
        if not isinstance(value, Mapping):
            raise ValueError(f"articulation must be a mapping of links, joints and states, got {value!r}")
        if unknown := sorted(set(value) - {"links", "joints", "states", "swept"}):
            raise ValueError(f"articulation has unknown keys {unknown}, known: links, joints, states, swept")
        if missing := sorted({"links", "joints"} - set(value)):
            raise ValueError(f"articulation is missing {missing}")
        return cls(**value)

    def unstructure(self) -> dict[str, object]:
        result: dict[str, object] = {"links": {link: list(nodes) for link, nodes in self.links.items()}, "joints": [joint.unstructure() for joint in self.joints]}
        if self.states:
            result["states"] = {state: {joint: round(value, 6) + 0.0 for joint, value in values.items()} for state, values in self.states.items()}
        if self.swept is not None:
            result["swept"] = [list(side) for side in self.swept]
        return result


def structure_articulation(value: Articulation | Mapping[str, typing.Any] | None) -> Articulation | None:
    return None if value is None else Articulation.parse(value)


def unstructure_articulation(value: Articulation | None) -> dict[str, object] | None:
    return None if value is None else value.unstructure()
