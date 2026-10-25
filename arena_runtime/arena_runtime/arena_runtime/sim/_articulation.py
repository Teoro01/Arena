"""Articulated objects in every simulator: one obstacle per link bundle, joints moved kinematically on the mechanism tick."""

from __future__ import annotations

import asyncio
import math
import typing
from collections.abc import Awaitable, Callable, Sequence

import attrs
from arena_simulation_setup.tree.assets.Object import ObjectIdentifier, ObjectView
from arena_simulation_setup.utils.articulation import Articulation, Joint, JointType, Values
from task_generator.shared import Obstacle, Pose, Position

if typing.TYPE_CHECKING:
    from arena_simulation_setup.shared import Light
    from arena_simulation_setup.utils.geometry import Scale

    from ._interface import MechanismITF

LINK_SEPARATOR = '__'
JOINT_TOLERANCE = 1e-6

Disc = tuple[str, tuple[float, float], float]
Point = tuple[float, float]


def joint_gap(joint: Joint, current: float, target: float) -> float:
    """Signed distance from current to target, the short way round for a continuous joint."""
    if joint.type is JointType.CONTINUOUS:
        return math.remainder(target - current, 2 * math.pi)
    return target - current


def scaled(pose: Pose, scale: Scale | None) -> Pose:
    """A model-frame link pose with its translation scaled componentwise."""
    if scale is None:
        return pose
    position = pose.position
    return Pose(position=Position(position.x * scale.x, position.y * scale.y, position.z * scale.z), orientation=pose.orientation)


def hull(points: Sequence[Point]) -> list[Point]:
    """Convex hull, counter-clockwise."""
    ordered = sorted(set(points))
    if len(ordered) < 3:
        return ordered

    def cross(o: Point, a: Point, b: Point) -> float:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list[Point] = []
    upper: list[Point] = []
    for chain, sequence in ((lower, ordered), (upper, reversed(ordered))):
        for point in sequence:
            while len(chain) >= 2 and cross(chain[-2], chain[-1], point) <= 0.0:
                chain.pop()
            chain.append(point)
    return lower[:-1] + upper[:-1]


def disc_hits(polygon: Sequence[Point], center: Point, radius: float) -> bool:
    """Whether a disc overlaps a convex counter-clockwise polygon."""
    if not polygon:
        return False
    x, y = center
    inside = len(polygon) >= 3
    for i, (ax, ay) in enumerate(polygon):
        bx, by = polygon[(i + 1) % len(polygon)]
        dx, dy = bx - ax, by - ay
        length = dx * dx + dy * dy
        t = 0.0 if length <= 0.0 else min(1.0, max(0.0, ((x - ax) * dx + (y - ay) * dy) / length))
        if math.hypot(x - (ax + t * dx), y - (ay + t * dy)) < radius:
            return True
        inside = inside and dx * (y - ay) - dy * (x - ax) >= 0.0
    return inside


@attrs.define
class ArticulatedObject:
    """One articulated obstacle: its link bundles and its initial, current and target joint values."""

    obstacle: Obstacle
    name: str
    sim_path: str
    entity: str
    articulation: Articulation
    links: dict[str, ObjectView]
    initial: Values
    current: Values
    target: Values
    blocked: bool = False
    placed: bool = True

    @property
    def moving(self) -> bool:
        return any(abs(joint_gap(joint, self.current[joint.name], self.target[joint.name])) > JOINT_TOLERANCE for joint in self.articulation.movable)

    def reset(self) -> None:
        """Current and target back to the initial values, the links placed again on the next tick."""
        self.current = dict(self.initial)
        self.target = dict(self.initial)
        self.blocked = False
        self.placed = False

    def advance(self, dt: float) -> Values:
        """Joint values one step of dt toward the target at each joint's speed."""
        values = dict(self.current)
        for joint in self.articulation.movable:
            current, target = self.current[joint.name], self.target[joint.name]
            gap = joint_gap(joint, current, target)
            step = joint.speed * dt
            values[joint.name] = target if abs(gap) <= step else joint.clamp(current + math.copysign(step, gap))
        return values

    def link_poses(self, values: Values) -> dict[str, Pose]:
        """Map-frame pose of every link bundle at the joint values."""
        poses = self.articulation.link_poses(values)
        return {link: self.obstacle.pose * scaled(poses[link], self.obstacle.scale) for link in self.links}

    def link_obstacle(self, link: str, pose: Pose) -> Obstacle:
        name = f'{self.name}{LINK_SEPARATOR}{link}'
        part = Obstacle(
            name=name,
            pose=pose,
            model=ObjectIdentifier.inline(self.links[link], name=f'{self.obstacle.model.name}/links/{link}'),
            level=self.obstacle.level,
            scale=self.obstacle.scale,
        )
        part.sim_path = f'{self.sim_path}{LINK_SEPARATOR}{link}'
        return part

    def link_obstacles(self) -> list[Obstacle]:
        """One obstacle per link bundle at the current joint values."""
        return [self.link_obstacle(link, pose) for link, pose in self.link_poses(self.current).items()]

    def footprint(self, link: str, pose: Pose) -> list[Point]:
        """Map-frame xy hull of a link's annotated box at a link pose, empty without a box."""
        box = self.links[link].extent
        if box is None:
            return []
        scale = self.obstacle.scale
        sx, sy, sz = (scale.x, scale.y, scale.z) if scale is not None else (1.0, 1.0, 1.0)
        corners = (pose * Position(x * sx, y * sy, z * sz) for x in box[0] for y in box[1] for z in box[2])
        return hull([(corner.x, corner.y) for corner in corners])


def link_light_pose(runtime: ArticulatedObject, light: Light) -> tuple[Position, float] | None:
    """Map position and yaw turn of a light riding on a link of the object at its current joint values, None off the moving links."""
    articulation = runtime.articulation
    if light.pose is None or not light.link or light.link == articulation.root or light.link not in articulation.links:
        return None
    origin = runtime.obstacle.pose
    motion = scaled(articulation.link_poses(runtime.current)[light.link], runtime.obstacle.scale)
    placed = origin * motion * (origin.inverse() * light.pose)
    return (placed.position, (placed.orientation * light.pose.orientation.inverse()).to_yaw())


async def resolve_articulated(obstacle: Obstacle, entity: str) -> ArticulatedObject | None:
    """The runtime of an obstacle whose object has an articulation and link bundles, its semantic entity named entity, None for a plain one."""
    try:
        view = await obstacle.model.resolve()
    except FileNotFoundError:
        return None
    articulation = view.articulation
    if articulation is None:
        return None
    links: dict[str, ObjectView] = {}
    for link in articulation.links:
        try:
            links[link] = view.link(link)
        except FileNotFoundError:
            continue
    if not links:
        return None
    initial = articulation.resolve(obstacle.joints)
    return ArticulatedObject(
        obstacle=obstacle,
        name=obstacle.name,
        sim_path=obstacle.sim_path,
        entity=entity,
        articulation=articulation,
        links=links,
        initial=initial,
        current=dict(initial),
        target=dict(initial),
    )


async def _dispatch(
    verb: Callable[[Sequence[Obstacle]], Awaitable[Sequence[bool]]],
    groups: Sequence[Sequence[Obstacle] | None],
) -> list[bool]:
    """Send every group in one batch, a group succeeds when all its obstacles do, None fails."""
    batch = [obstacle for group in groups if group is not None for obstacle in group]
    oks = list(await verb(batch)) if batch else []
    results: list[bool] = []
    start = 0
    for group in groups:
        if group is None:
            results.append(False)
            continue
        results.append(all(oks[start : start + len(group)]))
        start += len(group)
    return results


async def spawn_objects(mech: MechanismITF, obstacles: Sequence[Obstacle]) -> list[bool]:
    from ._mechanism_shim import _ensure_loop

    logger = mech.node.get_logger()

    async def resolve(obstacle: Obstacle) -> ArticulatedObject | Exception | None:
        try:
            return await resolve_articulated(obstacle, mech._realizer.prefix(obstacle.name))
        except ValueError as e:
            return e

    runtimes = await asyncio.gather(*map(resolve, obstacles))
    groups: list[Sequence[Obstacle] | None] = []
    for obstacle, runtime in zip(obstacles, runtimes, strict=True):
        if isinstance(runtime, Exception):
            logger.error(f"articulated object {obstacle.name!r} not spawned: {runtime}")
            groups.append(None)
        elif runtime is None:
            groups.append([obstacle])
        else:
            groups.append(runtime.link_obstacles())
    results = await _dispatch(mech.obstacle_spawn, groups)
    for runtime in runtimes:
        if not isinstance(runtime, ArticulatedObject):
            continue
        if (replaced := mech._articulated.get(runtime.name)) is not None:
            mech._semantics.detach(replaced.entity)
        mech._articulated[runtime.name] = runtime
        mech._semantics.attach('articulated', runtime.entity)
        _ensure_loop(mech)
    return results


async def move_objects(mech: MechanismITF, obstacles: Sequence[Obstacle]) -> list[bool]:
    groups: list[Sequence[Obstacle] | None] = []
    for obstacle in obstacles:
        runtime = mech._articulated.get(obstacle.name)
        if runtime is None:
            groups.append([obstacle])
            continue
        try:
            runtime.initial = runtime.articulation.resolve(obstacle.joints)
        except ValueError as e:
            mech.node.get_logger().error(f"articulated object {obstacle.name!r} keeps its initial joint values: {e}")
        runtime.obstacle = obstacle
        runtime.reset()
        runtime.placed = True
        groups.append(runtime.link_obstacles())
    return await _dispatch(mech.obstacle_move, groups)


async def delete_objects(mech: MechanismITF, obstacles: Sequence[Obstacle]) -> list[bool]:
    groups: list[Sequence[Obstacle] | None] = []
    for obstacle in obstacles:
        runtime = mech._articulated.pop(obstacle.name, None)
        if runtime is None:
            groups.append([obstacle])
            continue
        mech._semantics.detach(runtime.entity)
        groups.append(runtime.link_obstacles())
    return await _dispatch(mech.obstacle_delete, groups)


async def step_articulated(mech: MechanismITF, dt: float, discs: Sequence[Disc]) -> None:
    """Move every articulated object one tick toward its target, holding one whose moving links would enter an agent disc."""
    batch: list[Obstacle] = []
    for runtime in list(mech._articulated.values()):
        if not runtime.placed:
            runtime.placed = True
            batch.extend(runtime.link_obstacles())
            continue
        if not runtime.moving:
            runtime.blocked = False
            continue
        values = runtime.advance(dt)
        before = runtime.link_poses(runtime.current)
        after = runtime.link_poses(values)
        moved = [link for link, pose in after.items() if pose != before[link]]
        footprints = [runtime.footprint(link, after[link]) for link in moved]
        if any(disc_hits(footprint, xy, radius) for footprint in footprints for _name, xy, radius in discs):
            runtime.blocked = True
            continue
        runtime.blocked = False
        runtime.current = values
        batch.extend(runtime.link_obstacle(link, after[link]) for link in moved)
    if batch:
        await mech.obstacle_move(batch)
