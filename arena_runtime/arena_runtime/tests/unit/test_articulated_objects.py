from __future__ import annotations

import asyncio
import math
import typing
import uuid
from pathlib import Path

import pytest
import yaml

DT = 0.25
CABINET_LINKS = {
    "body": [[-0.5, 0.5], [-0.3, 0.3], [0.0, 1.0]],
    "door": [[-0.5, 0.5], [-0.32, -0.3], [0.0, 0.5]],
    "drawer": [[-0.45, 0.45], [-0.3, 0.25], [0.6, 0.8]],
}
CABINET = {
    "links": {"body": ["body"], "door": ["door"], "drawer": ["drawer"], "knob": []},
    "joints": [
        {"name": "door", "type": "revolute", "parent": "body", "child": "door", "origin": [0.5, -0.3, 0.0], "axis": [0, 0, 1], "limit": [0.0, 1.6]},
        {"name": "drawer", "type": "prismatic", "parent": "body", "child": "drawer", "axis": [0, -1, 0], "limit": [0.0, 0.4], "velocity": 0.2},
        {"name": "knob", "type": "continuous", "parent": "door", "child": "knob", "origin": [0.0, -0.32, 0.25], "axis": [0, 1, 0]},
    ],
    "states": {"closed": {"door": 0.0, "drawer": 0.0}, "open": {"door": 1.5, "drawer": 0.4}},
    "swept": [[-0.5, 0.52], [-1.32, 0.3], [0.0, 1.0]],
}


@pytest.fixture
def assets(tmp_path: Path):
    """Asset root with one articulated cabinet and one rigid crate, the model names unique per test."""
    from arena_simulation_setup.tree import DynamicPaths

    suffix = uuid.uuid4().hex[:8]
    cabinet = tmp_path / "Kitchen" / "Object" / f"Cabinet{suffix}"
    for link, box in CABINET_LINKS.items():
        (cabinet / "links" / link).mkdir(parents=True)
        (cabinet / "links" / link / "annotation.yaml").write_text(yaml.safe_dump({"bounding_box": box}))
    (cabinet / "annotation.yaml").write_text(yaml.safe_dump({"bounding_box": CABINET_LINKS["body"], "articulation": CABINET}))
    crate = tmp_path / "Kitchen" / "Object" / f"Crate{suffix}"
    crate.mkdir(parents=True)
    (crate / "annotation.yaml").write_text(yaml.safe_dump({"bounding_box": [[-0.2, 0.2], [-0.2, 0.2], [0.0, 0.4]]}))
    previous = DynamicPaths.ARENA.path
    DynamicPaths.ARENA.path = tmp_path
    yield {"cabinet": f"Kitchen/Cabinet{suffix}", "crate": f"Kitchen/Crate{suffix}"}
    DynamicPaths.ARENA.path = previous


def _realizer():
    from task_generator.manager.realizer import Realizer

    return Realizer(Realizer._Configuration(x=0.0, y=0.0, prefix="env_0"))


def _obstacle(name: str, model: str, x: float = 0.0, y: float = 0.0, yaw: float = 0.0, **kwargs: typing.Any):
    from task_generator.shared import Obstacle, Orientation, Pose, Position

    return _realizer().realize(Obstacle(name=name, model=model, pose=Pose(position=Position(x, y, 0.0), orientation=Orientation.from_yaw(yaw)), **kwargs))


@pytest.fixture
def on_dummy_sim():

    def run(body: typing.Callable[..., typing.Awaitable[None]]) -> None:
        from arena_rclpy_mixins import ArenaMixinNode
        from arena_rclpy_mixins.shared import Namespace

        from arena_runtime.sim.dummy_simulator import DummySimulator

        class RecordingSimulator(DummySimulator):
            def __init__(self, **kwargs: typing.Any) -> None:
                super().__init__(**kwargs)
                self.calls: list[tuple] = []

            async def obstacle_spawn(self, obstacles):
                self.calls.append(("spawn", list(obstacles)))
                return await super().obstacle_spawn(obstacles)

            async def obstacle_move(self, obstacles):
                self.calls.append(("move", list(obstacles)))
                return await super().obstacle_move(obstacles)

            async def obstacle_delete(self, obstacles):
                self.calls.append(("delete", list(obstacles)))
                return await super().obstacle_delete(obstacles)

            async def _move_light(self, light, position, yaw):
                self.calls.append(("light", light.name, position, yaw))
                return await super()._move_light(light, position, yaw)

        async def main() -> None:
            suffix = uuid.uuid4().hex[:8]
            node = ArenaMixinNode(f"articulated_{suffix}", namespace=f"/articulated_{suffix}")
            sim = RecordingSimulator(node=node, namespace=Namespace(f"/articulated_{suffix}"), realizer=_realizer())
            try:
                await body(sim)
            finally:
                await sim.stop_mechanisms()
                node.destroy_node()

        asyncio.run(main())

    return run


def _names(obstacles) -> list[str]:
    return [obstacle.name for obstacle in obstacles]


def _yaw_gap(a: float, b: float) -> float:
    return abs(math.remainder(a - b, 2 * math.pi))


def _snapshot(sim, entity: str = "env_0/cabinet"):
    return next(snap for snap in sim.semantics_snapshot() if snap.entity == entity)


def _register_jackal(sim) -> tuple[str, float]:
    from arena_robots.Robot import RobotIdentifier
    from task_generator.shared import Pose, Robot

    from arena_runtime.sim._interface import robot_footprint_radius

    model = RobotIdentifier.parse("jackal")
    robot = Robot(name="jackal", pose=Pose(), model=model, adapters={}, extra={})
    params = model.resolve_sync().model_params
    sim._register_agent_robot(robot, params)
    return robot.frame.tf(params.base_frame), robot_footprint_radius(params)


def _place(sim, frame: str, x: float, y: float, stamp: int) -> None:
    import geometry_msgs.msg

    t = geometry_msgs.msg.TransformStamped()
    t.header.frame_id = "map"
    t.header.stamp.sec = stamp
    t.child_frame_id = frame
    t.transform.translation.x = x
    t.transform.translation.y = y
    t.transform.rotation.w = 1.0
    sim._mechanism_tf_buffer.set_transform(t, "test")


def test_link_obstacles_carry_the_link_names_scale_and_poses(assets):
    from arena_simulation_setup.utils.geometry import Scale

    from arena_runtime.sim._articulation import resolve_articulated

    cabinet = _obstacle("cabinet", assets["cabinet"], x=1.0, y=2.0, yaw=math.pi / 2, scale=Scale(2.0, 2.0, 1.0), joints={"door": math.pi / 2, "drawer": 0.2})
    cabinet.level = "1"
    runtime = asyncio.run(resolve_articulated(cabinet, "env_0/cabinet"))
    assert runtime is not None
    assert runtime.initial == pytest.approx({"door": math.pi / 2, "drawer": 0.2, "knob": 0.0})
    links = {part.name: part for part in runtime.link_obstacles()}
    assert list(links) == ["cabinet__body", "cabinet__door", "cabinet__drawer"]
    expected = {"cabinet__body": (1.0, 2.0, math.pi / 2), "cabinet__door": (2.6, 2.4, math.pi), "cabinet__drawer": (1.4, 2.0, math.pi / 2)}
    for name, (x, y, yaw) in expected.items():
        part = links[name]
        assert (part.pose.position.x, part.pose.position.y, part.pose.position.z) == pytest.approx((x, y, 0.0))
        assert _yaw_gap(part.pose.orientation.to_yaw(), yaw) == pytest.approx(0.0, abs=1e-9)
        assert part.sim_path == f"env_0/{name}"
        assert part.scale == Scale(2.0, 2.0, 1.0)
        assert part.level == "1"
        assert part.light is None
        assert part.humansim == {}
        assert asyncio.run(part.model.resolve()).path.parts[-2:] == ("links", name.removeprefix("cabinet__"))


def test_a_rigid_or_unresolvable_object_is_not_articulated(assets):
    from arena_runtime.sim._articulation import resolve_articulated

    assert asyncio.run(resolve_articulated(_obstacle("crate", assets["crate"]), "env_0/crate")) is None
    assert asyncio.run(resolve_articulated(_obstacle("ghost", "Kitchen/NoSuchCabinet"), "env_0/ghost")) is None


def test_plain_and_articulated_obstacles_spawn_move_and_delete_in_one_batch_each(assets, on_dummy_sim):
    import attrs
    from task_generator.shared import Pose, Position

    async def body(sim):
        cabinet = _obstacle("cabinet", assets["cabinet"])
        crate = _obstacle("crate", assets["crate"], x=3.0)
        assert list(await sim.spawn_objects([cabinet, crate])) == [True, True]
        assert [(verb, _names(obstacles)) for verb, obstacles in sim.calls] == [("spawn", ["cabinet__body", "cabinet__door", "cabinet__drawer", "crate"])]
        assert set(sim._articulated) == {"cabinet"}
        assert _snapshot(sim).kind == "articulated"
        assert sim._mechanism_loop_task is not None

        sim.calls.clear()
        moved = attrs.evolve(cabinet, pose=Pose(position=Position(5.0, 0.0, 0.0)), joints="open")
        assert list(await sim.move_objects([moved, crate])) == [True, True]
        ((verb, obstacles),) = sim.calls
        assert (verb, _names(obstacles)) == ("move", ["cabinet__body", "cabinet__door", "cabinet__drawer", "crate"])
        assert obstacles[2].pose.position.x == pytest.approx(5.0)
        assert obstacles[2].pose.position.y == pytest.approx(-0.4)
        runtime = sim._articulated["cabinet"]
        assert runtime.current == runtime.target == runtime.initial == pytest.approx({"door": 1.5, "drawer": 0.4, "knob": 0.0})

        sim.calls.clear()
        assert list(await sim.delete_objects([moved, crate])) == [True, True]
        assert [(verb, _names(obstacles)) for verb, obstacles in sim.calls] == [("delete", ["cabinet__body", "cabinet__door", "cabinet__drawer", "crate"])]
        assert sim._articulated == {}
        assert sim.semantics_snapshot() == []

    on_dummy_sim(body)


def test_a_level_object_keeps_its_level_sim_path_and_takes_the_env_prefixed_semantic_name(assets, on_dummy_sim):
    async def body(sim):
        cabinet = _obstacle("cabinet", assets["cabinet"])
        cabinet.sim_path = "env_0/cabinet/1"
        await sim.spawn_objects([cabinet])
        ((_verb, obstacles),) = sim.calls
        assert [obstacle.sim_path for obstacle in obstacles] == ["env_0/cabinet/1__body", "env_0/cabinet/1__door", "env_0/cabinet/1__drawer"]
        assert [snap.entity for snap in sim.semantics_snapshot()] == ["env_0/cabinet"]

    on_dummy_sim(body)


def test_an_unknown_initial_state_fails_only_its_obstacle(assets, on_dummy_sim):
    async def body(sim):
        cabinet = _obstacle("cabinet", assets["cabinet"], joints="ajar")
        crate = _obstacle("crate", assets["crate"])
        assert list(await sim.spawn_objects([cabinet, crate])) == [False, True]
        assert [(verb, _names(obstacles)) for verb, obstacles in sim.calls] == [("spawn", ["crate"])]
        assert sim._articulated == {}

    on_dummy_sim(body)


def test_joints_reach_a_written_target_state_at_their_speed(assets, on_dummy_sim):
    from arena_runtime.sim import _mechanism_shim as shim

    async def body(sim):
        await sim.spawn_objects([_obstacle("cabinet", assets["cabinet"])])
        sim.calls.clear()
        assert sim.set_semantic_value("env_0/cabinet", "target", "open")
        runtime = sim._articulated["cabinet"]
        await shim._tick(sim, DT)
        assert runtime.current == pytest.approx({"door": 0.25, "drawer": 0.05, "knob": 0.0})
        ((verb, obstacles),) = sim.calls
        assert (verb, _names(obstacles)) == ("move", ["cabinet__door", "cabinet__drawer"])
        assert _snapshot(sim).predicates["moving"]
        for _ in range(5):
            await shim._tick(sim, DT)
        assert runtime.current["door"] == pytest.approx(1.5)
        assert _names(sim.calls[-1][1]) == ["cabinet__door", "cabinet__drawer"]
        await shim._tick(sim, DT)
        assert _names(sim.calls[-1][1]) == ["cabinet__drawer"]
        await shim._tick(sim, DT)
        assert runtime.current == pytest.approx({"door": 1.5, "drawer": 0.4, "knob": 0.0})
        calls = len(sim.calls)
        await shim._tick(sim, DT)
        assert len(sim.calls) == calls
        snapshot = _snapshot(sim)
        assert snapshot.discrete == {"state": "open", "target": "open"}
        assert snapshot.predicates == {"closed": False, "open": True, "moving": False, "blocked": False}

    on_dummy_sim(body)


def _door_light(link: str, x: float, y: float, yaw: float):
    from arena_simulation_setup.shared import Light
    from task_generator.shared import Orientation, Pose, Position

    origin = Pose(position=Position(x, y, 0.0), orientation=Orientation.from_yaw(yaw))
    return Light(name=f"cabinet_{link or 'free'}_light", fixture="downlight", lumens=100.0, owner="cabinet", link=link, pose=origin * Pose(position=Position(0.0, -0.31, 0.4), orientation=Orientation.from_euler((0.3, 0.0, 0.0))))


def _light_moves(sim) -> list[tuple]:
    return [call for call in sim.calls if call[0] == "light"]


def test_a_light_on_the_door_link_swings_with_the_door(assets, on_dummy_sim):
    from arena_runtime.sim import _mechanism_shim as shim

    x, y, yaw = 1.0, 2.0, 0.7

    async def body(sim):
        await sim.spawn_objects([_obstacle("cabinet", assets["cabinet"], x=x, y=y, yaw=yaw)])
        light = _door_light("door", x, y, yaw)
        assert await sim.spawn_lights([light])
        sim.calls.clear()
        assert sim.set_semantic_value("env_0/cabinet", "target", "open")
        runtime = sim._articulated["cabinet"]
        for _ in range(8):
            await shim._tick(sim, DT)
        assert runtime.current["door"] == pytest.approx(1.5)
        angle = 1.5
        px, py = 0.0 - 0.5, -0.31 + 0.3
        qx, qy = 0.5 + math.cos(angle) * px - math.sin(angle) * py, -0.3 + math.sin(angle) * px + math.cos(angle) * py
        expected = (x + math.cos(yaw) * qx - math.sin(yaw) * qy, y + math.sin(yaw) * qx + math.cos(yaw) * qy, 0.4)
        _verb, name, position, turn = _light_moves(sim)[-1]
        assert name == light.name
        assert (position.x, position.y, position.z) == pytest.approx(expected)
        assert _yaw_gap(turn, angle) == pytest.approx(0.0, abs=1e-9)

    on_dummy_sim(body)


def test_lights_off_the_moving_links_stay_put_while_the_door_swings(assets, on_dummy_sim):
    from arena_runtime.sim import _mechanism_shim as shim

    async def body(sim):
        await sim.spawn_objects([_obstacle("cabinet", assets["cabinet"], x=1.0, y=2.0, yaw=0.7)])
        assert await sim.spawn_lights([_door_light("", 1.0, 2.0, 0.7), _door_light("body", 1.0, 2.0, 0.7)])
        sim.calls.clear()
        assert sim.set_semantic_value("env_0/cabinet", "target", "open")
        for _ in range(8):
            await shim._tick(sim, DT)
        assert sim._articulated["cabinet"].current["door"] == pytest.approx(1.5)
        assert _light_moves(sim) == []

    on_dummy_sim(body)


def test_an_agent_in_the_swing_holds_the_object_until_it_leaves(assets, on_dummy_sim):
    from arena_runtime.sim import _mechanism_shim as shim

    async def body(sim):
        frame, radius = _register_jackal(sim)
        assert 0.2 < radius < 0.45
        await sim.spawn_objects([_obstacle("cabinet", assets["cabinet"])])
        sim.calls.clear()
        _place(sim, frame, 0.2, -1.2, stamp=1)
        assert sim.set_semantic_value("env_0/cabinet", "door", "1.5")
        runtime = sim._articulated["cabinet"]
        for _ in range(8):
            await shim._tick(sim, DT)
        held = runtime.current["door"]
        assert 0.0 < held < 1.5
        assert runtime.blocked
        assert _snapshot(sim).predicates["blocked"]
        assert {name for _verb, obstacles in sim.calls for name in _names(obstacles)} == {"cabinet__door"}
        await shim._tick(sim, DT)
        assert runtime.current["door"] == held
        _place(sim, frame, 10.0, 10.0, stamp=2)
        for _ in range(8):
            await shim._tick(sim, DT)
        assert runtime.current["door"] == pytest.approx(1.5)
        assert not runtime.blocked
        assert runtime.current["drawer"] == 0.0

    on_dummy_sim(body)


def test_semantic_writes_take_state_names_and_clamped_joint_values(assets, on_dummy_sim):
    async def body(sim):
        await sim.spawn_objects([_obstacle("cabinet", assets["cabinet"])])
        snapshot = _snapshot(sim)
        assert snapshot.discrete == {"state": "closed", "target": "closed"}
        assert snapshot.continuous == {"door": 0.0, "drawer": 0.0, "knob": 0.0}
        assert snapshot.predicates == {"closed": True, "open": False, "moving": False, "blocked": False}
        runtime = sim._articulated["cabinet"]
        assert sim.set_semantic_value("env_0/cabinet", "drawer", "9")
        assert runtime.target == {"door": 0.0, "drawer": 0.4, "knob": 0.0}
        assert sim.set_semantic_value("env_0/cabinet", "target", "open")
        assert runtime.target == {"door": 1.5, "drawer": 0.4, "knob": 0.0}
        assert sim.set_semantic_value("env_0/cabinet", "knob", str(3 * math.pi / 2))
        assert runtime.target["knob"] == pytest.approx(-math.pi / 2)
        assert _snapshot(sim).discrete == {"state": "closed", "target": "open"}
        for field, value, reason in (("target", "ajar", "malformed value"), ("drawer", "wide", "malformed value"), ("drawer", "nan", "malformed value"), ("lid", "1.0", "field not writable"), ("moving", "true", "field not writable")):
            with pytest.raises(ValueError, match=reason):
                sim.set_semantic_value("env_0/cabinet", field, value)
        assert runtime.target == {"door": 1.5, "drawer": 0.4, "knob": pytest.approx(-math.pi / 2)}

    on_dummy_sim(body)


def test_reset_returns_the_joints_to_the_initial_values_and_places_the_links_again(assets, on_dummy_sim):
    from arena_runtime.sim import _mechanism_shim as shim

    async def body(sim):
        await sim.spawn_objects([_obstacle("cabinet", assets["cabinet"], joints={"drawer": "open"})])
        runtime = sim._articulated["cabinet"]
        assert sim.set_semantic_value("env_0/cabinet", "target", "closed")
        await shim._tick(sim, DT)
        assert runtime.current["drawer"] == pytest.approx(0.35)
        sim.reset_semantics()
        assert runtime.current == runtime.target == {"door": 0.0, "drawer": 0.4, "knob": 0.0}
        sim.calls.clear()
        await shim._tick(sim, DT)
        ((verb, obstacles),) = sim.calls
        assert (verb, _names(obstacles)) == ("move", ["cabinet__body", "cabinet__door", "cabinet__drawer"])
        assert obstacles[2].pose.position.y == pytest.approx(-0.4)

    on_dummy_sim(body)


def test_an_episode_reset_move_starts_from_the_new_joints(assets, on_dummy_sim):
    import attrs

    async def body(sim):
        cabinet = _obstacle("cabinet", assets["cabinet"])
        await sim.spawn_objects([cabinet])
        assert sim.set_semantic_value("env_0/cabinet", "target", "open")
        await sim.move_objects([attrs.evolve(cabinet, joints={"door": "open"})])
        runtime = sim._articulated["cabinet"]
        assert runtime.current == runtime.target == runtime.initial == {"door": 1.5, "drawer": 0.0, "knob": 0.0}
        snapshot = _snapshot(sim)
        assert snapshot.discrete == {"state": "", "target": ""}
        assert snapshot.predicates == {"closed": False, "open": False, "moving": False, "blocked": False}
        sim.reset_semantics()
        assert runtime.current == {"door": 1.5, "drawer": 0.0, "knob": 0.0}
        await sim.move_objects([cabinet])
        assert runtime.current == {"door": 0.0, "drawer": 0.0, "knob": 0.0}

    on_dummy_sim(body)


def test_a_timeline_set_drives_the_object_and_the_judge_reads_its_state(assets, on_dummy_sim):
    import random
    import types

    from arena_simulation_setup.shared.conditions import parse_atom
    from arena_simulation_setup.shared.judge import Sample, atom_holds
    from arena_simulation_setup.tree.World.Scenario import TimelineEntry
    from task_generator.node import TaskGenerator

    from arena_runtime.sim import _mechanism_shim as shim

    class Logger:
        def warning(self, msg: str) -> None:
            raise AssertionError(msg)

    async def body(sim):
        await sim.spawn_objects([_obstacle("cabinet", assets["cabinet"])])
        node = types.SimpleNamespace(_semantic_names={}, _realizer=_realizer(), _simulator=sim, _sound_levels={}, _timeline_state=[{"rng": random.Random("7:0")}], get_logger=Logger)
        for method in ("_resolve_semantic_entity", "_apply_semantic_checked", "_set_semantic", "_resolve_timeline_value", "_semantic_value"):
            setattr(node, method, getattr(TaskGenerator, method).__get__(node))
        node._norm_token = TaskGenerator._norm_token
        node._stringify_float = TaskGenerator._stringify_float

        def holds(atom: str) -> bool | None:
            return atom_holds(parse_atom(atom), Sample(t=0.0, field=node._semantic_value), {})

        assert holds("cabinet.state == closed")
        TaskGenerator._fire_timeline_entry(node, 0, TimelineEntry(set=[{"entity": "cabinet", "field": "target", "value": "open"}], at=0.0))
        assert holds("cabinet.target == open")
        assert not holds("cabinet.state == open")
        for _ in range(8):
            await shim._tick(sim, DT)
        assert holds("cabinet.state == open")
        assert holds("cabinet.door == 1.5")
        assert holds("cabinet.moving == false")

    on_dummy_sim(body)


def test_disc_hits_a_convex_hull_inside_and_within_its_radius():
    from arena_runtime.sim._articulation import disc_hits, hull

    square = hull([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0), (0.5, 0.5)])
    assert sorted(square) == [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)]
    assert disc_hits(square, (0.5, 0.5), 0.01)
    assert disc_hits(square, (1.2, 0.5), 0.3)
    assert not disc_hits(square, (1.2, 0.5), 0.1)
    assert not disc_hits([], (0.0, 0.0), 1.0)
