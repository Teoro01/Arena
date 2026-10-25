from __future__ import annotations

import math

import numpy as np
import pytest

from arena_simulation_setup.utils.articulation import Articulation, Joint, JointType, matrix

CABINET = {
    "links": {"base": [], "door": ["door_mesh"], "drawer": ["drawer_mesh", "handle"]},
    "joints": [
        {"name": "door", "type": "revolute", "parent": "base", "child": "door", "origin": [0.5, -0.3, 0.0], "axis": [0, 0, 2], "limit": [0.0, 1.57]},
        {"name": "drawer", "type": "prismatic", "parent": "base", "child": "drawer", "origin": [0.0, 0.0, 0.6], "axis": [0, -1, 0], "limit": [0.0, 0.4]},
    ],
    "states": {"closed": {"door": 0.0, "drawer": 0.0}, "open": {"door": 1.5, "drawer": 0.35}},
}


def cabinet() -> Articulation:
    return Articulation.parse(CABINET)


def point(pose, xyz):
    return list(pose * type(pose.position)(*xyz))


def test_parse_normalizes_the_axis_and_finds_the_root():
    articulation = cabinet()
    assert list(articulation.joint_map["door"].axis) == [0.0, 0.0, 1.0]
    assert articulation.root == "base"
    assert [joint.name for joint in articulation.movable] == ["door", "drawer"]


def test_unstructure_round_trips():
    articulation = cabinet()
    assert Articulation.parse(articulation.unstructure()).unstructure() == articulation.unstructure()


def test_revolute_turns_about_its_origin():
    poses = cabinet().link_poses({"door": math.pi / 2})
    assert point(poses["door"], (0.5, -0.3, 1.0)) == pytest.approx([0.5, -0.3, 1.0])
    assert point(poses["door"], (0.0, -0.3, 0.0)) == pytest.approx([0.5, -0.8, 0.0])
    assert point(poses["base"], (1.0, 2.0, 3.0)) == pytest.approx([1.0, 2.0, 3.0])


def test_prismatic_slides_along_its_axis():
    poses = cabinet().link_poses({"drawer": 0.3})
    assert point(poses["drawer"], (0.1, 0.2, 0.6)) == pytest.approx([0.1, -0.1, 0.6])


def test_chained_joints_compose_parent_first():
    articulation = Articulation.parse(
        {
            "links": {"base": [], "carrier": [], "lid": ["lid"]},
            "joints": [
                {"name": "slide", "type": "prismatic", "parent": "base", "child": "carrier", "axis": [1, 0, 0], "limit": [0, 1]},
                {"name": "hinge", "type": "continuous", "parent": "carrier", "child": "lid", "origin": [0, 0, 1], "axis": [0, 1, 0]},
            ],
        }
    )
    pose = articulation.link_poses({"slide": 0.5, "hinge": math.pi})["lid"]
    assert point(pose, (0.0, 0.0, 1.0)) == pytest.approx([0.5, 0.0, 1.0])
    assert point(pose, (1.0, 0.0, 1.0)) == pytest.approx([-0.5, 0.0, 1.0])


def test_initial_is_the_closed_state_else_zero_clamped():
    assert cabinet().initial() == {"door": 0.0, "drawer": 0.0}
    articulation = Articulation.parse({"links": {"base": [], "lid": []}, "joints": [{"name": "lid", "type": "revolute", "parent": "base", "child": "lid", "limit": [0.2, 1.0]}]})
    assert articulation.initial() == {"lid": 0.2}


def test_resolve_reads_state_names_and_values_and_clamps():
    articulation = cabinet()
    assert articulation.resolve("open") == {"door": 1.5, "drawer": 0.35}
    assert articulation.resolve({"drawer": 9.0}) == {"door": 0.0, "drawer": 0.4}
    assert articulation.resolve({"door": "open"}) == {"door": 1.5, "drawer": 0.0}
    with pytest.raises(ValueError, match="unknown state 'ajar', states: closed, open"):
        articulation.resolve("ajar")
    with pytest.raises(ValueError, match="unknown movable joint 'lid'"):
        articulation.resolve({"lid": 1.0})


def test_state_of_matches_within_tolerance_and_is_empty_between():
    articulation = cabinet()
    assert articulation.state_of({"door": 1.5, "drawer": 0.3500001}) == "open"
    assert articulation.state_of({"door": 0.7, "drawer": 0.0}) == ""


def test_continuous_joint_wraps_and_fixed_stays():
    continuous = Joint(name="wheel", type=JointType.CONTINUOUS, parent="a", child="b")
    assert continuous.clamp(3 * math.pi / 2) == pytest.approx(-math.pi / 2)
    assert continuous.bounds == (-math.pi, math.pi)
    fixed = Joint(name="glass", type=JointType.FIXED, parent="a", child="c")
    assert fixed.clamp(1.0) == 0.0


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda value: value["joints"][0].pop("limit"), "revolute joint door needs a limit"),
        (lambda value: value["joints"][0].update(child="base"), "connects link base to itself"),
        (lambda value: value["joints"][1].update(child="door"), "a link has one parent joint, door has more"),
        (lambda value: value["joints"][0].update(parent="frame"), "names link 'frame', which links does not list"),
        (lambda value: value["links"].update(spare=[]), "exactly one root link"),
        (lambda value: value["links"].update(drawer=["door_mesh"]), "door_mesh is listed twice"),
        (lambda value: value["states"].update(moving={"door": 0.0}), "state name 'moving' is taken"),
        (lambda value: value["states"].update(wide={"door": 3.0}), "outside its limit"),
        (lambda value: value["joints"][0].update(axis=[0, 0, 0]), "joint axis must not be zero"),
        (lambda value: value["joints"][0].update(hinge="left"), "unknown keys \\['hinge'\\]"),
        (lambda value: value.update(extra=1), "unknown keys \\['extra'\\]"),
    ],
)
def test_parse_names_the_problem(change, message):
    value = {"links": {k: list(v) for k, v in CABINET["links"].items()}, "joints": [dict(joint) for joint in CABINET["joints"]], "states": {k: dict(v) for k, v in CABINET["states"].items()}}
    change(value)
    with pytest.raises(ValueError, match=message):
        Articulation.parse(value)


def test_a_cycle_is_refused():
    with pytest.raises(ValueError, match="exactly one root link|cycle"):
        Articulation.parse(
            {
                "links": {"a": [], "b": []},
                "joints": [
                    {"name": "ab", "type": "continuous", "parent": "a", "child": "b"},
                    {"name": "ba", "type": "continuous", "parent": "b", "child": "a"},
                ],
            }
        )


def test_only_fixed_joints_are_refused():
    with pytest.raises(ValueError, match="at least one joint that is not fixed"):
        Articulation.parse({"links": {"a": [], "b": []}, "joints": [{"name": "ab", "type": "fixed", "parent": "a", "child": "b"}]})


def test_transformed_follows_the_floor_transform_and_scales_prismatic_values():
    transform = np.eye(4)
    transform[:3, :3] = 0.01 * np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    transform[:3, 3] = [0.0, 0.0, 0.2]
    moved = cabinet().transformed(transform)
    door = moved.joint_map["door"]
    drawer = moved.joint_map["drawer"]
    assert list(door.origin) == pytest.approx([0.003, 0.005, 0.2])
    assert door.limit == (0.0, 1.57)
    assert list(drawer.axis) == pytest.approx([1.0, 0.0, 0.0])
    assert drawer.limit == pytest.approx((0.0, 0.004))
    assert moved.states["open"] == pytest.approx({"door": 1.5, "drawer": 0.0035})
    p = np.array([0.4, -0.2, 0.1, 1.0])
    original = matrix(cabinet().link_poses({"door": 1.0})["door"]) @ p
    carried = matrix(moved.link_poses({"door": 1.0})["door"]) @ (transform @ p)
    assert carried[:3] == pytest.approx((transform @ original)[:3])


def test_transformed_refuses_shear():
    shear = np.eye(4)
    shear[0, 1] = 0.5
    with pytest.raises(ValueError, match="uniform scale only"):
        cabinet().transformed(shear)


def test_sweep_covers_the_door_swing():
    swept = cabinet().sweep({"base": ((-0.5, 0.5), (-0.3, 0.3), (0.0, 1.0)), "door": ((-0.5, 0.5), (-0.32, -0.3), (0.0, 1.0))})
    (x_min, x_max), (y_min, y_max), (z_min, z_max) = swept
    assert y_min == pytest.approx(-1.30, abs=0.005)
    assert x_max == pytest.approx(0.52, abs=0.005)
    assert (x_min, y_max, z_min, z_max) == pytest.approx((-0.5, 0.3, 0.0, 1.0))


def test_configurations_fall_back_to_draws_for_many_joints():
    articulation = Articulation.parse(
        {
            "links": {"base": [], **{f"wheel_{i}": [] for i in range(10)}},
            "joints": [{"name": f"wheel_{i}", "type": "continuous", "parent": "base", "child": f"wheel_{i}"} for i in range(10)],
        }
    )
    configurations = list(articulation.configurations(samples=64))
    assert len(configurations) == 10 * 9 + 64
    assert configurations[0] == {"wheel_0": -math.pi}
