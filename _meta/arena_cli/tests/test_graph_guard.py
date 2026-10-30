"""Tests for the CLI's wait for a ROS graph name before it calls a service or echoes a topic."""

import shutil
import subprocess
import sys
import time
import uuid

import pytest

from arena_cli import cli

needs_ros2 = pytest.mark.skipif(shutil.which("ros2") is None, reason="ros2 is not on PATH")

ADVERTISE = """
import sys, rclpy
from std_msgs.msg import String
from std_srvs.srv import Trigger
rclpy.init()
node = rclpy.create_node("graph_guard_standin")
kind, name = sys.argv[1:]
if kind == "service":
    node.create_service(Trigger, name, lambda _req, res: res)
else:
    node.create_publisher(String, name, 1)
rclpy.spin(node)
"""


def _guard(kind: str, name: str) -> subprocess.Popen[str]:
    return subprocess.Popen(["bash", "-c", f'{cli._graph_guard(kind, name, "graph name unreachable")}; echo found'], stdout=subprocess.PIPE, text=True)


@needs_ros2
@pytest.mark.parametrize("kind", ["service", "topic"])
def test_guard_finds_a_name_advertised_after_its_first_listing(kind: str) -> None:
    name = f"/graph_guard_{kind}_{uuid.uuid4().hex[:8]}"
    guard = _guard(kind, name)
    time.sleep(2.0)
    standin = subprocess.Popen([sys.executable, "-c", ADVERTISE, kind, name])
    try:
        out, _ = guard.communicate(timeout=cli._DISCOVERY_WAIT_S + 10)
    finally:
        standin.terminate()
        standin.wait(timeout=10)
    assert (guard.returncode, out.strip()) == (0, "found")


@needs_ros2
def test_guard_reports_unreachable_once_the_wait_runs_out() -> None:
    started = time.monotonic()
    guard = _guard("service", f"/graph_guard_absent_{uuid.uuid4().hex[:8]}")
    out, _ = guard.communicate(timeout=cli._DISCOVERY_WAIT_S + 20)
    assert (guard.returncode, out.strip()) == (1, "graph name unreachable")
    assert time.monotonic() - started >= cli._DISCOVERY_WAIT_S - 1
