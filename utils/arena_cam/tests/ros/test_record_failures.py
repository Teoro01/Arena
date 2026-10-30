"""`arena cam --record --lockstep` against a stand-in sim viewport: a failed take exits nonzero, a transient capture failure is retried."""

from __future__ import annotations

import shutil
import subprocess
import sys
import threading
from pathlib import Path

import pytest
import rclpy
import rclpy.executors

from arena_cam.client import CAPTURE_ATTEMPTS

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None, reason="ffmpeg/ffprobe not on PATH")

WIDTH, HEIGHT = 32, 24
FPS = 10
FRAMES = 10


class Viewport:
    """The /arena surface of a held sim: hold, step, /clock, the camera pose and a capture service that fails on the frames in `failures`."""

    def __init__(self, failures: dict[int, int], hold: bool = True) -> None:
        from arena_rclpy_mixins.qos import latched
        from arena_runtime_msgs.msg import LockstepStatus
        from arena_runtime_msgs.srv import LifecycleHold, LifecycleStep
        from geometry_msgs.msg import PoseStamped
        from rosgraph_msgs.msg import Clock
        from viewport_control_msgs.srv import ViewportCapture, ViewportSetView

        self.node = rclpy.create_node("arena_cam_record_viewport")
        self.failures = dict(failures)
        self.frames = 0
        self.clock_ns = 0
        self.lock = threading.Lock()
        self.node.create_service(ViewportSetView, "/arena/viewport/set_view", self._ok)
        self.node.create_service(ViewportCapture, "/arena/viewport/capture", self._capture)
        if hold:
            self.node.create_service(LifecycleHold, "/arena/sim_lifecycle/hold", self._hold)
        self.node.create_service(LifecycleStep, "/arena/sim_lifecycle/step", self._step)
        self._clock = self.node.create_publisher(Clock, "/clock", 10)
        self._pose = self.node.create_publisher(PoseStamped, "/arena/viewport/camera_pose", 10)
        self.node.create_publisher(LockstepStatus, "/arena/state/lockstep", latched(1)).publish(LockstepStatus(active=False))
        self._Clock, self._PoseStamped = Clock, PoseStamped
        self.node.create_timer(0.05, self._tick)

    def _stamp(self):
        from builtin_interfaces.msg import Time

        return Time(sec=self.clock_ns // 1_000_000_000, nanosec=self.clock_ns % 1_000_000_000)

    def _tick(self) -> None:
        with self.lock:
            stamp = self._stamp()
        self._clock.publish(self._Clock(clock=stamp))
        pose = self._PoseStamped()
        pose.pose.position.z = 5.0
        pose.pose.orientation.w = 1.0
        self._pose.publish(pose)

    def _ok(self, request, response):
        response.success = True
        return response

    def _hold(self, request, response):
        response.paused = request.action == request.ACQUIRE
        return response

    def _step(self, request, response):
        with self.lock:
            self.clock_ns += round(request.seconds * 1e9)
        response.success = True
        response.advanced = request.seconds
        return response

    def _capture(self, request, response):
        from sensor_msgs.msg import Image

        with self.lock:
            frame = self.frames
            if self.failures.get(frame, 0) > 0:
                self.failures[frame] -= 1
                response.success = False
                response.message = "capture timed out, min_sim_time not reached"
                return response
            self.frames += 1
            stamp = self._stamp()
        response.image = Image(height=HEIGHT, width=WIDTH, encoding="rgb8", step=WIDTH * 3, data=bytes([frame % 256]) * (WIDTH * HEIGHT * 3))
        response.image.header.stamp = stamp
        response.success = True
        return response


def _take(tmp_path: Path, viewport: Viewport) -> tuple[subprocess.CompletedProcess[str], int]:
    executor = rclpy.executors.MultiThreadedExecutor(num_threads=2)
    executor.add_node(viewport.node)
    spin = threading.Thread(target=executor.spin, daemon=True)
    spin.start()
    out = tmp_path / "take.mp4"
    try:
        done = subprocess.run(
            [sys.executable, "-m", "arena_cam.cli", "hold", f"duration={FRAMES / FPS}", "--sim", f"--record={out}", "--fps", str(FPS), "--lockstep"],
            capture_output=True,
            text=True,
            timeout=180,
            check=False,
        )
    finally:
        executor.shutdown()
        spin.join(timeout=10)
        viewport.node.destroy_node()
    if not out.exists():
        return done, 0
    probe = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", str(out)], capture_output=True, text=True, check=False)
    return done, int(probe.stdout.strip() or 0)


@pytest.fixture(autouse=True)
def _rclpy():
    rclpy.init()
    yield
    rclpy.shutdown()


def test_a_full_take_exits_zero(tmp_path: Path) -> None:
    done, frames = _take(tmp_path, Viewport({}))
    assert done.returncode == 0, done.stderr
    assert frames == FRAMES


def test_a_capture_that_recovers_within_its_attempts_keeps_every_frame(tmp_path: Path) -> None:
    done, frames = _take(tmp_path, Viewport({3: CAPTURE_ATTEMPTS - 1}))
    assert done.returncode == 0, done.stderr
    assert frames == FRAMES


def test_a_capture_that_keeps_failing_stops_the_take_and_exits_nonzero(tmp_path: Path) -> None:
    viewport = Viewport({3: CAPTURE_ATTEMPTS})
    done, frames = _take(tmp_path, viewport)
    assert done.returncode == 1
    assert "capture failed" in done.stderr
    assert frames == 3
    assert viewport.frames == 3


def test_a_hold_that_times_out_exits_nonzero(tmp_path: Path) -> None:
    done, frames = _take(tmp_path, Viewport({}, hold=False))
    assert done.returncode == 1
    assert "sim hold timed out" in done.stderr
    assert frames == 0
