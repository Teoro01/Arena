"""The pov face watch of `CamNode` against live faces/tracked and view topics."""

from __future__ import annotations

import math
import threading

import pytest
import rclpy
import rclpy.executors
from arena_people_msgs.msg import FaceView
from arena_rclpy_mixins.spin import run_main
from hri_msgs.msg import IdsList
from sensor_msgs.msg import CameraInfo

from arena_cam.camera import Camera
from arena_cam.client import CamNode, FaceWatch
from arena_cam.surfaces import TargetSelection

_TRACKED = "/arena/env_78/humans/faces/tracked"
_FACE = "env_78_agent_2"
_VIEW = f"/arena/env_78/humans/faces/{_FACE}/view"
_TARGETS = TargetSelection(include_sim=True, viz_all=False, viz_env=None)


class _FaceWatcher(CamNode):
    watched: FaceWatch | None = None

    async def setup(self) -> None:
        _FaceWatcher.watched = await self.watch_face("2", timeout=10.0)
        rclpy.try_shutdown()


def test_watching_a_face_reads_its_view():
    context = rclpy.Context()
    rclpy.init(context=context)
    producer = rclpy.create_node("pov_faces_producer", context=context)
    executor = rclpy.executors.SingleThreadedExecutor(context=context)
    executor.add_node(producer)
    spinner = threading.Thread(target=executor.spin, daemon=True)
    try:
        tracked = producer.create_publisher(IdsList, _TRACKED, 10)
        view = producer.create_publisher(FaceView, _VIEW, 10)
        f = 256.0 / math.tan(math.radians(25.0))
        info = CameraInfo(width=512, height=512, k=[f, 0.0, 256.0, 0.0, f, 256.0, 0.0, 0.0, 1.0])
        producer.create_timer(0.05, lambda: tracked.publish(IdsList(ids=[_FACE])))
        producer.create_timer(0.05, lambda: view.publish(FaceView(info=info, clip_near=0.12)))
        spinner.start()

        run_main(lambda: _FaceWatcher(timeline=Camera(_TARGETS), targets=_TARGETS, node_name="pov_faces_cam"))

        watch = _FaceWatcher.watched
        assert watch is not None
        assert watch.frame == f"face_{_FACE}"
        assert watch.fov == pytest.approx(math.radians(50.0))
        assert watch.clip_near == pytest.approx(0.12)
    finally:
        executor.shutdown()
        producer.destroy_node()
        rclpy.shutdown(context=context)
        spinner.join(timeout=5.0)
