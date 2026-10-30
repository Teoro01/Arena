"""The isaac bridge build: both message packages land in their own install base as plain files."""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from arena_cli.features import isaac

META = Path(isaac.__file__).resolve().parents[2]
needs_colcon = pytest.mark.skipif(shutil.which("colcon") is None or "ARENA_WS_DIR" not in os.environ, reason="needs colcon and a sourced Arena workspace")


@needs_colcon
def test_the_bridge_builds_without_symlinks(tmp_path: Path) -> None:
    env = {**os.environ, "BUILD_BASE": str(tmp_path / "build"), "INSTALL_BASE": str(tmp_path / "install"), "PYTHONPATH": os.pathsep.join([str(META), os.environ.get("PYTHONPATH", "")])}
    done = subprocess.run([sys.executable, "-c", "from arena_cli.features import isaac; raise SystemExit(isaac._build_bridge())"], env=env, capture_output=True, text=True, check=False)
    assert done.returncode == 0, done.stdout + done.stderr
    bridge = tmp_path / "install" / isaac.BRIDGE_DIR
    assert (bridge / "local_setup.bash").is_file()
    assert {p.name for p in bridge.iterdir() if p.is_dir()} >= set(isaac.BRIDGE_PACKAGES)
    assert [p for p in bridge.rglob("*") if p.is_symlink()] == []
