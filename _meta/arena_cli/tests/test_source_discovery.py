"""Tests for the ROS 2 discovery scope set by _meta/tools/source and the DDS profiles."""

import shutil
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

ARENA = Path(__file__).resolve().parents[3]
NS = "{http://www.eprosima.com/XMLSchemas/fastRTPS_Profiles}"
SOURCE = ARENA / "_meta" / "tools" / "source"
PRINT = 'printf "%s\\n" "$ROS_AUTOMATIC_DISCOVERY_RANGE" "$FASTRTPS_DEFAULT_PROFILES_FILE" "$CYCLONEDDS_URI"'


def _sourced(tmp_path: Path, env: dict[str, str]) -> list[str]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for tool in ("python3", "realpath"):
        found = shutil.which(tool)
        assert found is not None
        (bin_dir / tool).symlink_to(found)
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "Arena").symlink_to(ARENA)
    base = {"PATH": str(bin_dir), "HOME": str(tmp_path), "ARENA_WS_DIR": str(tmp_path), "ARENA_ROS_DISTRO": "none", "ARENA_VENV_DIR": str(tmp_path / "venv")}
    result = subprocess.run([shutil.which("bash") or "bash", "-c", f'source "$0" >/dev/null 2>&1; {PRINT}', str(SOURCE)], env={**base, **env}, capture_output=True, text=True, check=True, cwd=tmp_path)
    return result.stdout.splitlines()


@pytest.mark.parametrize("inherited", ["LOCALHOST", "SYSTEM_DEFAULT", ""])
def test_inherited_discovery_range_is_reset_to_subnet(tmp_path: Path, inherited: str) -> None:
    out = _sourced(tmp_path, {"ROS_AUTOMATIC_DISCOVERY_RANGE": inherited} if inherited else {})
    assert out == ["SUBNET", f"{tmp_path}/src/Arena/_meta/docker/fastdds.local.xml", f"file://{tmp_path}/src/Arena/_meta/docker/cyclonedds.local.xml"]


def test_lan_net_keeps_subnet_with_the_lan_profiles(tmp_path: Path) -> None:
    out = _sourced(tmp_path, {"ROS_AUTOMATIC_DISCOVERY_RANGE": "LOCALHOST", "ARENA_NET": "lan"})
    assert out == ["SUBNET", f"{tmp_path}/src/Arena/_meta/docker/fastdds.lan.xml", f"file://{tmp_path}/src/Arena/_meta/docker/cyclonedds.lan.xml"]


def test_profiles_pin_loopback_for_local_net() -> None:
    fastdds = (ARENA / "_meta" / "docker" / "fastdds.local.xml").read_text()
    cyclone = (ARENA / "_meta" / "docker" / "cyclonedds.local.xml").read_text()
    assert "<address>127.0.0.1</address>" in fastdds
    assert "<useBuiltinTransports>false</useBuiltinTransports>" in fastdds
    assert '<NetworkInterface name="lo" />' in cyclone


@pytest.mark.parametrize("net", ["local", "lan"])
def test_fastdds_profiles_probe_more_than_32_participant_ids(net: str) -> None:
    transport = ET.parse(ARENA / "_meta" / "docker" / f"fastdds.{net}.xml").getroot().find(f"{NS}transport_descriptors/{NS}transport_descriptor")
    assert transport is not None
    assert int(transport.findtext(f"{NS}maxInitialPeersRange", "0")) == 120


def test_local_fastdds_profile_sizes_socket_buffers_to_4_mb() -> None:
    transport = ET.parse(ARENA / "_meta" / "docker" / "fastdds.local.xml").getroot().find(f"{NS}transport_descriptors/{NS}transport_descriptor")
    assert transport is not None
    assert int(transport.findtext(f"{NS}sendBufferSize", "0")) == 4194304
    assert int(transport.findtext(f"{NS}receiveBufferSize", "0")) == 4194304
