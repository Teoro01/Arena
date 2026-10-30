"""arena settings sim against a temporary .env and installed-features file."""

import os
import subprocess
import sys
from pathlib import Path

ARENA = Path(__file__).parents[3]


def _sim(tmp_path: Path, *args: str) -> subprocess.CompletedProcess[str]:
    env = {
        **os.environ,
        "ARENA_DIR": str(ARENA),
        "ARENA_WS_DIR": str(tmp_path),
        "ARENA_ENV_FILE": str(tmp_path / ".env"),
        "INSTALLED": str(tmp_path / ".installed"),
        "PYTHONPATH": os.pathsep.join([str(ARENA / "_meta"), os.environ.get("PYTHONPATH", "")]),
    }
    return subprocess.run([sys.executable, "-m", "arena_cli", "settings", "sim", *args], env=env, capture_output=True, text=True, check=False)


def _rows(out: str) -> dict[str, str]:
    return {key: value.strip() for key, _, value in (line.partition(":") for line in out.splitlines() if ":" in line and not line.startswith("set it"))}


def _choices(tmp_path: Path) -> list[str]:
    hint = _sim(tmp_path).stdout.splitlines()[-1]
    return hint.removeprefix("set it with 'arena settings sim ").removesuffix("'").split("|")


def _simulators(tmp_path: Path) -> list[str]:
    auto, dummy, *order = _choices(tmp_path)
    assert (auto, dummy) == ("auto", "dummy")
    return order


def test_bare_resolves_to_dummy_without_simulators(tmp_path):
    done = _sim(tmp_path)
    assert done.returncode == 0, done.stderr
    assert _rows(done.stdout) == {"sim": "dummy", "setting": "auto", "source": "default", "installed": "none"}
    assert len(_simulators(tmp_path)) >= 2


def test_bare_takes_the_first_installed_simulator_in_order(tmp_path):
    order = _simulators(tmp_path)
    (tmp_path / ".installed").write_text(f"{order[-1]}\n{order[1]}\n")
    rows = _rows(_sim(tmp_path).stdout)
    assert rows["sim"] == order[1]
    assert rows["installed"] == f"{order[1]}, {order[-1]}"


def test_set_installed_simulator_persists(tmp_path):
    order = _simulators(tmp_path)
    (tmp_path / ".installed").write_text(f"{order[0]}\n{order[-1]}\n")
    (tmp_path / ".env").write_text("ARENA_NET=lan\n")
    done = _sim(tmp_path, order[-1])
    assert done.returncode == 0, done.stderr
    assert _rows(done.stdout)["sim"] == order[-1]
    assert (tmp_path / ".env").read_text() == f"ARENA_NET=lan\nARENA_SIM={order[-1]}\n"


def test_set_uninstalled_simulator_is_refused(tmp_path):
    order = _simulators(tmp_path)
    (tmp_path / ".installed").write_text(f"{order[-1]}\n")
    done = _sim(tmp_path, order[0])
    assert done.returncode != 0
    assert f"arena feature {order[0]} install" in done.stderr
    assert not (tmp_path / ".env").exists()


def test_unknown_value_lists_the_choices(tmp_path):
    done = _sim(tmp_path, "nosuchsim")
    assert done.returncode != 0
    assert f"expected one of: {', '.join(_choices(tmp_path))}" in done.stderr


def test_stale_setting_falls_back_with_a_warning(tmp_path):
    order = _simulators(tmp_path)
    (tmp_path / ".installed").write_text(f"{order[-1]}\n")
    (tmp_path / ".env").write_text(f"ARENA_SIM={order[0]}\n")
    rows = _rows(_sim(tmp_path).stdout)
    assert rows["sim"] == order[-1]
    assert rows["warning"] == f"{order[0]} is not installed, so launches use auto"
