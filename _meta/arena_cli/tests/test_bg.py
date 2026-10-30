"""Tests for detached launches, their pid records and the escalating stop, against real processes."""

import json
import os
import pathlib
import signal
import sys
import time
from collections.abc import Iterator

import pytest

from arena_cli import bg, cli
from arena_cli.common import CLIError

STANDIN = """
import signal, sys, time
need = int(sys.argv[1])
seen = 0
def on_int(_signo, _frame):
    global seen
    seen += 1
    print(f"sigint {seen}", flush=True)
    if seen >= need:
        sys.exit(0)
signal.signal(signal.SIGINT, on_int)
print("up", flush=True)
while True:
    time.sleep(0.05)
"""


@pytest.fixture
def started(tmp_path: pathlib.Path) -> Iterator[list[int]]:
    pids: list[int] = []
    yield pids
    for pid in pids:
        try:
            os.kill(pid, signal.SIGKILL)
            os.waitpid(pid, 0)
        except (ProcessLookupError, ChildProcessError):
            pass


def _standin(tmp_path: pathlib.Path, started: list[int], need: int) -> tuple[pathlib.Path, dict]:
    state = tmp_path / "state"
    argv = [sys.executable, "-c", STANDIN, str(need)]
    pid, log = bg.detach(argv, "standin", str(tmp_path / "logs"))
    started.append(pid)
    deadline = time.monotonic() + 10
    while "up" not in pathlib.Path(log).read_text() and time.monotonic() < deadline:
        time.sleep(0.05)
    bg.record(str(state), "launch", pid, [str(need)], log)
    return state, json.loads((state / f"{pid}.json").read_text())


def test_take_flag_strips_every_bg_token_and_keeps_order() -> None:
    assert bg.take_flag(["sim:=dummy", "--bg", "env.n:=2"]) == (True, ["sim:=dummy", "env.n:=2"])
    assert bg.take_flag(["sim:=dummy"]) == (False, ["sim:=dummy"])


def test_detach_runs_in_its_own_session_with_output_in_the_log(tmp_path: pathlib.Path, started: list[int]) -> None:
    pid, log = bg.detach([sys.executable, "-c", "import os, sys; print(os.getsid(0), sys.stdin.read() == '')"], "probe", str(tmp_path / "logs"))
    started.append(pid)
    os.waitpid(pid, 0)
    assert pathlib.Path(log).parent == tmp_path / "logs"
    assert pathlib.Path(log).read_text().split() == [str(pid), "True"]


def test_running_lists_a_live_record_and_drops_it_once_the_process_exits(tmp_path: pathlib.Path, started: list[int]) -> None:
    state, rec = _standin(tmp_path, started, need=1)
    assert [r["pid"] for r in bg.running(str(state))] == [rec["pid"]]
    os.kill(rec["pid"], signal.SIGINT)
    os.waitpid(rec["pid"], 0)
    assert bg.running(str(state)) == []
    assert list(state.iterdir()) == []


def test_running_filters_by_kind(tmp_path: pathlib.Path, started: list[int]) -> None:
    state, rec = _standin(tmp_path, started, need=1)
    assert [r["pid"] for r in bg.running(str(state), "launch")] == [rec["pid"]]
    assert bg.running(str(state), "viz") == []


def test_running_drops_a_record_whose_pid_was_reused(tmp_path: pathlib.Path) -> None:
    state = tmp_path / "state"
    state.mkdir()
    (state / f"{os.getpid()}.json").write_text(json.dumps({"kind": "launch", "pid": os.getpid(), "start": "1", "args": [], "log": "x"}))
    assert bg.running(str(state)) == []
    assert list(state.iterdir()) == []


def test_stop_ends_a_launch_that_exits_on_the_first_sigint(tmp_path: pathlib.Path, started: list[int], capsys: pytest.CaptureFixture[str]) -> None:
    state, rec = _standin(tmp_path, started, need=1)
    assert bg.stop(str(state), windows=(5.0,)) == 0
    assert "stopped" in capsys.readouterr().out
    assert list(state.iterdir()) == []
    assert "sigint 1" in pathlib.Path(rec["log"]).read_text()


def test_stop_repeats_sigint_until_the_launch_exits(tmp_path: pathlib.Path, started: list[int]) -> None:
    state, rec = _standin(tmp_path, started, need=3)
    assert bg.stop(str(state), windows=(0.5, 0.5, 5.0)) == 0
    assert pathlib.Path(rec["log"]).read_text().split("\n")[1:4] == ["sigint 1", "sigint 2", "sigint 3"]


def test_stop_fails_when_the_launch_outlives_every_signal(tmp_path: pathlib.Path, started: list[int], capsys: pytest.CaptureFixture[str]) -> None:
    state, rec = _standin(tmp_path, started, need=99)
    assert bg.stop(str(state), windows=(0.3, 0.3)) == 1
    assert f"still alive after 2 signals: {rec['pid']}" in capsys.readouterr().err


def test_stop_refuses_without_a_background_launch(tmp_path: pathlib.Path) -> None:
    with pytest.raises(CLIError, match="no background launch is running"):
        bg.stop(str(tmp_path / "state"))


def test_field_reads_a_spawn_env_response() -> None:
    srv = pytest.importorskip("arena_runtime_msgs.srv")
    ok = repr(srv.SpawnEnv.Response(success=True, env_id=3, ns="/env_3", log_path="/l/arena_env_3.log", error_msg=""))
    assert [bg._field(ok, k) for k in ("success", "env_id", "ns", "log_path", "error_msg")] == ["True", "3", "/env_3", "/l/arena_env_3.log", ""]
    echoed = f"requester: making request: {srv.SpawnEnv.Request(ns='', headless=True)!r}\n\nresponse:\n{ok}"
    assert [bg._field(echoed, k) for k in ("ns", "success")] == ["/env_3", "True"]
    failed = repr(srv.SpawnEnv.Response(success=False, error_msg="reserve failed: it's taken"))
    assert bg._field(failed, "error_msg") == "reserve failed: it's taken"


@pytest.mark.parametrize("args", [["down", "3"], ["3", "down", "x"], ["x", "down"]])
def test_env_down_names_its_form_on_a_bad_argument(args: list[str]) -> None:
    with pytest.raises(CLIError, match="usage: arena env ENV down"):
        cli.env_(args)


def test_runtime_down_refuses_arguments() -> None:
    with pytest.raises(CLIError, match="takes no arguments"):
        cli.runtime(["down", "now"])
