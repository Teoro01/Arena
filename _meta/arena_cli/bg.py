"""Background launches: detached sessions logging to the ROS log dir, supervisors and viz tracked by pid record."""

import datetime
import json
import os
import re
import signal
import subprocess
import sys
import time

from arena_cli.common import CLIError

FLAG = "--bg"
SUPERVISOR = "arena_bringup.supervisor"
SPAWN_ENV = "/arena/spawn_env"
STOP_WINDOWS_S = (90.0, 10.0, 10.0, 5.0)


def take_flag(args: list[str]) -> tuple[bool, list[str]]:
    """Whether --bg is among args, and args without it."""
    return FLAG in args, [a for a in args if a != FLAG]


def log_dir() -> str:
    if os.environ.get("ROS_LOG_DIR"):
        return os.environ["ROS_LOG_DIR"]
    ros_home = os.environ.get("ROS_HOME") or os.path.join(os.path.expanduser("~"), ".ros")
    return os.path.join(ros_home, "log")


def state_dir() -> str:
    return os.path.join("/tmp", "arena-bg", os.environ.get("ROS_DOMAIN_ID", "0"))


def detach(argv: list[str], name: str, logs: str) -> tuple[int, str]:
    """Start argv in its own session, stdin closed and output appended to logs/<name>_<ts>.log."""
    os.makedirs(logs, exist_ok=True)
    log = os.path.join(logs, f"{name}_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}.log")
    with open(log, "ab") as out:
        proc = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.STDOUT, start_new_session=True)
    return proc.pid, log


def _stat(pid: int) -> tuple[str, str] | None:
    """The state letter and start time of pid, None when it is gone."""
    try:
        with open(f"/proc/{pid}/stat") as f:
            fields = f.read().rpartition(")")[2].split()
    except OSError:
        return None
    return fields[0], fields[19]


def record(state: str, kind: str, pid: int, args: list[str], log: str) -> None:
    stat = _stat(pid)
    if stat is None:
        raise CLIError(f"{kind} exited at once, see {log}")
    os.makedirs(state, exist_ok=True)
    with open(os.path.join(state, f"{pid}.json"), "w") as f:
        json.dump({"kind": kind, "pid": pid, "start": stat[1], "args": args, "log": log}, f)


def _alive(rec: dict) -> bool:
    stat = _stat(rec["pid"])
    return stat is not None and stat[0] != "Z" and stat[1] == rec["start"]


def running(state: str, kind: str | None = None) -> list[dict]:
    """Live records under state (of one kind if given), oldest first, records of exited processes removed."""
    try:
        names = os.listdir(state)
    except FileNotFoundError:
        return []
    recs = []
    for name in names:
        path = os.path.join(state, name)
        try:
            with open(path) as f:
                rec = json.load(f)
        except (OSError, ValueError):
            continue
        if not _alive(rec):
            os.remove(path)
        elif kind in (None, rec["kind"]):
            recs.append(rec)
    return sorted(recs, key=lambda r: int(r["start"]))


def launch_supervisor(args: list[str], state: str, logs: str) -> None:
    pid, log = detach([sys.executable, "-m", SUPERVISOR, *args], "arena_launch", logs)
    record(state, "launch", pid, args, log)
    print(f"arena: launch running in the background (pid {pid})")
    print(f"  log:  tail -f {log}")
    print("  stop: arena runtime down")


def stop(state: str, windows: tuple[float, ...] = STOP_WINDOWS_S) -> int:
    """SIGINT every background launch and viz, once per window while any survives."""
    recs = running(state)
    if not recs:
        raise CLIError("no background launch is running, a foreground runtime stops with Ctrl-C in its terminal")
    for rec in recs:
        print(f"arena runtime down: stopping {rec['kind']} pid {rec['pid']} ({' '.join(rec['args']) or 'defaults'})")
    for round_, window in enumerate(windows, start=1):
        for rec in recs:
            if _alive(rec):
                os.kill(rec["pid"], signal.SIGINT)
        deadline = time.monotonic() + window
        tick = time.monotonic() + 10.0
        while any(_alive(r) for r in recs) and time.monotonic() < deadline:
            if time.monotonic() >= tick:
                tick += 10.0
                print(f"arena runtime down: waiting for {sum(_alive(r) for r in recs)} process(es) to exit (signal {round_})", file=sys.stderr)
            time.sleep(0.2)
        if not any(_alive(r) for r in recs):
            running(state)
            print("arena runtime down: stopped")
            return 0
    left = [str(r["pid"]) for r in recs if _alive(r)]
    print(f"arena runtime down: still alive after {len(windows)} signals: {' '.join(left)}", file=sys.stderr)
    return 1


def _service_up(name: str) -> bool:
    try:
        out = subprocess.run(["ros2", "service", "list"], capture_output=True, text=True, check=False, timeout=30).stdout
    except subprocess.TimeoutExpired:
        return False
    return name in out.split()


def _field(output: str, name: str) -> str:
    """One field of the response in `ros2 service call` output, which echoes the request first."""
    match = re.search(rf"\b{name}=(?:'((?:[^'\\]|\\.)*)'|\"((?:[^\"\\]|\\.)*)\"|(\w+))", output.rpartition("response:")[2])
    if match is None:
        return ""
    return next(g for g in match.groups() if g is not None)


def spawn_env(launch_args: list[str]) -> int:
    """Have the runtime spawn and own one env, waiting for the runtime and for the env to reach ACTIVE."""
    waited = 0
    while not _service_up(SPAWN_ENV):
        time.sleep(1)
        waited += 1
        if waited % 10 == 0:
            print(f"arena env: waiting for {SPAWN_ENV} ({waited}s elapsed)", file=sys.stderr)
    print("arena env: spawning an env in the runtime, waiting for it to reach ACTIVE", file=sys.stderr)
    request = json.dumps({"ns": "", "headless": True, "launch_args": launch_args})
    proc = subprocess.run(["ros2", "service", "call", SPAWN_ENV, "arena_runtime_msgs/srv/SpawnEnv", request], capture_output=True, text=True, check=False)
    response = proc.stdout + proc.stderr
    if _field(response, "success") != "True":
        raise CLIError(f"spawn failed: {_field(response, 'error_msg') or response.strip()}")
    env_id = _field(response, "env_id")
    print(f"arena env: env {env_id} running in the runtime ({_field(response, 'ns')})")
    print(f"  log:  tail -f {_field(response, 'log_path')}")
    print(f"  stop: arena env {env_id} down")
    return 0
