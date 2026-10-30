"""Workspace settings persisted in the .env file."""

import os
import sys
from types import ModuleType

from arena_cli import features
from arena_cli.common import CLIError, _env, _env_file, _env_set, _host_path, _reg_list, _reg_require, _row, make_verb
from arena_cli.complete import Static

DESCRIPTION = "Workspace settings, persisted in .env.\n\nEvery `source arena` reads them, on the host and in the container. A bare command shows the current value."

NET_MODES = {
    "local": "ROS 2 stays on loopback (default)",
    "lan": "ROS 2 uses every interface",
}


def _defaults() -> ModuleType:
    """arena_bringup.defaults, from the source tree when no installed arena_bringup is on the path."""
    arena = _env("ARENA_DIR")
    for path in (os.path.join(arena, "arena_bringup"), os.path.join(arena, "arena_runtime", "arena_runtime")):
        if path not in sys.path:
            sys.path.append(path)
    from arena_bringup import defaults

    return defaults


def _sim_modes() -> dict[str, str]:
    defaults = _defaults()
    dummy = defaults.SimSimulator.DUMMY.value
    order = [s.value for s in defaults.SIM_ORDER]
    return {
        "auto": f"the first installed of {', '.join(order)}, else {dummy} (default)",
        dummy: "no physics engine",
        **{name: f"needs 'arena feature {name} install'" for name in order},
    }


def _env_value(key: str) -> str | None:
    """The last value of key in the .env file, None when absent."""
    env_file = _env_file()
    if not os.path.exists(env_file):
        return None
    with open(env_file) as f:
        values = [line.split("=", 1)[1].strip() for line in f if line.lstrip().startswith(f"{key}=")]
    return values[-1] if values else None


def _net() -> tuple[str, str]:
    """The ARENA_NET value and where it came from: the .env file, or the shell's default."""
    value = _env_value("ARENA_NET")
    if value is not None:
        return value, _env_file()
    return os.environ.get("ARENA_NET", ""), "default"


def _container_net() -> str:
    """The ARENA_NET the running arena container was created with: a mode, 'down' or 'unknown'."""
    import json

    ids = features.containers("arena")
    if not ids:
        return "down"
    out = features.engine_output(["inspect", "--format", "{{json .Config.Env}}", ids[0]])
    if out is None:
        return "unknown"
    for entry in json.loads(out or "[]"):
        key, _, value = entry.partition("=")
        if key == "FASTRTPS_DEFAULT_PROFILES_FILE":
            return value.rsplit(".", 2)[-2] if value.endswith(".xml") else "unknown"
    return "unknown"


def _net_status(hint: bool) -> None:
    value, source = _net()
    mode = value if value in NET_MODES else "local"
    _row("net", mode)
    _row("source", source if source == "default" else _host_path(source))
    if value and value not in NET_MODES:
        _row("warning", f"{value!r} is not a recognized value, so it counts as local")
    container = "down" if features.in_container() else _container_net()
    if container == "unknown":
        _row("warning", "could not inspect the running container")
    elif container != "down" and container != mode:
        _row("warning", "the running container is stale, run 'source arena' to recreate it")
    if hint:
        print(f"set it with 'arena settings net {'|'.join(NET_MODES)}'")


def net(argv: list[str]) -> None:
    """Show or set the network scope of ROS 2.

    `arena settings net [local|lan]`. `local` keeps DDS discovery and data
    on loopback (default), `lan` uses every interface. Gazebo's own
    transport always stays on loopback. New shells pick the value up on
    `source arena`, running nodes keep theirs until restarted. `source
    arena` sets ROS_AUTOMATIC_DISCOVERY_RANGE=SUBNET so the DDS profile
    alone sets the scope.
    """
    if len(argv) > 1:
        raise CLIError(f"expected at most one argument, got {len(argv)}")
    if argv:
        if argv[0] not in NET_MODES:
            raise CLIError(f"unknown argument '{argv[0]}', expected one of: {', '.join(NET_MODES)}")
        _env_set("ARENA_NET", argv[0])
    _net_status(hint=not argv)


def _sim_status(hint: bool) -> None:
    defaults = _defaults()
    modes = _sim_modes()
    value = _env_value("ARENA_SIM")
    setting = value or "auto"
    installed = _reg_list()
    resolved = defaults.resolve_sim(setting, installed)
    _row("sim", resolved)
    _row("setting", setting)
    _row("source", "default" if value is None else _host_path(_env_file()))
    _row("installed", ", ".join(defaults.available_sims(installed)) or "none")
    if setting not in modes:
        _row("warning", f"{setting!r} is not a recognized value, so launches use auto")
    elif setting not in ("auto", resolved):
        _row("warning", f"{setting} is not installed, so launches use auto")
    if hint:
        print(f"set it with 'arena settings sim {'|'.join(modes)}'")


def sim(argv: list[str]) -> None:
    """Show or set the simulator a launch without sim:= uses.

    `arena settings sim [auto|dummy|SIMULATOR]`. `auto` (default) takes
    the first installed simulator in `SIM_ORDER` of arena_bringup's
    defaults.py, or dummy when none is installed. The bare command lists
    the simulators in that order. A simulator must be installed to be set.
    Every launch reads the value when it starts, no `source arena` needed.
    An explicit sim:= always wins.
    """
    if len(argv) > 1:
        raise CLIError(f"expected at most one argument, got {len(argv)}")
    if argv:
        modes = _sim_modes()
        if argv[0] not in modes:
            raise CLIError(f"unknown argument '{argv[0]}', expected one of: {', '.join(modes)}")
        if argv[0] in [s.value for s in _defaults().SIM_ORDER]:
            _reg_require(argv[0])
        _env_set("ARENA_SIM", argv[0])
    _sim_status(hint=not argv)


COMMANDS = {v.name: v for v in (make_verb("net", net, complete=Static(NET_MODES)), make_verb("sim", sim, complete=Static(_sim_modes)))}
