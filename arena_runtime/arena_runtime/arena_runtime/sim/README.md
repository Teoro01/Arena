# task_generator sim interface

`BaseSim` and the four sub-interfaces it combines. Implementations are
registered in `SimulatorRegistry` and instantiated by key at runtime. See also
[human simulator](../../../../task_generator/task_generator/simulators/human/README.md) for the pedestrian-logic counterpart
(`BaseHumanSimulator`).

## Sub-interfaces

Defined in [`_interface.py`](_interface.py):

### `ObstacleITF`

[`_interface.py:20`](_interface.py#L20)

| Abstract method | Signature |
| --- | --- |
| `obstacle_spawn` | `(Sequence[Obstacle]) -> Sequence[bool]` |
| `obstacle_move` | `(Sequence[Obstacle]) -> Sequence[bool]` |
| `obstacle_delete` | `(Sequence[Obstacle]) -> Sequence[bool]` |

### `PedestrianITF`

[`_interface.py:38`](_interface.py#L38)

| Abstract method | Signature |
| --- | --- |
| `pedestrian_spawn` | `(Sequence[DynamicObstacle]) -> Sequence[bool]` |
| `pedestrian_move` | `(Sequence[DynamicObstacle]) -> Sequence[bool]` |
| `pedestrian_delete` | `(Sequence[DynamicObstacle]) -> Sequence[bool]` |
| `pedestrian_update` | `(Pedestrians) -> Sequence[bool]` |

### `RobotITF`

[`_interface.py:61`](_interface.py#L61)

| Abstract method | Signature |
| --- | --- |
| `robot_spawn` | `(Sequence[Robot]) -> Sequence[bool]` |
| `robot_move` | `(Sequence[Robot]) -> Sequence[bool]` |
| `robot_delete` | `(Sequence[Robot]) -> Sequence[bool]` |

### `WorldITF`

[`_interface.py:116`](_interface.py#L116)

| Abstract method | Signature |
| --- | --- |
| `spawn_walls` | `(Sequence[Wall]) -> bool` |
| `spawn_floors` | `(Sequence[Floor]) -> bool` |
| `remove_world` | `() -> bool` (default raises `NotImplementedError`) |

### `MechanismITF`

[`_interface.py:138`](_interface.py#L138): door and elevator orchestration. Provides concrete default implementations driven by an internal sim-time tick loop ([`_mechanism_shim.py`](_mechanism_shim.py)) on top of four box/robot primitives. Any simulator that implements the primitives gets door animation, elevator rides, and ped/robot teleport for free; simulators with native support can override the four top-level methods.

| Default method | Signature | Purpose |
| --- | --- | --- |
| `spawn_doors` | `(Sequence[Door]) -> bool` | spawn door geometry + register runtime; starts the tick loop |
| `remove_doors` | `(Sequence[str]) -> bool` | delete door geometry by name |
| `spawn_elevators` | `(Sequence[Elevator]) -> bool` | spawn cabin walls + synthesized door per elevator |
| `remove_elevators` | `(Sequence[str]) -> bool` | delete cabin geometry + synthesized door by name |
| `stop_mechanisms` | `() -> None` | cancel the tick loop (call on shutdown) |
| `attach_human_simulator` | `(HumanSimulator) -> None` | bind the human-sim Protocol that supplies ped positions + ped teleport |
| `spawn_objects` | `(Sequence[Obstacle]) -> Sequence[bool]` | spawn obstacles, an articulated one as its link obstacles (see [Articulated objects](#articulated-objects)) |
| `move_objects` | `(Sequence[Obstacle]) -> Sequence[bool]` | move obstacles, an articulated one reset to the initial values of its `joints` |
| `delete_objects` | `(Sequence[Obstacle]) -> Sequence[bool]` | delete obstacles, an articulated one with its link obstacles and semantics |

| Primitive (override required for default behavior) | Signature |
| --- | --- |
| `spawn_box` | `(name, size, pose, material=None) -> bool` |
| `move_box` | `(name, pose) -> bool` |
| `delete_box` | `(name) -> bool` |
| `set_robot_pose` | `(sim_path, pose) -> bool` |

Robot tracking is not a primitive to override: call `_register_agent_robot(robot, model_params)` on spawn and `_forget_agent_robot(sim_path)` on remove, and the base class serves `robot_discs()` / `robot_pose()` from TF.

Lights follow the same pattern. `spawn_lights(lights)` and `remove_lights()` keep the light table, and every tick pushes each `light` semantic entity whose output changed, after moving frame-anchored lights that moved more than 2 cm or 0.02 rad. A simulator renders lights by overriding four no-op hooks:

| Light hook | Signature | Purpose |
| --- | --- | --- |
| `_spawn_lights` | `(Sequence[Light]) -> bool` | create the light entities, dome and sun replacing the default lighting |
| `_remove_lights` | `(Sequence[Light]) -> bool` | delete them and restore the default lighting |
| `_apply_light` | `(light, level, alive) -> None` | render at `level` in 0..1 with the per-fixture alive mask |
| `_move_light` | `(light, position, yaw) -> None` | place a frame-anchored light, its direction turned by `yaw` |

A light with `glow` also makes that material of its `owner` entity emit the light's color at `level`: Gazebo recolors the model's `glow_<material>` visual over `/world/default/material_color`, Isaac sets the material's emissive color on the session layer, MuJoCo sets the emission and color of the placement's own glow materials before each render.

The `HumanSimulator` Protocol the shim consumes from the attached human-sim:

| Method | Signature |
| --- | --- |
| `pedestrian_discs` | `() -> Iterable[tuple[str, tuple[float, float], float]]` (sync, ground truth: name, xy, radius) |
| `pedestrian_teleport` | `(Mapping[str, tuple[float, float]]) -> bool` (async) |
| `elevator_doors_changed` | `(Mapping[str, bool]) -> None` (async, cabin name to door open) |

Optional, looked up with `getattr` so human sims without it keep working:

| Method | Signature |
| --- | --- |
| `set_mechanism_walls` | `(key: str, segments: list[((x0, y0), (x1, y1))] \| None) -> None` (map frame, `None` removes the set) |

The shim calls it with `elevator:<name>` (back and side cabin walls) at elevator spawn and `None` at removal,
and with `door:<name>` (the door segment) only when a door changes between "cannot open for pedestrians"
(gate effectively locked with no authorized agent near, or `activation_distance <= 0` with no pressed plate
driving it) and "can open".

The attachment happens in `BaseHumanSimulator.__init__` (see [human simulator](../../../../task_generator/task_generator/simulators/human/README.md)). Without an attached human-sim, the shim still drives doors against `robot_discs` alone and logs a no-op for ped teleports.

Pedestrians ride like robots: the shim counts a ped inside a cabin footprint as an occupant, and when the cabin door has closed behind it the ride hands the destinations to `pedestrian_teleport`. An empty cabin opens for a robot or ped outside within `activation_distance` of its door, a cabin with an occupant only for one within `CABIN_DOOR_THRESHOLD` (0.6 m), so riders queuing for the next trip do not hold a loaded cabin. The shim also tells the human-sim which cabin doors are passable: each tick it collects the cabins whose door turned `open` or left `open` since the last report, starting from closed at spawn and including the close an episode reset causes, and sends them in one `elevator_doors_changed` call, nothing on ticks without a change.

#### Shim behaviour

- Door kinds: `sliding` (slides along start->end), `sliding_top` (lifts), `teleport` (drops away) and
  `hinged` (rotates about `start` by `swing_direction * progress * 90 deg`). Sliding and hinged slabs are tested
  against agent discs: an opening step holds when the slab would enter an agent (the swing arc of a hinged
  door, the wall pocket past the end of a sliding one), so an agent pressed against a half-open sliding slab
  does not stop it, and a blocked close re-triggers the door and sets `close_blocked` (semantic
  `blocked_close`) until the remaining path is clear.
- Elevators: the cabins linked through their destinations form a group served by one car, which starts at
  the group's first cabin by name and returns there on reset (`_CarState` per cabin: `present`, `departing`,
  `absent`, `arriving`). Only the cabin holding the car opens its door, so at most one door per group is open,
  and a tick with two is reported as an incident. A call at a cabin without the car (an agent in the call
  radius, or anyone shut inside) waits in `call_since`, and the idle car fetches the oldest call with an empty
  trip. `destination` may hold several realized cabin names (comma separated). A loaded departure goes to
  `_ElevatorRuntime.current_destination()`: the written `target` if set, else round-robin (one step per
  departure). `capacity` (live occupants) suppresses outside calls when full.
  `call_mode: edge` makes outside calls edge-triggered (newcomer in the call radius, or a shut idle cabin),
  so agents waiting in the lobby do not hold the door open; `level` (default) re-fires every tick.
  Door holds are bounded: outside callers keep a loaded car open at most `hold_time` after its first boarder
  entered, an empty car stops answering them once a call elsewhere has waited `hold_time`, and arrivals hold
  the door at most `TELEPORT_GRACE` (10 s sim time) while a call waits, riding along if still inside when
  the car leaves. A recall fetches the car to the recalled cabin and holds it there open.
- Failed rides: a refused/raising robot or ped teleport releases the riders at once (source unsealed, the
  destination stops waiting), and the riders left in the source cabin call the car back. A silent failure is
  released after `TELEPORT_GRACE` (10 s sim time) with a warning.

#### Articulated objects

The human simulator places every static obstacle through `spawn_objects`,
`move_objects` and `delete_objects` ([`_articulation.py`](_articulation.py)),
which hand a plain obstacle unchanged to the simulator's `obstacle_*` verb.
An obstacle whose object annotation carries `articulation:` and that has link
bundles (`links/<link>/` under the object) expands into one static obstacle
per link bundle, named and sim-pathed `<obstacle>__<link>`, its model the link
bundle, posed at `obstacle.pose * link_pose` with the link translation scaled
componentwise by `obstacle.scale` and the scale kept on the link. Each call
sends all its obstacles to the simulator in one batch. Links without meshes
have no bundle and no obstacle. MuJoCo spawns a link from the bundle's SDF, as
it does any object, not from the Y-up OBJ the bundle carries for the URDF.
Every simulator thus moves joints through the
`obstacle_move` it already has, MuJoCo and Isaac keep the box center offset of
a box-spawned obstacle when it moves, flatsim places each outline at the z
span of the link box through its full pose.

Each articulated obstacle keeps initial (`articulation.resolve(obstacle.joints)`),
current and target joint values. A `move_objects` call, which is how the human
simulator re-places a known obstacle at an episode reset, takes the new pose and
`joints`, resets current and target to the new initial values and moves every
link. On the mechanism tick each joint steps toward its target at its speed and
the links whose pose changed go to `obstacle_move` in one batch per tick. A step
that would put a moving link's footprint (its link box through the object and
link pose, projected to xy) onto a robot or pedestrian disc holds the whole
object for that tick and marks it `blocked`, it resumes once clear. The
`articulated` semantic kind is attached under the env-prefixed obstacle name on
spawn and detached on delete, its `reset()` returns the joints to their initial
values and the next tick places the links, see
[AUTHORING.md](../../../../arena_simulation_setup/AUTHORING.md#articulated-objects)
for its fields.

### `SimLifecycle`

[`_interface.py:18`](_interface.py#L18): process-singleton host counterpart to `BaseSim`. Owned by `arena_node` (one instance per process), drives sim-wide pause/unpause and namespace cleanup that cuts across env-scoped simulators.

| Abstract method | Purpose |
| --- | --- |
| `pause` / `unpause` | toggle the underlying sim clock |
| `cleanup_namespace(prefix)` | delete all entities under `prefix`, return count removed |
| `ensure_ready` | block until the sim's services are reachable |

Registered in `LifecycleRegistry` alongside `SimulatorRegistry` ([`__init__.py`](__init__.py)).

`arena_node` wraps the registered `SimLifecycle` in a liveness-accounting
proxy: two consecutive `pause`/`unpause` failures (`False` or raised) or step
failures mark the sim dead, published latched on `state/sim`
([`SimState.msg`](../../../arena_runtime_msgs/msg/SimState.msg)), see the
[arena_runtime README](../README.md) topic table.
Adapters signal a dead transport (a timed-out service call, not a genuine
rejection) by raising [`SimUnavailable`](_interface.py) instead of returning
`False`. `cleanup_namespace` should let it propagate. `arena_node` never
retries or recovers a dead sim on its own.

## `BaseSim`

[`__init__.py:17`](__init__.py#L17)

```python
class BaseSim(NodeInterface, ObstacleITF, PedestrianITF, RobotITF, WorldITF, MechanismITF, abc.ABC):
```

Overridable hooks (concrete default no-ops):

| Method | Purpose |
| --- | --- |
| `before_reset_episode()` | episode-boundary hook before every reset, for sim work that must not ride the mid-episode pause path; default no-op |
| `after_reset_episode()` | episode-boundary hook after every reset; default no-op |
| `step(n=1)` | advance simulation by `n` ticks; default no-op returns `True` |

## Lockstep

`SimLifecycle.step_seconds(seconds) -> float` advances a held sim by an exact
sim-time delta, quantized to whole physics steps (`physics_dt` rosparam,
default 0.0333 matching `empty.sdf`). The base raises `NotImplementedError`.
Gazebo sends one `ControlWorld` request with `pause=True` and `multi_step=n`
(both fields in the same message, else gz free-runs). A step of more than one
tick first raises gz `real_time_factor` so the limiter does not pace it, and
the next unpause restores it. A single-tick step leaves the factor alone. Isaac calls
`/isaac/StepSimulationN` and awaits `/clock` reaching the projected target.
MuJoCo calls `/mujoco/Step` with `seconds / 0.002` physics steps (the server
publishes `/clock` and drains its sensor and control pumps every 20 ms of sim
time inside the call) and awaits `/clock` the same way. Its pause, unpause
and namespace cleanup calls are idempotent and are resent every 10 s until
answered, so a request or reply lost in transit cannot hold an env forever.
The dummy host adds the delta to its synthetic clock, so `sim:=dummy` runs the
whole control plane without a physics engine. The flatsim host sends
`control_world` with `multi_step` to its server and awaits `/clock` reaching
the time the server returns (see [flatsim backend](#flatsim-backend)). The
dummy host is what
[`tests/ros/`](../../tests/ros/) exercises, the gate math itself is
[`lockstep_gates.py`](../lockstep_gates.py) and has ROS-free unit tests.

`arena_node` exposes it as `sim_lifecycle/step` (valid only while holds are
active and no unpause window is open), plus the lockstep control plane
driving the [`LockstepScheduler`](../lockstep.py). Every adapter advances at
least one physics step for any positive delta, so a delta below half a step
is exactly one step. The task generator steps that way to teleport robots
during a reset hold (see the [manager README](../../../../task_generator/task_generator/manager/README.md)).

Components self-register their data channels via
`sim_lifecycle/lockstep/register` (LockstepRegister: `caller`, `env`,
`channels[]` of `{name, topic, type, period_s, hard}`. Re-registering under
a `caller` replaces its set, an empty `channels[]` clears it, per-env
registrations drop when their env despawns, and registrations merge into a
running lockstep live. The arena_humansim adapter self-registers `engine`
and `peds` (both hard). Its roster never trails the engine: every `arena_peds`
is stamped no earlier than the last engine frame, and removing all agents
publishes an empty roster at that stamp, so `peds` is covered whenever
`engine` is. The hunav adapter self-registers `roster` (hard).
The arena_auditory `array_renderer` registers `audio/<robot>` (hard, one
render block, 20 ms for `auditory.array.spec:=four_mic`) and renders off
`/clock` for the same reason as the humansim engine below.
`robot.hearing:=srp|seld` also registers `hearing/<robot>` per fleet robot
(hard, one front-end hop, `LockstepHeartbeat` stamped with the audio
consumed), and the arena_hearing stack registers `belief/<robot>` and
`policy/<robot>` per fleet robot (hard, their publish periods) so the sim
never runs ahead of a belief grid or a speed mask.
The arena_robots task_server registers per-robot beats only while a goal is
active: `nav/<robot>` (hard, pulsed per cmd_vel) during goto_pose for both
nav2 (one controller period) and the goal-window passthrough stacks whose
control loop Arena does not own (rosnav_rl/external, fixed 0.25 s liveness
period, action held open until arrival), `reach/<robot>` and
`gesture/<robot>` (hard, pulsed per JTC controller_state) during reach_pose
/ play_gesture. The arena_planners bridge (`mobile:=drl`) registers
`planner/<robot>` (hard, no keepalive) for the lifetime of its run loop,
with the period snapped to a whole number of physics steps and every pulse
stamped with the observation time its action was computed for, so a
lockstep run steps exactly one planner tick per action: obs at the frozen
boundary, action, step. The `none`, `manual`, and `test_collision` bringups
keep fire-and-forget goto_pose and stay ungated. `arena cam ... lockstep=true` recording registers `cam`
(hard, 1/fps) for the take when a lockstep run is active and captures each
frame at the tick the run froze on for it before beating that stamp, so the
recording is gated like any other producer and frame-exact at any target rtf. With no
run active it holds and steps the sim itself. Beats republish
`LockstepHeartbeat` coverage stamps, and a wall-clock grace watchdog publishes
forward keepalives through legitimately silent phases (MoveIt planning,
nav2 recovery behaviors), pacing the sim at wall rate so a producer that
itself needs sim time to pass cannot freeze the sim. Training-mode robots have no task_server and stay
outside the gate set.
Producers publish with `header.stamp` reflecting the sim time they have
actually covered. A hard channel is satisfied at sim time `T` when its
latest stamp is newer than `T - period_s`, compared at the clock's
nanosecond resolution: the gate asks whether the producer has covered the
clock, never whether it landed on a window boundary, so a producer whose
own grid is phased against the ledger's cannot deadlock the run. Windows
only pick the step sizes the scheduler takes. A producer driven by the sim
clock registers `period_s` equal to its own tick length, which paces the
sim to the producer's true coverage. An elapsed-based rate loop registers
a period strictly above its interval. The arena_humansim engine ticks off
`/clock` itself rather than an rcl timer, so a clock the gate is holding
still owes it the tick the gate is waiting for.

`sim_lifecycle/lockstep/start` (LockstepStart: `target_rtf`, `ungated`)
starts a run or reconfigures one in place: `target_rtf` of 0 is unpaced
while gates hold the pace and wall rate while no channel is registered,
`ungated` free-steps ignoring every gate. Hard channels freeze-and-wait: the
scheduler will not advance past a tick until every hard channel has a
`header.stamp` covering it, so the weakest hard producer sets the pace by
design. `soft` channels are observed and reported only. The scheduler also
idles while a foreign hold is open (episode resets) and rebases its window
grid to whatever sim time passed meanwhile. An unpause window never unpauses
the sim during a run: the scheduler keeps stepping for it, also under a
foreign hold or `pause`, gated on every hard channel except those registered
in the window holder's env.
`sim_lifecycle/lockstep/stop` ends the run and restores the pre-run
pause state and gz `real_time_factor`, responding as soon as the hold is
released (the guard against a draining gz chunk re-pausing the world runs
in the background). `sim_lifecycle/lockstep/pause` /
`sim_lifecycle/lockstep/resume` freeze and unfreeze the clock within a run
without ending it. Status is the latched `/arena/state/lockstep` topic
(LockstepStatus): `active`, `paused`, `ungated`, `target_rtf`,
`measured_rtf`, `tick`, `registrations`, `waiting_on`, `arrived`, and the
run's cumulative counters: `gated_ticks` (ticks that waited on a hard
channel) and `counts` (per channel topic, the ticks it was `due` on and
the messages `received`), both reset when a run starts.

`lockstep.autostart`/`lockstep.channels`/`lockstep.target_rtf`/
`lockstep.paused` params are read once at bringup (launch args
`lockstep:=true lockstep.channels:="..."`), registering the channels under
caller `launch`. Channel specs are `name|topic|type|period_s|hard-or-soft`
with `{env}` expanded per env. CLI: `arena lockstep run [rtf:=N] [ungated]`
(blocks, ctrl-c pauses), `on [rtf:=N] [ungated]`, `off`, `pause`, `resume`,
`status`, and `arena lockstep gate [engine|peds|<spec> ...]` to register
extra channels under caller `cli`.

## Sim-paused invariant

The sim is paused for the entire body of `Task._reset_episode`. Only
node-discovery and lifecycle signals are observable while the sim is paused;
tf, costmap, and sim-clock topics are not advancing, except for the single
physics steps a robot teleport takes through `sim_lifecycle/step` until TF
shows the robot landed. `switch_controller` is the
one controller_manager call that needs the sim to tick (on both gz_ros2_control
and Isaac's `ros2_control_node`, whose update loops follow sim time), so
controller activation lives inside the robot bring-up unpause window.

## Registered implementations

[`__init__.py:57`](__init__.py#L57): `SimulatorRegistry` maps
`Constants.SimSimulator` keys to async factory functions:

| Key | Class | File | Notes |
| --- | --- | --- | --- |
| `dummy` | `DummySimulator` | [`dummy_simulator.py`](dummy_simulator.py) | no-op entity verbs; `DummyHost` publishes a synthetic `/clock` (wall time minus paused time plus stepped time, also while paused) and supports `step_seconds` |
| `flatsim` | `FlatsimSimulator` | [`flatsim_simulator/flatsim_simulator.py`](flatsim_simulator/flatsim_simulator.py) | client of the flatsim server (2D Box2D, no renderer), one world for every env. `FlatsimHost` pauses and steps the server, which owns `/clock`. Optional feature |
| `gazebo` | `GazeboSimulator` | [`gazebo_simulator/gazebo_simulator.py`](gazebo_simulator/gazebo_simulator.py) | Gazebo (Ignition) via gz-transport. Discovery runs on a port pair per ROS domain (`GZ_DISCOVERY_MSG_PORT` = 5500 + 2 x `ROS_DOMAIN_ID`, `GZ_DISCOVERY_SRV_PORT` one above, below the DDS port blocks), set by `source arena`, by the `arena launch` / `arena env` supervisor for every process it spawns, and by `arena_runtime.launch.py` for everything it starts, so forks and lanes never receive each other's gz discovery. A shell that changes `ROS_DOMAIN_ID` after sourcing re-exports both before calling the `gz` CLI |
| `isaac` | `IsaacSimulator` | [`isaac_simulator.py`](isaac_simulator.py) | Isaac Sim integration |
| `mujoco` | `MujocoSimulator` | [`mujoco_simulator.py`](mujoco_simulator.py) | MuJoCo server over the `/mujoco` services, see [arena_mujoco](../../../../arena_mujoco/README.md) |

Flatland and Unity have stubs (commented out) in `__init__.py`; they are not
active. The active simulator is selected by `node.conf.Arena.SIM` and
instantiated via `SimulatorRegistry.get(key, **kwargs)`.

## flatsim backend

`sim:=flatsim` runs the [flatsim](../../../../flatsim/README.md) server, one
process for the whole runtime, started by `sim.launch.py` through
`arena feature flatsim launch`. It is an optional feature
(`arena feature flatsim install`): arena_runtime does not depend on it, and
selecting the sim without it fails with the install hint. The server owns
`/clock`, steps one world that holds every env at its realizer offset, as
Gazebo and Isaac do, and publishes each robot's
`map -> odom`, `odom -> base`, Odometry, zero joint states and sensor
messages. Its interface is documented in the flatsim README, the notes here
are what Arena sends it.

- [`FlatsimHost`](flatsim_simulator/flatsim_simulator.py): `pause` and `unpause` are
  `control_world` calls. `step_seconds` sends `multi_step` in chunks of at
  most one sim second, the server takes the steps before it answers, and the
  call returns once arena_node's own clock carries the returned time.
  `cleanup_namespace` deletes every entity, floor and robot whose name starts
  with the env prefix `env_<id>/`.
- [`FlatsimSimulator`](flatsim_simulator/flatsim_simulator.py): the per-env client. It
  keeps no world of its own. Spawns and deletes are service calls, pedestrians
  go out on `/flatsim/discs` and door and cabin moves on `/flatsim/poses`
  without waiting for an answer, as Gazebo's `move_box` does, since the
  mechanism moves every door on every tick.
- Geometry: wall segments and obstacle footprints are static outlines, sent in
  one call per spawn. Doors are boxes behind the four mechanism primitives,
  pedestrians are discs fed by `pedestrian_update`.
- Levels: floors define level slabs. A robot collides and casts rays only
  against the level containing its z. The shipped multi-level worlds lay their
  levels side by side in xy at one z, so an elevator ride there is an xy
  teleport within one level. An entity that ends within 0.05 m above its level
  floor, such as an elevator cabin floor, is no obstacle.
- Robots: body from the mobile cap footprint, driven by the Twist on
  `<robot>/cmd_vel` through a [drive model](flatsim_simulator/planar.py) read from
  the `planar` block of the robot's `caps/mobile.yaml`, the same keys for every
  robot: per-axis (linear, lateral, angular) velocity bounds, acceleration and
  gain, a first-order `lag`, a `traction` cap on planar acceleration, the
  `pivot` x the body turns about and the command `timeout`. A diff drive has
  lateral bounds `[0, 0]`. A robot without the block is unbounded and moves
  sideways only when `is_holonomic`. No controller_manager runs. The adapter
  launches `robot_state_publisher`, the server gets the URDF and the tf topic
  of the env.
- Odometry: the pose is the exact body pose. The twist is the wheel reading,
  the body twist divided by the gain and taken at the pivot, so it tracks the
  command like a wheel-odometry twist does while the IMU reports the body rate.
- `sim.flatsim.dynamics:=rigid|kinematic`: a runtime argument, the same for
  every env. `rigid` (default) lets walls and obstacles stop or push the robot
  while pedestrians touch it without pushing, `kinematic` lets it pass through.
  Collision events come from the task generator's collision tracker in both.
- Sensors: `laserscan`, `pointcloud`, `imu` and `contact` specs, each at its
  own update rate rounded to whole physics steps, and every sensor publishes
  once more when an episode reset ends (`republish`). Lidar and camera geometry
  comes from the URDF `<sensor>` block backing the spec (the mobile cap's
  `laser` block when there is none), the mount from the joint chain. Rays
  respect height: a beam hits an entity only where the entity's z extent
  holds it, so one cast per azimuth yields the scan and every ring of a 3D
  lidar cloud. A downward beam that meets nothing first returns its level
  floor, so a costmap raytraces through the voxels it crosses as it would on a
  real floor. Depth and RGB-D cameras yield a decimated cloud, images are not
  produced. The IMU reports yaw, yaw rate and the body acceleration of the
  last step plus gravity. Contacts list the entities the footprint touches.
- Lockstep: no channel is registered, so the other producers set the strides.
  A step is complete when the service answers, every message of it is already
  published.

## Adding a new simulator

1. Subclass `BaseSim`; implement all abstract methods from the four
   sub-interfaces. Define `SIM_NAME` (`ClassVar[str]`), consumed by
   `MechanismITF.__init__` to tag semantics with the simulator key.
   `before_reset_episode`/`after_reset_episode`/`step` default to no-ops,
   override only if the simulator needs episode-boundary or stepped-sim work.
2. Implement the four `MechanismITF` primitives (`spawn_box`, `move_box`,
   `delete_box`, `set_robot_pose`) to get door + elevator animation out of
   the box, and call `_register_agent_robot` on spawn / `_forget_agent_robot`
   on remove so the door gate can see the simulator's robots. Override
   `spawn_doors`/`remove_doors`/`spawn_elevators`/`remove_elevators` only if
   the simulator has native door support that supersedes the shim, and call
   `await self.stop_mechanisms()` from `shutdown` if you use the defaults.
3. Register a lazy async factory:

```python
@SimulatorRegistry.register(Constants.SimSimulator.MY_SIM)
async def lazy_mysim(**kwargs: typing.Unpack[SimKwargs]) -> BaseSim:
    from .my_sim import MySimulator
    return await MySimulator.create(**kwargs)
```

4. Add `MY_SIM = "my_sim"` to `Constants.SimSimulator`.
