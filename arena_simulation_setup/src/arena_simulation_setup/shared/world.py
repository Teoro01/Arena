from __future__ import annotations

import math
import typing

import attrs
import cattrs
import numpy as np

from arena_simulation_setup.tree.assets.Material import Material, MaterialIdentifier
from arena_simulation_setup.utils.cattrs import Parseable, Serializable, converter
from arena_simulation_setup.utils.geometry import Orientation, Pose, Position, Vector3

from .entities import Named
from .light_state import ObjectLightSettings, check_light_state
from .semantics import SemanticCfg, parse_semantics

CABIN_DOOR_INSET = 0.05
CABIN_DOOR_WIDTH = 0.05
CABIN_DOOR_THRESHOLD = 0.6


ActivationDistance = typing.NewType('ActivationDistance', tuple[float, float])


def _activation_distance_converter(x: float | typing.Sequence[float]) -> ActivationDistance:
    if isinstance(x, (int, float)):
        return ActivationDistance((float(x), float(x)))
    a, b = x
    return ActivationDistance((float(a), float(b)))


converter.register_structure_hook(ActivationDistance, lambda value, _: _activation_distance_converter(value))


def _call_mode_validator(_instance: object, _attribute: object, value: str) -> None:
    if value not in ('level', 'edge'):
        raise ValueError(f"elevator call_mode must be 'level' or 'edge', got {value!r}")


def _optional_capacity(value: object) -> int | None:
    if value is None:
        return None
    capacity = int(value)  # type: ignore[call-overload]
    if capacity < 1:
        raise ValueError(f'elevator capacity must be >= 1 or null, got {value!r}')
    return capacity


def _swing_direction_converter(value: object) -> int:
    direction = int(value)  # type: ignore[call-overload]
    if direction not in (1, -1):
        raise ValueError(f'door swing_direction must be 1 or -1, got {value!r}')
    return direction


@attrs.define
class Elevator(Named):
    position: Position = attrs.field(converter=Position.converter)
    size: list[float] = attrs.field(factory=lambda: [2.0, 2.0, 2.0])
    door_side: typing.Literal['+x', '-x', '+y', '-y'] | None = None
    material: MaterialIdentifier = attrs.field(converter=MaterialIdentifier.converter, default=Material.default('elevator'))
    destinations: list[str] = attrs.field(factory=list, converter=lambda value: [str(name).strip() for name in value])
    # must cover v_max * transition_time plus latency so the door is fully open before arrival
    activation_distance: float = 3.0
    transition_time: float = 1.0
    hold_time: float = 2.0
    travel_time: float = 3.0
    accept_outside_calls: bool = True
    recall_on: str | None = None
    # None = unlimited
    capacity: int | None = attrs.field(default=None, converter=_optional_capacity)
    call_mode: typing.Literal['level', 'edge'] = attrs.field(default='level', validator=_call_mode_validator)
    semantics: list[SemanticCfg] = attrs.field(factory=list, converter=parse_semantics)

    @property
    def destination(self) -> str:
        """The cabin a ride goes to, the first of the destinations."""
        return self.destinations[0] if self.destinations else ''

    def boarding_radius(self, agent_radius: float) -> float:
        """Largest distance from the cabin center at which a disc of agent_radius stays inside the cabin and clear of its door."""
        return min(self.size[0], self.size[1]) / 2.0 - CABIN_DOOR_INSET - CABIN_DOOR_WIDTH / 2.0 - agent_radius

    def cabin_corners(self) -> list[Position]:
        cx, cy = self.position.x, self.position.y
        hw, hh = self.size[0] / 2.0, self.size[1] / 2.0
        return [
            Position(cx - hw, cy - hh),
            Position(cx + hw, cy - hh),
            Position(cx + hw, cy + hh),
            Position(cx - hw, cy + hh),
        ]


@attrs.define
class Door(Named):
    start: Position = attrs.field(converter=Position.converter)
    end: Position = attrs.field(converter=Position.converter)
    kind: typing.Literal['sliding', 'hinged', 'teleport', 'sliding_top'] = 'sliding'
    width: float = 0.1
    height: float = attrs.field(default=2.0)
    material: MaterialIdentifier = attrs.field(converter=MaterialIdentifier.converter, default=Material.default('door'))
    activation_distance: ActivationDistance = attrs.field(
        converter=_activation_distance_converter,
        default=ActivationDistance((3.0, 3.0)),
    )
    transition_time: float = 1.0
    hold_time: float = 2.0
    # hinged only: +1 swings towards the left normal of start->end, -1 towards the right.
    swing_direction: int = attrs.field(default=1, converter=_swing_direction_converter)
    semantics: list[SemanticCfg] = attrs.field(factory=list, converter=parse_semantics)

    @classmethod
    def parse(cls, value: dict) -> Door:
        value = dict(value)
        if 'start' in value or 'end' in value:
            raise ValueError(f"door {value.get('name')!r} takes span: [[x, y], [x, y]], not start and end")
        span = value.pop('span', None)
        if not isinstance(span, (list, tuple)) or len(span) != 2:
            raise ValueError(f"door {value.get('name')!r} requires span: [[x, y], [x, y]]")
        if isinstance(value.get('activation_distance'), (int, float)):
            value['activation_distance'] = [value['activation_distance']] * 2
        return super().parse({**value, 'start': span[0], 'end': span[1]})

    def serialize(self) -> dict:
        result = super().serialize()
        result['span'] = [result.pop('start'), result.pop('end')]
        return result

    @property
    def corners(self) -> list[Position]:
        direction = np.array(list(self.end)) - np.array(list(self.start))
        direction = direction / np.linalg.norm(direction)
        perp = np.array([-direction[1], direction[0], 0])
        projected_half_width = Position(*(self.width / 2 * perp))
        return [
            self.start + projected_half_width,
            self.start - projected_half_width,
            self.end - projected_half_width,
            self.end + projected_half_width,
        ]


@attrs.define
class Floor(Named):
    pos: Position = attrs.field(converter=Position.converter)
    x_length: float = attrs.field(converter=float, default=20.0)
    y_length: float = attrs.field(converter=float, default=20.0)
    material: MaterialIdentifier = attrs.field(converter=MaterialIdentifier.converter, default=Material.default('floor'))


@attrs.define
class Ceiling(Named):
    pos: Position = attrs.field(converter=Position.converter)
    x_length: float = attrs.field(converter=float, default=20.0)
    y_length: float = attrs.field(converter=float, default=20.0)
    z: float = attrs.field(converter=float, default=2.0)
    cast_shadows: bool = attrs.field(default=False)
    material: MaterialIdentifier = attrs.field(converter=MaterialIdentifier.converter, default=Material.default('ceiling'))


@attrs.define
class Schedule(Named):
    """Standalone time-windowed semantic entity (kind = schedule), no geometry."""

    semantics: list[SemanticCfg] = attrs.field(factory=list, converter=parse_semantics)


@attrs.define
class Signal(Named):
    """Standalone cycling-phase semantic entity (kind = signal), no geometry, optional stop-line params `position` and `radius`."""

    semantics: list[SemanticCfg] = attrs.field(factory=list, converter=parse_semantics)

    def __attrs_post_init__(self) -> None:
        position = self._param('position')
        if position is not None:
            if not isinstance(position, (list, tuple)) or len(position) != 2 or not all(isinstance(v, (int, float)) for v in position):
                raise ValueError(f"signal {self.name!r} position must be [x, y], got {position!r}")
        radius = self._param('radius')
        if radius is not None and float(radius) <= 0.0:
            raise ValueError(f"signal {self.name!r} radius must be positive, got {radius!r}")

    def _param(self, key: str) -> typing.Any:  # noqa: ANN401 - untyped YAML semantics params
        for cfg in self.semantics:
            if key in cfg.params:
                return cfg.params[key]
        return None

    @property
    def stop_position(self) -> tuple[float, float] | None:
        """Stop-line position, None when not given."""
        position = self._param('position')
        return None if position is None else (float(position[0]), float(position[1]))

    @property
    def stop_radius(self) -> float | None:
        """Stop-line radius, None when not given."""
        radius = self._param('radius')
        return None if radius is None else float(radius)


def _optional_position(value: object) -> Position | None:
    if value is None:
        return None
    return Position.instance_or(Position.parse)(value)


@attrs.define
class Sound(Named):
    """Standalone audible semantic entity (kind = sound), placed in the world."""

    asset_id: str = attrs.field(converter=lambda value: str(value).strip())
    position: Position | None = attrs.field(converter=_optional_position, default=None)
    attach: str = attrs.field(converter=lambda value: str(value).strip(), default='')
    frame: str = attrs.field(converter=lambda value: str(value).strip().strip('/'), default='')
    offset: Position = attrs.field(converter=Position.converter, factory=lambda: Position(0.0, 0.0, 0.0))
    level: str = attrs.field(converter=lambda value: str(value).strip(), default='')
    loop: bool = True
    reference_distance_m: float = attrs.field(converter=float, default=1.0)
    semantics: list[SemanticCfg] = attrs.field(factory=list, converter=parse_semantics)

    def __attrs_post_init__(self) -> None:
        if not self.name or ':' in self.name:
            raise ValueError("sound name must be non-empty and contain no ':'")
        if not self.asset_id or ':' in self.asset_id:
            raise ValueError(f"sound {self.name!r} asset_id must be non-empty and contain no ':'")
        if sum((bool(self.attach), self.position is not None, bool(self.frame))) != 1:
            raise ValueError(f"sound {self.name!r} requires exactly one of position, attach or frame")
        if self.level and self.position is None:
            raise ValueError(f"sound {self.name!r} takes level only with a direct position")
        if self.reference_distance_m <= 0.0:
            raise ValueError(f"sound {self.name!r} reference_distance_m must be positive")


@attrs.frozen
class LightFixture:
    """Emitter shape of one fixture type."""

    shape: str
    x_length: float = 0.0
    y_length: float = 0.0
    radius: float = 0.0
    cone_deg: float = 180.0

    @property
    def ambient(self) -> bool:
        return self.shape in ('dome', 'distant')

    @property
    def directed(self) -> bool:
        return self.shape not in ('dome', 'sphere')

    @property
    def area(self) -> float:
        if self.shape == 'rect':
            return self.x_length * self.y_length
        if self.shape == 'disk':
            return math.pi * self.radius**2
        if self.shape == 'sphere':
            return 4.0 * math.pi * self.radius**2
        return 0.0


LIGHT_FIXTURES: dict[str, LightFixture] = {
    'panel': LightFixture(shape='rect', x_length=0.6, y_length=0.6),
    'tube': LightFixture(shape='rect', x_length=1.2, y_length=0.1),
    'downlight': LightFixture(shape='disk', radius=0.1),
    'spot': LightFixture(shape='disk', radius=0.05, cone_deg=40.0),
    'bulb': LightFixture(shape='sphere', radius=0.03),
    'dome': LightFixture(shape='dome'),
    'sun': LightFixture(shape='distant'),
}

_LIGHT_AXIS = Vector3(0.0, 0.0, -1.0)


def cct_to_rgb(cct_K: float) -> tuple[float, float, float]:
    """Blackbody color of a color temperature in kelvin, each channel in 0..1."""
    t = min(max(float(cct_K), 1000.0), 40000.0) / 100.0
    if t <= 66.0:
        red = 255.0
        green = 99.4708025861 * math.log(t) - 161.1195681661
    else:
        red = 329.698727446 * (t - 60.0) ** -0.1332047592
        green = 288.1221695283 * (t - 60.0) ** -0.0755148492
    if t >= 66.0:
        blue = 255.0
    elif t <= 19.0:
        blue = 0.0
    else:
        blue = 138.5177312231 * math.log(t - 10.0) - 305.0447927307
    return (
        min(max(red, 0.0), 255.0) / 255.0,
        min(max(green, 0.0), 255.0) / 255.0,
        min(max(blue, 0.0), 255.0) / 255.0,
    )


def _optional_float(value: float | None) -> float | None:
    if value is None:
        return None
    return float(value)


def _pose(value: object) -> Pose:
    return Pose.instance_or(lambda raw: converter.structure(raw, Pose))(value)


def _optional_pose(value: object) -> Pose | None:
    if value is None:
        return None
    return _pose(value)


@attrs.define
class Light(Named):
    """Standalone light semantic entity (kind = light): one fixture, an ambient light or a ceiling rig."""

    fixture: str = attrs.field(converter=lambda value: str(value).strip())
    pose: Pose | None = attrs.field(converter=_optional_pose, default=None)
    attach: str = attrs.field(converter=lambda value: str(value).strip(), default='')
    frame: str = attrs.field(converter=lambda value: str(value).strip().strip('/'), default='')
    offset: Pose = attrs.field(converter=_pose, factory=Pose)
    lumens: float | None = attrs.field(converter=_optional_float, default=None)
    lux: float | None = attrs.field(converter=_optional_float, default=None)
    cct_K: float = attrs.field(converter=float, default=6500.0)
    cone_deg: float | None = attrs.field(converter=_optional_float, default=None)
    cast_shadows: bool = False
    light_on: str = attrs.field(converter=lambda value: str(value).strip(), default='')
    lit: bool | None = None
    dim: float = attrs.field(converter=float, default=1.0)
    dead_fraction: float = attrs.field(converter=float, default=0.0)
    level: str = attrs.field(converter=lambda value: str(value).strip(), default='')
    intrinsic: bool = False
    owner: str = attrs.field(converter=lambda value: str(value).strip(), default='')
    link: str = attrs.field(converter=lambda value: str(value).strip(), default='')
    glow: str = attrs.field(converter=lambda value: str(value).strip(), default='')
    rig: bool = False
    fixtures: list[Position] = attrs.field(factory=list)
    fixture_ranks: list[float] = attrs.field(factory=list)

    def __attrs_post_init__(self) -> None:
        if not self.name or ':' in self.name:
            raise ValueError(f"light name {self.name!r} must be non-empty and contain no ':'")
        if self.fixture not in LIGHT_FIXTURES:
            raise ValueError(f'light {self.name!r} has unknown fixture {self.fixture!r}, expected one of {sorted(LIGHT_FIXTURES)}')
        ambient = self.spec.ambient
        anchors = sum((self.pose is not None and not ambient, bool(self.attach), bool(self.frame)))
        if self.offset != Pose() and not (self.attach or self.frame):
            raise ValueError(f'light {self.name!r} takes an offset only with attach or frame')
        if self.link and not (self.attach or self.owner):
            raise ValueError(f'light {self.name!r} takes a link only with attach or owner')
        for placed in (self.pose, self.offset):
            if placed is not None and not math.isclose(math.sqrt(sum(component * component for component in placed.orientation)), 1.0, abs_tol=1e-3):
                raise ValueError(f'light {self.name!r} orientation must be a unit quaternion')
        if ambient:
            if anchors:
                raise ValueError(f'light {self.name!r} ({self.fixture}) takes no attach or frame')
            if self.pose is not None and any(self.pose.position):
                raise ValueError(f'light {self.name!r} ({self.fixture}) takes no position, a pose only aims it')
            if self.rig:
                raise ValueError(f'light {self.name!r} ({self.fixture}) cannot be a rig')
            if self.lumens is not None:
                raise ValueError(f'light {self.name!r} ({self.fixture}) takes lux, not lumens')
            if self.lux is None or not self.lux >= 0.0:
                raise ValueError(f'light {self.name!r} ({self.fixture}) requires lux >= 0')
        else:
            if self.lux is not None:
                raise ValueError(f'light {self.name!r} ({self.fixture}) takes lumens, not lux')
            if self.lumens is None or not self.lumens >= 0.0:
                raise ValueError(f'light {self.name!r} ({self.fixture}) requires lumens >= 0')
            if self.rig:
                if anchors:
                    raise ValueError(f'rig light {self.name!r} takes no pose, attach or frame')
                if not self.fixtures:
                    raise ValueError(f'rig light {self.name!r} requires at least one fixture position')
                if len(self.fixture_ranks) != len(self.fixtures):
                    raise ValueError(f'rig light {self.name!r} requires one fixture rank per fixture position')
            elif anchors != 1:
                raise ValueError(f'light {self.name!r} requires exactly one of pose, attach or frame')
        if not self.rig and (self.fixtures or self.fixture_ranks):
            raise ValueError(f'light {self.name!r} takes fixture positions only as a rig')
        if not self.spec.directed and self._placement.orientation != Orientation.identity():
            raise ValueError(f'light {self.name!r} ({self.fixture}) takes no orientation')
        if self.cone_deg is not None and (self.fixture != 'spot' or not 0.0 < self.cone_deg < 180.0):
            raise ValueError(f'light {self.name!r} takes cone_deg only as a spot, within 0..180 exclusive')
        check_light_state(f'light {self.name!r}', self.cct_K, self.dim, self.dead_fraction, self.light_on, self.lit)
        if self.dead_fraction and not self.rig:
            raise ValueError(f'light {self.name!r} takes dead_fraction only as a rig')

    @property
    def _placement(self) -> Pose:
        return self.pose if self.pose is not None else self.offset

    @property
    def position(self) -> Position:
        """Where the fixture sits: in the level frame, or in the frame of its anchor while it sits on one."""
        return self._placement.position

    @property
    def direction(self) -> tuple[float, float, float]:
        """Unit vector the fixture emits along, the local -z axis of its pose, or of its offset while it sits on an anchor."""
        along = self._placement.orientation * _LIGHT_AXIS
        return (along.x, along.y, along.z)

    @property
    def spec(self) -> LightFixture:
        spec = LIGHT_FIXTURES[self.fixture]
        return spec if self.cone_deg is None else attrs.evolve(spec, cone_deg=self.cone_deg)

    @property
    def semantics(self) -> list[SemanticCfg]:
        lit = None if self.light_on else (True if self.lit is None else self.lit)
        result = [
            SemanticCfg(role='predicate', name='lit', value=lit, params={'light_on': self.light_on} if self.light_on else {}),
            SemanticCfg(role='state', name='dim', value=self.dim),
        ]
        if self.rig:
            result.append(SemanticCfg(role='state', name='dead_fraction', value=self.dead_fraction))
        return result

    def alive(self, dead_fraction: float) -> list[bool]:
        """Per rig fixture, whether it survives the given dead fraction."""
        return [rank >= dead_fraction for rank in self.fixture_ranks]


_OBJECT_LIGHT_KEYS = frozenset({'name', 'fixture', 'pose', 'lumens', 'cct_K', 'cone_deg', 'glow', 'link'})


def object_light(owner: str, entry: dict, settings: ObjectLightSettings | None) -> Light:
    """Light an object annotation entry declares, named after its owning entity and offset in the object frame."""
    unknown = set(entry) - _OBJECT_LIGHT_KEYS
    if unknown:
        raise ValueError(f'object light of {owner!r} has unknown keys {sorted(unknown)}')
    fixture = str(entry.get('fixture', 'bulb'))
    if fixture not in LIGHT_FIXTURES or LIGHT_FIXTURES[fixture].ambient:
        raise ValueError(f'object light of {owner!r} has fixture {fixture!r}, expected one of {sorted(name for name, spec in LIGHT_FIXTURES.items() if not spec.ambient)}')
    offset = _pose(entry['pose']) if 'pose' in entry else Pose()
    settings = settings if settings is not None else ObjectLightSettings()
    return Light(
        name=f"{owner}_{entry.get('name', 'light')}",
        fixture=fixture,
        attach=owner,
        offset=offset if LIGHT_FIXTURES[fixture].directed else Pose(position=offset.position),
        lumens=entry.get('lumens'),
        cct_K=entry.get('cct_K', 6500.0),
        cone_deg=entry.get('cone_deg') if fixture == 'spot' else None,
        light_on=settings.light_on,
        lit=settings.lit,
        dim=settings.dim,
        intrinsic=True,
        glow=entry.get('glow') or '',
        link=entry.get('link') or '',
    )


@attrs.define
class CeilingLights(Parseable, Serializable):
    """Zone ceiling rig, expanded into one rig Light per zone."""

    fixture: str = attrs.field(converter=lambda value: str(value).strip(), default='panel')
    spacing: float = attrs.field(converter=float, default=2.4)
    lumens: float = attrs.field(converter=float, default=3600.0)
    cct_K: float = attrs.field(converter=float, default=6500.0)
    cast_shadows: bool = False
    light_on: str = attrs.field(converter=lambda value: str(value).strip(), default='')
    lit: bool | None = None
    dim: float = attrs.field(converter=float, default=1.0)
    dead_fraction: float = attrs.field(converter=float, default=0.0)

    def __attrs_post_init__(self) -> None:
        if self.fixture not in LIGHT_FIXTURES or LIGHT_FIXTURES[self.fixture].ambient:
            raise ValueError(f'ceiling_lights fixture {self.fixture!r} must be one of {sorted(name for name, spec in LIGHT_FIXTURES.items() if not spec.ambient)}')
        if not self.spacing > 0.0:
            raise ValueError('ceiling_lights spacing must be positive')
        if not self.lumens >= 0.0:
            raise ValueError('ceiling_lights requires lumens >= 0')
        check_light_state('ceiling_lights', self.cct_K, self.dim, self.dead_fraction, self.light_on, self.lit)

    def serialize(self) -> dict:
        return cattrs.gen.make_dict_unstructure_fn(type(self), converter, _cattrs_omit_if_default=True)(self)
