"""Ports: named textures with their kinds, what a monitor shows.

An input or output dataclass declares a field's kind on the field itself,
`velocity: Annotated[Texture, TextureKind.VELOCITY]`, and `ports_of` reads the ports from it. A
component that owns textures declares them once in its `ports` (`HasPorts`), and a stage joins its
components' ports under their names (`prefixed`), so each texture is written once."""

from dataclasses import dataclass, fields
from enum import IntEnum, auto
from functools import cache
from typing import Callable, Protocol, get_type_hints

from modules.gl import Texture, Texture3D

from .flow_util import FlowUtil


class TextureKind(IntEnum):
    """What a texture holds, which says how a monitor draws it: an image as the picture it is; a velocity
    (widths/s) and an acceleration (widths/s², shown as the velocity it adds in a reference step) by their
    direction, as arrows and as pushes (a 2D one in RG, a 3D one in RGB); a scalar centered on mid grey."""
    IMAGE = 0
    VELOCITY = auto()
    SCALAR = auto()
    ACCELERATION = auto()

    @property
    def vector(self) -> bool:
        """Whether it is a vector, drawn by its direction."""
        return self in (TextureKind.VELOCITY, TextureKind.ACCELERATION)


@dataclass(frozen=True)
class LightPlace:
    """Where the light a volume is lit by is, in the volume's texture coordinates (0 to 1, z 0 the front):
    a lamp's place, or for a sun where its light enters the box."""
    point: bool                             # a lamp, else a sun
    uvw: tuple[float, float, float]


@dataclass
class Port:
    """A named texture with its kind. `texture` is the `Fbo`, `SwapFbo`, `Tensor` or 3D texture object
    itself, never a buffer taken from it. A volume (`volume`) is always a 3D texture, so a view reads
    it with one sampler type; `extent` is its depth as a share of its width, the box a view draws it
    in, so a layer gap looks as deep as it is. `prepare`, when set, is what a view calls before it
    reads the texture: its owner brings it up to date there, once however many views ask, or learns
    that it is shown. `light`, on a volume, is where the light it is lit by is, drawn in its box."""
    texture: Texture | Texture3D
    kind: TextureKind
    extent: float = 0.0
    prepare: Callable[[], None] | None = None
    held: int = 0           # the channels holding its values, x first; 0 from its kind and format (`channels`)
    light: LightPlace | None = None

    @property
    def volume(self) -> bool:
        """Whether the texture is a volume, a 3D texture."""
        return isinstance(self.texture, Texture3D)

    @property
    def channels(self) -> int:
        """The channels that hold its values, x first, the ones a monitor draws: `held` where set (an RGB
        image in RGBA), else a vector's components, at most x, y and z (a 3D vector sits in RGBA, w
        unused), else every channel of its format."""
        if self.held > 0:
            return self.held
        stored: int = FlowUtil.get_num_channels(self.texture.internal_format)
        return min(stored, 3) if self.kind.vector else stored


class OnDemand:
    """Work done at most once between two `mark`s, and only when asked (`__call__`): a view's
    `Port.prepare` for a texture its owner refreshes only while something reads it."""

    def __init__(self, work: Callable[[], None]) -> None:
        self._work: Callable[[], None] = work
        self._stale: bool = True

    def mark(self) -> None:
        """What the work reads has changed: the next call does it again."""
        self._stale = True

    def __call__(self) -> None:
        if self._stale:
            self._stale = False
            self._work()


class HasPorts(Protocol):
    """A component that owns textures and names them, each with its kind."""
    @property
    def ports(self) -> dict[str, Port]: ...


@dataclass
class Ports:
    """Textures a stage owns itself, beside its components', as one more part."""
    ports: dict[str, Port]


def prefixed(prefix: str, ports: dict[str, Port]) -> dict[str, Port]:
    """`ports` named `<prefix>.<name>`; an empty prefix leaves the names as they are."""
    if not prefix:
        return dict(ports)
    return {f'{prefix}.{name}': port for name, port in ports.items()}


@cache
def _kinds(cls: type) -> tuple[tuple[str, TextureKind], ...]:
    """The fields of the dataclass `cls` that declare a kind, in their order."""
    hints = get_type_hints(cls, include_extras=True)
    declared: list[tuple[str, TextureKind]] = []
    for f in fields(cls):
        kind = next((m for m in getattr(hints[f.name], '__metadata__', ()) if isinstance(m, TextureKind)), None)
        if kind is not None:
            declared.append((f.name, kind))
    return tuple(declared)


def ports_of(values: object) -> dict[str, Port]:
    """The ports the dataclass `values` declares: each field with a kind whose value is set, by its
    name; a value that is no texture is a view holding one as `texture` (a `DepthView`)."""
    ports: dict[str, Port] = {}
    for name, kind in _kinds(type(values)):
        value = getattr(values, name)
        if value is not None and not isinstance(value, (Texture, Texture3D)):
            value = value.texture
        if value is not None:
            ports[name] = Port(value, kind)
    return ports
