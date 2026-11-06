"""Ports: named textures with their kinds, what a monitor shows.

An input or output dataclass declares a field's kind on the field itself,
`velocity: Annotated[Texture, TextureKind.VELOCITY]`, and `ports_of` reads the ports from it. A
component that owns textures declares them once in its `ports` (`HasPorts`), and a stage joins its
components' ports under their names (`prefixed`), so each texture is written once."""

from dataclasses import dataclass, fields
from enum import IntEnum, auto
from functools import cache
from typing import Protocol, get_type_hints

from modules.gl import Texture


class TextureKind(IntEnum):
    """How a texture is drawn when nothing else is asked: an image as the picture it is, a velocity
    as arrows (a 2D one in RG, a 3D one in RGB with z as a ring), a scalar as a colormap centered on mid grey."""
    IMAGE = 0
    VELOCITY = auto()
    SCALAR = auto()


@dataclass
class Port:
    """A named texture with its kind. `texture` is the `Fbo`, `SwapFbo` or `Tensor` object itself,
    never a buffer taken from it."""
    texture: Texture
    kind: TextureKind


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
        if value is not None and not isinstance(value, Texture):
            value = value.texture
        if value is not None:
            ports[name] = Port(value, kind)
    return ports
