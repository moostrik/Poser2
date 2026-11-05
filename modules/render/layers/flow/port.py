"""A stage's ports: its named textures with their kinds, what a monitor shows."""

from dataclasses import dataclass
from enum import IntEnum, auto

from modules.gl import Texture


class TextureKind(IntEnum):
    """How a texture is drawn when nothing else is asked: as the picture it is, as arrows (a 2D
    velocity in RG, a 3D one in RGB with z as a ring), as a signed scalar in two colors, or as a
    signed depth scalar, red toward the camera and green away."""
    IMAGE = 0
    VELOCITY = auto()
    SCALAR = auto()
    DEPTH = auto()


@dataclass
class Port:
    """A named texture of a stage, with its kind. `texture` is the `Fbo`, `SwapFbo` or `Tensor` object
    itself, never a buffer taken from it."""
    texture: Texture
    kind: TextureKind
