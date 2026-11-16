"""deep_flow's chain: every layer the renderer runs, in the order it runs, with its size, its panel's shape and
when it runs."""

from dataclasses import dataclass
from enum import IntEnum

from ..settings import Layers


class Size(IntEnum):
    """The size a layer is allocated at."""
    OUTPUT = 0      # the output, OUTPUT_WIDTH × OUTPUT_HEIGHT
    VIEW = 1        # the view's, the motion's `width` × `height`
    LARGE = 2       # twice the output
    PANEL = 3       # a panel of the main window


@dataclass(frozen=True)
class ChainLayer:
    """A layer of the chain: its size, the layer whose texture is its panel's shape (None: the output's), whether
    it runs every tick (else only while a panel shows it), and whether once per capture."""
    layer: Layers
    size: Size = Size.OUTPUT
    shape: Layers | None = None
    tick: bool = False
    capture: bool = False


# Each stage with its layers, the outputs it hands on and its display layers
CHAIN: tuple[ChainLayer, ...] = (
    # the capture: the camera's image and its crop, the capture's image and depth into GL
    ChainLayer(Layers.cam_image,       shape=Layers.cam_image, tick=True),
    ChainLayer(Layers.cam_crop,        shape=Layers.cam_crop, tick=True),
    ChainLayer(Layers.capture_image,   shape=Layers.capture_image, tick=True, capture=True),
    ChainLayer(Layers.capture_depth,   shape=Layers.capture_image, tick=True, capture=True),   # on the image's view, at the depth's own grid
    # the view: the window cut, the depth read, the masks, the foreground
    ChainLayer(Layers.view,            Size.VIEW, Layers.view_image, tick=True, capture=True),
    ChainLayer(Layers.view_image,      Size.VIEW, Layers.view_image),
    ChainLayer(Layers.view_foreground, Size.VIEW, Layers.view_image),
    ChainLayer(Layers.view_mask,       Size.VIEW, Layers.view_image),
    ChainLayer(Layers.view_depth_mask, Size.VIEW, Layers.view_image),
    ChainLayer(Layers.view_depth,      Size.VIEW, Layers.view_image, tick=True),
    ChainLayer(Layers.view_pose,       Size.VIEW, Layers.view_image, tick=True),
    ChainLayer(Layers.foreground_look, Size.VIEW, Layers.view_image, tick=True, capture=True),  # the density's color
    ChainLayer(Layers.color_mask,      Size.VIEW, Layers.view_image, tick=True),
    # the motion, the bridge, the fluid
    ChainLayer(Layers.motion,          Size.LARGE, tick=True, capture=True),
    ChainLayer(Layers.bridge,          Size.LARGE, tick=True, capture=True),
    ChainLayer(Layers.fluid,           Size.LARGE, tick=True),
    # the display
    ChainLayer(Layers.monitor_a,       Size.LARGE, tick=True),
    ChainLayer(Layers.monitor_b,       Size.LARGE, tick=True),
    ChainLayer(Layers.composite,       Size.LARGE, tick=True),
    ChainLayer(Layers.tracker,         Size.PANEL, Layers.cam_image, tick=True),   # camera stretched into a panel-sized FBO
    ChainLayer(Layers.pose,            Size.PANEL, Layers.cam_crop, tick=True),    # crop stretched into a panel-sized FBO
)
LAYERS: dict[Layers, ChainLayer] = {chain_layer.layer: chain_layer for chain_layer in CHAIN}
TICK_LAYERS: list[Layers] = [chain_layer.layer for chain_layer in CHAIN if chain_layer.tick]
CAPTURE_LAYERS: list[Layers] = [chain_layer.layer for chain_layer in CHAIN if chain_layer.capture]


def size_of(layer: Layers) -> Size:
    """The size `layer` is allocated at; a layer the chain does not name is output-sized."""
    return LAYERS[layer].size if layer in LAYERS else Size.OUTPUT


def layers_of(size: Size) -> list[Layers]:
    """The chain's layers allocated at `size`, in order."""
    return [chain_layer.layer for chain_layer in CHAIN if chain_layer.size == size]


def shape_of(layer: Layers) -> Layers | None:
    """The layer whose texture is `layer`'s panel's shape; None for the output's."""
    return LAYERS[layer].shape if layer in LAYERS else None
