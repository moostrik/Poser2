"""deep_flow's chain: every layer the renderer runs, in the order it runs, with its size, its panel's shape and
when it runs."""

from dataclasses import dataclass
from enum import IntEnum

from ..settings import Layers, PoseFrame


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
    ChainLayer(Layers.capture_monitor, shape=Layers.capture_image, tick=True),
    # the view: the window cut, the depth read, the masks, the foreground, the color
    ChainLayer(Layers.view,            Size.VIEW, Layers.view, tick=True, capture=True),
    ChainLayer(Layers.view_monitor,    Size.VIEW, Layers.view, tick=True),
    # the motion, the bridge, the fluid
    ChainLayer(Layers.motion,          Size.LARGE, tick=True, capture=True),
    ChainLayer(Layers.motion_monitor,  Size.LARGE, tick=True),
    ChainLayer(Layers.bridge,          Size.LARGE, tick=True, capture=True),
    ChainLayer(Layers.bridge_monitor,  Size.LARGE, tick=True),
    ChainLayer(Layers.fluid,           Size.LARGE, tick=True),
    ChainLayer(Layers.fluid_monitor,   Size.LARGE, tick=True),
    # the display
    ChainLayer(Layers.composite,       Size.LARGE, tick=True),
    # the pose, beside the chain: its shape its frame's (`DeepFlowRender._layer_size`)
    ChainLayer(Layers.pose_monitor,    Size.PANEL, tick=True),
)
LAYERS: dict[Layers, ChainLayer] = {chain_layer.layer: chain_layer for chain_layer in CHAIN}
POSE_SHAPES: dict[PoseFrame, Layers] = {PoseFrame.CROP: Layers.cam_crop, PoseFrame.CAMERA: Layers.cam_image, PoseFrame.VIEW: Layers.view}
MONITORS: tuple[Layers, ...] = (Layers.capture_monitor, Layers.view_monitor, Layers.motion_monitor, Layers.bridge_monitor,
                                Layers.fluid_monitor)
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
