"""superficial_flow's chain: every layer the renderer runs, in the order it runs, with its size, its panel's shape
and when it runs."""

from dataclasses import dataclass
from enum import IntEnum, auto

from ..settings import Layers, PoseFrame


class ChainLayers(IntEnum):
    """Every layer the renderer runs, in the chain's order, each stage with its monitor; the pose last."""
    # the inference frame
    inference_frame_image = 0
    inference_frame_monitor = auto()
    # the view
    view =          auto()
    view_monitor =  auto()
    # the density's color
    color =         auto()
    color_monitor = auto()
    # the app's timings
    time_series =   auto()
    # the pose: the camera's image and its crop, and the pose's monitor
    cam_image =     auto()
    cam_crop =      auto()
    pose_monitor =  auto()


class Size(IntEnum):
    """The size a layer is allocated at."""
    OUTPUT = 0      # the output, OUTPUT_WIDTH × OUTPUT_HEIGHT
    VIEW = 1        # the view's `width` × `height`
    PANEL = 2       # a panel of the main window


@dataclass(frozen=True)
class ChainLayer:
    """A layer of the chain: its size, the layer whose texture is its panel's shape (None: the output's), whether
    it runs every tick (else only while a panel shows it), and whether once per frame."""
    layer: ChainLayers
    size: Size = Size.OUTPUT
    shape: ChainLayers | None = None
    tick: bool = False
    per_inference_frame: bool = False


# Each stage with its layers, the outputs it hands on and its display layers
CHAIN: tuple[ChainLayer, ...] = (
    # the frame: the camera's image and its crop, the frame's image into GL
    ChainLayer(ChainLayers.cam_image,               shape=ChainLayers.cam_image, tick=True),
    ChainLayer(ChainLayers.cam_crop,                shape=ChainLayers.cam_crop, tick=True),
    ChainLayer(ChainLayers.inference_frame_image,   shape=ChainLayers.inference_frame_image, tick=True, per_inference_frame=True),
    ChainLayer(ChainLayers.inference_frame_monitor, shape=ChainLayers.inference_frame_image, tick=True),
    # the view: the window cut, the mask; then the density's color from them
    ChainLayer(ChainLayers.view,                    Size.VIEW, ChainLayers.view, tick=True, per_inference_frame=True),
    ChainLayer(ChainLayers.view_monitor,            Size.VIEW, ChainLayers.view, tick=True),
    ChainLayer(ChainLayers.color,                   Size.VIEW, ChainLayers.view, tick=True, per_inference_frame=True),
    ChainLayer(ChainLayers.color_monitor,           Size.VIEW, ChainLayers.view, tick=True),
    # the pose, beside the chain: its shape its frame's (`SuperficialFlowRender._layer_size`)
    ChainLayer(ChainLayers.pose_monitor,            Size.PANEL, tick=True),
)
LAYERS: dict[ChainLayers, ChainLayer] = {chain_layer.layer: chain_layer for chain_layer in CHAIN}
POSE_SHAPES: dict[PoseFrame, ChainLayers] = {PoseFrame.CROP: ChainLayers.cam_crop, PoseFrame.CAMERA: ChainLayers.cam_image, PoseFrame.VIEW: ChainLayers.view}
MONITORS: tuple[ChainLayers, ...] = (ChainLayers.view_monitor, ChainLayers.color_monitor)   # the flow's
TICK_LAYERS: list[ChainLayers] = [chain_layer.layer for chain_layer in CHAIN if chain_layer.tick]
INFERENCE_FRAME_LAYERS: list[ChainLayers] = [chain_layer.layer for chain_layer in CHAIN if chain_layer.per_inference_frame]

# What a panel can show, the select's `Layers`, and the chain's layer that draws it
SHOWN: dict[Layers, ChainLayers] = {
    Layers.inference_frame: ChainLayers.inference_frame_monitor,
    Layers.view:        ChainLayers.view_monitor,
    Layers.color:       ChainLayers.color_monitor,
    Layers.time_series: ChainLayers.time_series,
    Layers.pose:        ChainLayers.pose_monitor,
}


def shown(layers: list[Layers]) -> list[ChainLayers]:
    """The chain's layers that draw what a panel's `layers` select, in order."""
    return [SHOWN[layer] for layer in layers]


def size_of(layer: ChainLayers) -> Size:
    """The size `layer` is allocated at; a layer the chain does not name is output-sized."""
    return LAYERS[layer].size if layer in LAYERS else Size.OUTPUT


def layers_of(size: Size) -> list[ChainLayers]:
    """The chain's layers allocated at `size`, in order."""
    return [chain_layer.layer for chain_layer in CHAIN if chain_layer.size == size]


def shape_of(layer: ChainLayers) -> ChainLayers | None:
    """The layer whose texture is `layer`'s panel's shape; None for the output's."""
    return LAYERS[layer].shape if layer in LAYERS else None
