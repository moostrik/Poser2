""" Optical Flow Layer - computes and visualizes optical flow from camera images """

# Standard library imports

# Third-party imports
import numpy as np
from OpenGL.GL import * # type: ignore

# Local application imports
from modules.gl import Texture, Style
from modules.render.layers.LayerBase import LayerBase, Rect

from modules.flow import OpticalFlow, OpticalFlowConfig, Visualizer, VisualisationFieldConfig
from modules.render.layers import LayerBase

from modules.utils.HotReloadMethods import HotReloadMethods
from enum import IntEnum, auto

class DrawModes(IntEnum):
    INPUT = 0
    OUTPUT = auto()
    FIELD = auto()

class OpticalFlowLayer(LayerBase):
    def __init__(self, source: LayerBase) -> None:
        self._source: LayerBase = source

        # Flow pipeline
        self._optical_flow: OpticalFlow = OpticalFlow()
        self._visualizer: Visualizer = Visualizer()

        self.vis_config.element_length = 40.0
        self.draw_mode: DrawModes = DrawModes.FIELD

        hot_reload = HotReloadMethods(self.__class__, True, True)

    @property
    def config(self) -> OpticalFlowConfig:
        """Access to optical flow configuration."""
        return self._optical_flow.config  # type: ignore

    @property
    def vis_config(self) -> VisualisationFieldConfig:
        """Access to visualization configuration."""
        return self._visualizer.config  # type: ignore

    @property
    def texture(self) -> Texture:
        return self._optical_flow._output

    def allocate(self, width: int, height: int, internal_format: int) -> None:
        self._optical_flow.allocate(width, height)
        self._visualizer.allocate(width, height)

    def deallocate(self) -> None:
        self._optical_flow.deallocate()
        self._visualizer.deallocate()

    def update(self) -> None:
        active: bool = getattr(self._source, "available", True)

        if not active:
            self._optical_flow.reset()
            return

        dirty: bool = getattr(self._source, "dirty", True)
        if dirty:
            prev_tex: Texture | None = getattr(self._source, "prev_texture", None)
            if prev_tex is not None:
                self._optical_flow.set_color(prev_tex)

            curr_tex: Texture = self._source.texture
            self._optical_flow.set_color(curr_tex)

            self._optical_flow.update()


    def draw(self, rect: Rect) -> None:
        Style.push_style()
        Style.set_blend_mode(Style.BlendMode.DISABLED)

        if self.draw_mode == DrawModes.INPUT:
            self._optical_flow.draw_input(rect)
        elif self.draw_mode == DrawModes.OUTPUT:
            self._optical_flow.draw_output(rect)
        else:
            Style.set_blend_mode(Style.BlendMode.ADDITIVE)
            self._visualizer.draw(self._optical_flow.velocity, rect)

        Style.pop_style()

    def reset(self) -> None:
        """Reset bridge state."""
        self._optical_flow.reset()
