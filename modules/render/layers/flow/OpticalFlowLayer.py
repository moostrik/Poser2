""" Optical Flow Layer - computes and visualizes optical flow from camera images """

# Standard library imports

# Third-party imports
import numpy as np
from OpenGL.GL import * # type: ignore

# Local application imports
from modules.gl import Texture, Style
from modules.render.layers.LayerBase import LayerBase, Rect

from modules.flow import OpticalFlow, OpticalFlowConfig, Velocity, VelocityConfig, VisualizationMode
from modules.render.layers.flow.FlowDefinitions import DrawModes
from modules.render.layers import LayerBase

from modules.utils.HotReloadMethods import HotReloadMethods


class OpticalFlowLayer(LayerBase):
    def __init__(self, source: LayerBase) -> None:
        self._source: LayerBase = source

        # Flow pipeline
        self._optical_flow: OpticalFlow = OpticalFlow()
        self.flow_config: OpticalFlowConfig = self._optical_flow.config
        self._velocity_vis: Velocity = Velocity()
        self.vis_config: VelocityConfig = self._velocity_vis.config

        self.draw_mode = DrawModes.SCALAR
        self._velocity_vis.config.arrow_length = 40.0

        self.draw_mode: DrawModes = DrawModes.OUTPUT

        hot_reload = HotReloadMethods(self.__class__, True, True)

    @property
    def texture(self) -> Texture:
        return self._optical_flow.output

    def allocate(self, width: int, height: int, internal_format: int) -> None:
        self._optical_flow.allocate(width, height)
        self._velocity_vis.allocate(width, height)

    def deallocate(self) -> None:
        self._optical_flow.deallocate()
        self._velocity_vis.deallocate()

    def update(self) -> None:
        active: bool = getattr(self._source, "available", True)

        if not active:
            self._optical_flow.reset()
            return

        dirty: bool = getattr(self._source, "dirty", True)
        if dirty:
            prev_tex: Texture | None = getattr(self._source, "prev_texture", None)
            if prev_tex is not None:
                self._optical_flow.set(prev_tex)

            curr_tex: Texture = self._source.texture
            self._optical_flow.set(curr_tex)

            self._optical_flow.update()


    def draw(self, rect: Rect) -> None:

        self.draw_mode = DrawModes.SCALAR
        Style.push_style()
        Style.set_blend_mode(Style.BlendMode.DISABLED)
        if self.draw_mode == DrawModes.INPUT:
            self._optical_flow.draw_input(rect)
        elif self.draw_mode == DrawModes.OUTPUT:
            self._optical_flow.draw_output(rect)
        else:
            Style.set_blend_mode(Style.BlendMode.ADDITIVE)
            self._velocity_vis.config.mode = VisualizationMode.DIRECTION_MAP if self.draw_mode == DrawModes.SCALAR else VisualizationMode.ARROW_FIELD
            self._velocity_vis.set(self._optical_flow.output)
            self._velocity_vis.update()

            self._velocity_vis.draw(rect)
        Style.pop_style()