""" Optical Flow Layer - computes and visualizes optical flow from camera images """

# Standard library imports

# Third-party imports
import numpy as np
from OpenGL.GL import * # type: ignore

# Local application imports
from modules.DataHub import DataHub
from modules.gl import Fbo, Texture, Image
from modules.DataHub import DataHub, DataHubType
from modules.render.layers.LayerBase import LayerBase, Rect

from modules.flow import OpticalFlow, OpticalFlowConfig, Velocity, VelocityConfig, VisualizationMode

from modules.utils.HotReloadMethods import HotReloadMethods


class OpticalFlowLayer(LayerBase):
    def __init__(self, cam_id: int, data_hub: DataHub) -> None:
        self._cam_id: int = cam_id
        self._data_hub: DataHub = data_hub
        self._fbo: Fbo = Fbo()

        # Image textures for uploading numpy arrays
        self.curr_image: Image = Image(channel_order='BGR')  # OpenCV uses BGR
        self.prev_image: Image = Image(channel_order='BGR')
        self._p_images: tuple[np.ndarray, np.ndarray] | None = None

        # Flow pipeline
        self._optical_flow: OpticalFlow = OpticalFlow()
        self._velocity_viz: Velocity = Velocity()

        hot_reload = HotReloadMethods(self.__class__, True, True)

    def allocate(self, width: int, height: int, internal_format: int) -> None:
        self._fbo.allocate(width, height, internal_format)

        # Allocate Image textures (auto-allocate on first use, but we can pre-allocate)
        self.prev_image.allocate(width, height, GL_RGB8)
        self.curr_image.allocate(width, height, GL_RGB8)

        # Allocate flow pipeline
        self._optical_flow.allocate(width, height)
        self._velocity_viz.allocate(width, height)

    def deallocate(self) -> None:
        self._fbo.deallocate()
        self.prev_image.deallocate()
        self.curr_image.deallocate()
        self._optical_flow.deallocate()
        self._velocity_viz.deallocate()

    def draw(self, rect: Rect) -> None:
        self._fbo.draw(rect.x, rect.y, rect.width, rect.height)
        # self._optical_flow.draw(rect)
        # self._optical_flow.draw_input(rect)
        # self.curr_image.draw(rect.x, rect.y, rect.width, rect.height)

    def update(self) -> None:
        key: int = self._cam_id

        images: tuple[np.ndarray, np.ndarray] | None = self._data_hub.get_item(DataHubType.flow_images, key)

        if images is self._p_images:
            # print("OpticalFlowLayer: No change in images, skipping update")
            return # no update needed
        self._p_images = images

        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        self._fbo.clear(0.0, 0.0, 0.0, 0.0)
        if images is None:
            return

        # Upload images to GPU textures
        self.prev_image.set_image(images[0])
        self.curr_image.set_image(images[1])
        self.prev_image.update()
        self.curr_image.update()

        # Feed frames to optical flow (build frame history)
        # Note: OpticalFlow expects frames in sequence, so we feed prev then curr
        self._optical_flow.set(self.prev_image)
        self._optical_flow.set(self.curr_image)
        self._optical_flow.update()

        self._velocity_viz.config.mode = VisualizationMode.ARROW_FIELD
        self._velocity_viz.config.scale = 3.0
        self._velocity_viz.config.arrow_scale = 10.5
        self._velocity_viz.config.grid_spacing = 10

        # Visualize velocity field
        self._velocity_viz.set(self._optical_flow.output)
        self._velocity_viz.update()

        # Render visualization to FBO
        self._fbo.begin()
        self._velocity_viz.output.draw(0, 0, self._fbo.width, self._fbo.height)
        self._fbo.end()

    def get_fbo(self) -> Fbo:
        return self._fbo

