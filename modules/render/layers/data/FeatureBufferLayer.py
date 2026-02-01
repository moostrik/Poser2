# Standard library imports
import math

# Third-party imports
import torch
from OpenGL.GL import *  # type: ignore

# Local application imports
from modules.DataHub import DataHub, DataHubType
from modules.gl import Fbo, Texture, Blit, clear_color, draw_box_string, text_init
from modules.gl.Tensor import Tensor
from modules.pose.features.Angles import ANGLE_NUM_LANDMARKS, ANGLE_LANDMARK_NAMES
from modules.render.layers.LayerBase import LayerBase, DataCache, Rect
from modules.render.shaders import PoseAngleVelocityWindow as shader
from .Colors import POSE_COLOR_LEFT, POSE_COLOR_RIGHT

from modules.utils.HotReloadMethods import HotReloadMethods

# Display range for visualization (will clamp values to this range)
DISPLAY_RANGE = (-math.pi, math.pi)


class FeatureBufferLayer(LayerBase):
    """Visualizes a single track from the RollingFeatureBuffer.

    Displays feature values as horizontal lines (one per angle element),
    with time flowing left-to-right. Uses GPU tensors directly via
    CUDA-OpenGL interop for efficient rendering.

    Similar to PDLayer but sources data from GPU feature buffer instead of pandas.
    """

    def __init__(self, track_id: int, data_hub: DataHub) -> None:
        self._track_id: int = track_id
        self._data_hub: DataHub = data_hub

        self._fbo: Fbo = Fbo()
        self._label_fbo: Fbo = Fbo()
        self._tensor: Tensor = Tensor()
        self._data_cache: DataCache[tuple[torch.Tensor, torch.Tensor]] = DataCache()

        self.draw_labels: bool = True

        self._shader: shader = shader()


        self.hot_reloader = HotReloadMethods(self.__class__, True, True)

    @property
    def texture(self) -> Texture:
        return self._fbo.texture

    def allocate(self, width: int | None = None, height: int | None = None, internal_format: int | None = None) -> None:
        if width is None or height is None or internal_format is None:
            return
        self._fbo.allocate(width, height, internal_format)
        self._label_fbo.allocate(width, height, internal_format)
        FeatureBufferLayer.render_labels(self._label_fbo)
        self._shader.allocate()

    def deallocate(self) -> None:
        self._fbo.deallocate()
        self._label_fbo.deallocate()
        self._tensor.deallocate()
        self._shader.deallocate()

    def draw(self) -> None:
        if self._fbo.allocated:
            Blit.use(self._fbo.texture)
            if self.draw_labels:
                Blit.use(self._label_fbo.texture)

    def update(self) -> None:
        # Get buffer from DataHub: (values, mask) tuple
        buffer_data: tuple[torch.Tensor, torch.Tensor] | None = self._data_hub.get_item(
            DataHubType.feature_buffer, 0
        )
        self._data_cache.update(buffer_data)

        if self._data_cache.idle or buffer_data is None:
            return

        values, mask = buffer_data

        # Extract this track's data: (window_size, feature_length)
        track_values = values[self._track_id]
        track_mask = mask[self._track_id]

        with torch.no_grad():
            stream_image = torch.stack([track_values.T, track_mask.T], dim=-1).flip(0)

        # Upload to OpenGL texture via CUDA-GL interop
        self._tensor.set_tensor(stream_image)
        self._tensor.update()

        self._shader.reload()

        # Render using StreamPose shader
        self._fbo.begin()
        clear_color()
        num_samples = stream_image.shape[1]  # window_size (width)
        num_streams = stream_image.shape[0]  # feature_length (height)
        output_aspect = self._fbo.width / self._fbo.height
        display_range = DISPLAY_RANGE[1]  # max value (e.g., pi)
        self._shader.use(self._tensor, num_samples, num_streams, line_width= 3 / self._fbo.height, output_aspect_ratio=output_aspect, display_range=display_range)
        self._fbo.end()

    @staticmethod
    def render_labels(fbo: Fbo) -> None:
        """Render angle labels overlay."""
        text_init()

        rect = Rect(0, 0, fbo.width, fbo.height)

        fbo.begin()
        clear_color()

        angle_num: int = ANGLE_NUM_LANDMARKS
        step: float = rect.height / angle_num
        colors: list[tuple[float, float, float, float]] = [
            (*POSE_COLOR_LEFT, 1.0),
            (*POSE_COLOR_RIGHT, 1.0)
        ]

        for i in range(angle_num):
            string: str = ANGLE_LANDMARK_NAMES[i]
            x: int = int(rect.x + 10)
            y: int = int(rect.y + rect.height - (rect.height - (i + 0.5) * step) - 9)
            clr: int = i % 2

            draw_box_string(x, y, string, colors[clr], (0.0, 0.0, 0.0, 0.3))  # type: ignore

        fbo.end()
