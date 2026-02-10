""" Draws the full camera image, depth tracklets, pose lines, and bounding boxes."""

# Standard library imports
from dataclasses import dataclass

# Local application imports
from modules.ConfigBase import ConfigBase, config_field
from modules.gl import Fbo, Texture, Blit
from modules.DataHub import DataHub, Stage
from modules.render.layers.LayerBase import LayerBase, Rect

from modules.render.layers.cam.BBoxRenderer import BBoxRenderer, BBoxRendererConfig
from modules.render.layers.cam.TrackletRenderer import TrackletRenderer
from modules.render.layers.cam.PoseRenderer import PoseRenderer, PoseRendererConfig

from modules.utils.HotReloadMethods import HotReloadMethods


@dataclass
class TrackerCompositorConfig(ConfigBase):
    stage: Stage = config_field(Stage.LERP, description="Pipeline stage for pose data", fixed=True)
    pose_line_width: float = config_field(1.0, min=0.5, max=10.0, description="Pose line width")
    bbox_line_width: int = config_field(2, min=1, max=10, description="Bounding box line width in pixels")


class TrackerCompositor(LayerBase):
    def __init__(self, cam_id: int, data: DataHub, cam_texture: Texture, config: TrackerCompositorConfig | None = None) -> None:
        self._config: TrackerCompositorConfig = config or TrackerCompositorConfig()
        self._cam_id: int = cam_id
        self._data_hub: DataHub = data
        self._fbo: Fbo = Fbo()

        self._cam_texture: Texture = cam_texture
        self._depth_track_renderer: TrackletRenderer = TrackletRenderer(cam_id, data)

        bbox_config: BBoxRendererConfig = BBoxRendererConfig(stage=self._config.stage, line_width=self._config.bbox_line_width)
        self._bbox_renderer: BBoxRenderer = BBoxRenderer(cam_id, data, (1.0, 1.0, 1.0, 1.0), bbox_config)

        pose_config: PoseRendererConfig = PoseRendererConfig(stage=self._config.stage, line_width=self._config.pose_line_width, line_smooth=0.0, use_scores=False, use_bbox=True)
        self._pose_renderer: PoseRenderer = PoseRenderer(cam_id, data, colors=None, config=pose_config)

        self.hot_reload = HotReloadMethods(self.__class__)

    @property
    def texture(self) -> Texture:
        return self._fbo.texture

    @property
    def bbox_color(self) -> tuple[float, float, float, float]:
        return self._bbox_renderer.bbox_color
    @bbox_color.setter
    def bbox_color(self, value: tuple[float, float, float, float]) -> None:
        self._bbox_renderer.bbox_color = value

    def allocate(self, width: int, height: int, internal_format: int) -> None:
        self._fbo.allocate(width, height, internal_format)
        self._depth_track_renderer.allocate(width, height, internal_format)
        self._bbox_renderer.allocate(width, height, internal_format)
        self._pose_renderer.allocate(width, height, internal_format)

    def deallocate(self) -> None:
        self._fbo.deallocate()
        self._depth_track_renderer.deallocate()
        self._bbox_renderer.deallocate()
        self._pose_renderer.deallocate()

    def update(self) -> None:
        # Update sub-layers
        self._depth_track_renderer.update()
        self._bbox_renderer.update()
        self._pose_renderer.update()

        # Composite: camera + tracklets + bboxes + all pose lines
        self._fbo.begin()
        Blit.use(self._cam_texture)
        self._depth_track_renderer.draw()
        self._bbox_renderer.draw()
        self._pose_renderer.draw()
        self._fbo.end()