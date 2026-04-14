""" Draws the full camera image, depth tracklets, pose lines, and bounding boxes."""

# Local application imports
from modules.settings import Field, BaseSettings
from modules.gl import Fbo, Texture, Blit
from modules.data_hub import DataHub, Stage
from modules.render.layers.LayerBase import LayerBase, Rect

from modules.render.layers.cam.BBoxRenderer import BBoxRenderer, BBoxRendererSettings
from modules.render.layers.cam.TrackletRenderer import TrackletRenderer
from modules.render.layers.cam.PoseRenderer import PoseRenderer, PoseRendererSettings
from modules.render.color_settings import ColorSettings


class TrackerCompSettings(BaseSettings):
    stage:          Field[Stage] = Field(Stage.LERP, description="Pipeline stage for pose data")
    pose_line_width:Field[float] = Field(2.0, min=0.5, max=10.0, description="Pose line width")
    bbox_line_width:Field[int]   = Field(2, min=1, max=10, description="Bounding box line width in pixels")


class TrackerCompositor(LayerBase):
    def __init__(self, cam_id: int, data: DataHub, cam_texture: Texture, settings: TrackerCompSettings, color_settings: ColorSettings) -> None:
        self.settings: TrackerCompSettings = settings
        self._color_settings: ColorSettings = color_settings
        self._cam_id: int = cam_id
        self._data_hub: DataHub = data
        self._fbo: Fbo = Fbo()

        self._cam_texture: Texture = cam_texture
        self._depth_track_renderer: TrackletRenderer = TrackletRenderer(cam_id, data)

        bbox_config_A: BBoxRendererSettings = BBoxRendererSettings(stage=Stage.CLEAN, line_width=self.settings.bbox_line_width * 2.0, color=color_settings.history)
        self._bbox_renderer_A: BBoxRenderer = BBoxRenderer(cam_id, data, bbox_config_A)

        bbox_config_B: BBoxRendererSettings = BBoxRendererSettings(stage=Stage.LERP, line_width=self.settings.bbox_line_width, color=color_settings.track_colors[cam_id])
        self._bbox_renderer_B: BBoxRenderer = BBoxRenderer(cam_id, data, bbox_config_B)

        pose_config: PoseRendererSettings = PoseRendererSettings(stage=self.settings.stage, line_width=self.settings.pose_line_width, line_smooth=0.0, use_scores=False, use_bbox=True)
        self._pose_renderer: PoseRenderer = PoseRenderer(cam_id, data, color_settings=None, settings=pose_config)

    @property
    def texture(self) -> Texture:
        return self._fbo.texture

    def allocate(self, width: int, height: int, internal_format: int) -> None:
        self._fbo.allocate(width, height, internal_format)
        self._depth_track_renderer.allocate(width, height, internal_format)
        self._bbox_renderer_A.allocate(width, height, internal_format)
        self._bbox_renderer_B.allocate(width, height, internal_format)
        self._pose_renderer.allocate(width, height, internal_format)

    def deallocate(self) -> None:
        self._fbo.deallocate()
        self._depth_track_renderer.deallocate()
        self._bbox_renderer_A.deallocate()
        self._bbox_renderer_B.deallocate()
        self._pose_renderer.deallocate()

    def update(self) -> None:
        # Update sub-layers
        self._depth_track_renderer.update()
        self._bbox_renderer_A.update()
        self._bbox_renderer_B.update()
        self._pose_renderer.update()

        # Composite: camera + tracklets + bboxes + all pose lines
        self._fbo.begin()
        Blit.use(self._cam_texture)
        self._depth_track_renderer.draw()
        self._bbox_renderer_A.settings.color = self._color_settings.history
        self._bbox_renderer_A.draw()
        self._bbox_renderer_B.settings.color = self._color_settings.track_colors[self._cam_id]
        self._bbox_renderer_B.draw()
        self._pose_renderer.draw()
        self._fbo.end()