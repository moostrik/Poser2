""" Draws the full camera image, depth tracklets, pose lines, and bounding boxes."""

# Local application imports
from typing import Any

from modules.settings import Field, BaseSettings
from modules.gl import Fbo, Texture, clear_color
from ...layer_base import LayerBase

from ..pose import PoseBoxDraw, PoseDraw, PoseDrawSettings
from .tracklet_draw import TrackletDraw
from ...color_settings import ColorSettings


class TrackerCompositorSettings(BaseSettings):
    stage:          Field[int] = Field(0, description="Pipeline stage for pose data")
    box_stage:      Field[int] = Field(0, description="Pipeline stage of the bounding boxes")
    pose_line_width:Field[float] = Field(2.0, min=0.5, max=10.0, description="Pose line width")
    bbox_line_width:Field[int]   = Field(2, min=1, max=10, description="Bounding box line width in pixels")
    image:          Field[bool]  = Field(True, description="The camera image under them; off, they lie over any layer of its frame")


class TrackerCompositor(LayerBase):
    def __init__(self, cam_id: int, board: Any, cam_texture: Texture, settings: TrackerCompositorSettings, color_settings: ColorSettings) -> None:
        self.settings: TrackerCompositorSettings = settings
        self._color_settings: ColorSettings = color_settings
        self._cam_id: int = cam_id
        self._fbo: Fbo = Fbo()

        self._cam_texture: Texture = cam_texture
        self._depth_track_draw: TrackletDraw = TrackletDraw(cam_id, board)

        # The boxes: the history color 2x wide under the track color; their stage, widths and color, and the pose's
        # stage and width, are inputs, handed on from the settings each update (`_set_inputs`).
        self._bbox_draw_A: PoseBoxDraw = PoseBoxDraw(cam_id, board)
        self._bbox_draw_B: PoseBoxDraw = PoseBoxDraw(cam_id, board, color_settings)

        pose_settings: PoseDrawSettings = PoseDrawSettings(line_smooth=0.0, use_scores=False, use_bbox=True)
        self._pose_draw: PoseDraw = PoseDraw(cam_id, board, pose_settings)
        self._set_inputs()

    @property
    def texture(self) -> Texture:
        return self._fbo

    def allocate(self, width: int, height: int, internal_format: int) -> None:
        self._fbo.allocate(width, height, internal_format)
        self._depth_track_draw.allocate(width, height, internal_format)
        self._bbox_draw_A.allocate(width, height, internal_format)
        self._bbox_draw_B.allocate(width, height, internal_format)
        self._pose_draw.allocate(width, height, internal_format)

    def deallocate(self) -> None:
        self._fbo.deallocate()
        self._depth_track_draw.deallocate()
        self._bbox_draw_A.deallocate()
        self._bbox_draw_B.deallocate()
        self._pose_draw.deallocate()

    def _set_inputs(self) -> None:
        """The boxes' and the pose's stages and widths, and the history box's color, from the settings as the panel
        has them now."""
        s: TrackerCompositorSettings = self.settings
        self._bbox_draw_A.stage = s.box_stage
        self._bbox_draw_A.line_width = s.bbox_line_width * 2.0
        self._bbox_draw_A.color = self._color_settings.history
        self._bbox_draw_B.stage = s.box_stage
        self._bbox_draw_B.line_width = s.bbox_line_width
        self._pose_draw.stage = s.stage
        self._pose_draw.line_width = s.pose_line_width

    def update(self) -> None:
        self._set_inputs()
        self._depth_track_draw.update()
        self._bbox_draw_A.update()
        self._bbox_draw_B.update()
        self._pose_draw.update()

        # Composite: camera + tracklets + bboxes + all pose lines
        self._fbo.begin()
        if self.settings.image:
            self._cam_texture.draw()
        else:
            clear_color()
        self._depth_track_draw.draw()
        self._bbox_draw_A.draw()
        self._bbox_draw_B.draw()
        self._pose_draw.draw()
        self._fbo.end()