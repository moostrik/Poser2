"""Renders pose lines using the points a view placed the pose at (hd_trio's `CenterGeometry`, the flow's `ViewLayer`)."""

# Standard library imports
from typing import Protocol

# Local application imports
from modules.settings import Field, BaseSettings
from modules.gl import Fbo, Texture, clear_color, shaders
from modules.pose import PointLandmark, Points2D
from modules.utils import Point2f
from ...layer_base import LayerBase
from .shaders import PosePointLines
from ...color_settings import ColorSettings


class ViewPoseLayerSettings(BaseSettings):
    """Configuration for ViewPoseLayer pose line rendering."""
    line_width:     Field[float] = Field(3.0, min=0.5, max=10.0, description="Pose line thickness")
    line_smooth:    Field[float] = Field(0.0, min=0.0, max=5.0, description="Line anti-aliasing radius")
    use_scores:     Field[bool]  = Field(False, description="Color lines by confidence scores")
    draw_anchors:   Field[bool]  = Field(False, description="Show anchor points as circles")


class PosePlacement(Protocol):
    """Where a view puts the pose: `crop_pose_points` the pose's points in the view, uv with rows down, None
    without a pose; `lost`, the pose went since the last update."""
    @property
    def lost(self) -> bool: ...
    @property
    def crop_pose_points(self) -> Points2D | None: ...


def landmark_midpoint(points: Points2D, left: PointLandmark, right: PointLandmark) -> Point2f | None:
    """The midpoint of two landmarks, None when either is missing."""
    if points.get_valid(left) and points.get_valid(right):
        return (points.get_point2f(left) + points.get_point2f(right)) / 2
    return None


class ViewPoseLayer(LayerBase):
    """Renders pose keypoint lines in the view `geometry` places them in."""

    def __init__(self, cam_id: int, geometry: PosePlacement, settings: ViewPoseLayerSettings, color_settings: ColorSettings) -> None:
        self._cam_id: int = cam_id
        self._geometry: PosePlacement = geometry
        self._fbo: Fbo = Fbo()
        self._shader: PosePointLines = PosePointLines()
        self._circle_shader: shaders.DrawCircles = shaders.DrawCircles()

        # Configuration
        self.settings: ViewPoseLayerSettings = settings or ViewPoseLayerSettings()
        self._color_settings: ColorSettings = color_settings

    @property
    def texture(self) -> Texture:
        """Output texture for external use."""
        return self._fbo

    def allocate(self, width: int, height: int, internal_format: int) -> None:
        self._fbo.allocate(width, height, internal_format)
        self._shader.allocate()
        self._circle_shader.allocate()

    def deallocate(self) -> None:
        self._fbo.deallocate()
        self._shader.deallocate()
        self._circle_shader.deallocate()

    def update(self) -> None:
        if self._geometry.lost:
            self._fbo.clear()
            return

        transformed_points = self._geometry.crop_pose_points
        if transformed_points is None:
            return

        line_width: float = 1.0 / self._fbo.height * self.settings.line_width
        line_smooth: float = 1.0 / self._fbo.height * self.settings.line_smooth
        anchor_size: float = line_width * 4.0
        anchor_smooth: float = line_smooth
        anchor_color: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)

        # Candidate snap-point dots: shoulder / hip / feet midpoints, computed from
        # the transformed pose points (same space as the skeleton).
        mids = [
            landmark_midpoint(transformed_points, PointLandmark.left_shoulder, PointLandmark.right_shoulder),
            landmark_midpoint(transformed_points, PointLandmark.left_hip, PointLandmark.right_hip),
            landmark_midpoint(transformed_points, PointLandmark.left_ankle, PointLandmark.right_ankle),
        ]
        positions = [(m.x, m.y) for m in mids if m is not None]

        aspect_ratio = self._fbo.width / self._fbo.height if self._fbo.height > 0 else 1.0

        self._fbo.begin()
        clear_color()
        color = self._color_settings.track_colors[self._cam_id % len(self._color_settings.track_colors)].to_tuple()
        self._shader.use(transformed_points, line_width, line_smooth, color, self.settings.use_scores)
        if self.settings.draw_anchors and positions:
            self._circle_shader.use(positions, anchor_size, anchor_smooth, anchor_color, aspect_ratio)
        self._fbo.end()
