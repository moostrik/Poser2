""" Renders pose lines for all tracks visible in a camera """

# Standard library imports
from dataclasses import dataclass

# Third-party imports
from OpenGL.GL import * # type: ignore

# Local application imports
from modules.ConfigBase import ConfigBase, config_field
from modules.DataHub import DataHub, Stage
from modules.pose.Frame import Frame
from modules.pose.features.Points2D import Points2D
from modules.render.layers.LayerBase import LayerBase, Rect
from modules.render.layers.data.colors import TRACK_COLORS
from modules.render.shaders import PosePointLines


@dataclass
class PoseRendererConfig(ConfigBase):
    stage: Stage = config_field(Stage.LERP, description="Pipeline stage for pose data", fixed=True)
    line_width: float = config_field(1.0, min=0.5, max=10.0, description="Pose line width")
    line_smooth: float = config_field(0.0, min=0.0, max=10.0, description="Line smoothing/antialiasing width")
    use_scores: bool = config_field(False, description="Use confidence scores for line opacity")
    use_bbox: bool = config_field(True, description="Transform points to image space using bbox")


class PoseRenderer(LayerBase):
    def __init__(self, cam_id: int, data: DataHub, colors: list[tuple[float, float, float, float]] | None = None, config: PoseRendererConfig | None = None) -> None:
        self._config: PoseRendererConfig = config or PoseRendererConfig()
        self._data: DataHub = data
        self._cam_id: int = cam_id
        self._colors: list[tuple[float, float, float, float]] | None = colors
        self._cam_poses: set[Frame] = set()
        self._height: int = 1

        self._shader: PosePointLines = PosePointLines()

    def allocate(self, width: int | None = None, height: int | None = None, internal_format: int | None = None) -> None:
        self._shader.allocate()
        if height is not None:
            self._height = height

    def deallocate(self) -> None:
        self._shader.deallocate()

    def draw(self) -> None:
        line_width: float = 1.0 / self._height * self._config.line_width
        line_smooth: float = 1.0 / self._height * self._config.line_smooth

        for pose in self._cam_poses:
            # Get color for this track if colors provided, otherwise use anatomical colors (None)
            color: tuple[float, float, float, float] | None = None
            if self._colors is not None:
                color = self._colors[pose.track_id % len(self._colors)]

            # Transform points to image space if use_bbox is enabled
            points: Points2D = pose.points
            if self._config.use_bbox:
                points = self._transform_to_image_space(points, pose.bbox.to_rect())

            self._shader.use(points, line_width, line_smooth, color=color, use_scores=self._config.use_scores)

    def update(self) -> None:
        self._cam_poses = self._data.get_poses_for_cam(self._config.stage, self._cam_id)

    @staticmethod
    def _transform_to_image_space(points: Points2D, bbox: Rect) -> Points2D:
        """Transform pose points from bbox-relative [0,1] to image space."""
        x_bbox, y_bbox = points.get_xy_arrays()
        x_image = x_bbox * bbox.width + bbox.x
        y_image = y_bbox * bbox.height + bbox.y
        return Points2D.from_xy_arrays(x_image, y_image, points.scores)
