""" Renders pose lines for all tracks visible in a camera """

# Third-party imports
from OpenGL.GL import * # type: ignore

# Local application imports
from modules.settings import Field, BaseSettings
from modules.blackboard import HasFrames
from modules.pose.frame import Frame
from modules.pose.features import Points2D, BBox
from modules.render.layers.LayerBase import LayerBase, Rect
from modules.render.shaders import PosePointLines
from modules.render.color_settings import ColorSettings


class PoseRendererSettings(BaseSettings):
    stage:      Field[int] = Field(3, description="Pipeline stage for pose data")
    line_width: Field[float] = Field(1.0, min=0.5, max=10.0, description="Pose line width")
    line_smooth:Field[float] = Field(0.0, min=0.0, max=10.0, description="Line smoothing/antialiasing width")
    use_scores: Field[bool]  = Field(False, description="Use confidence scores for line opacity")
    use_bbox:   Field[bool]  = Field(True, description="Transform points to image space using bbox")


class PoseRenderer(LayerBase):
    def __init__(self, cam_id: int, board: HasFrames, color_settings: ColorSettings | None = None, settings: PoseRendererSettings | None = None) -> None:
        self.settings: PoseRendererSettings = settings or PoseRendererSettings()
        self._board: HasFrames = board
        self._cam_id: int = cam_id
        self._color_settings: ColorSettings | None = color_settings
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
        line_width: float = 1.0 / self._height * self.settings.line_width
        line_smooth: float = 1.0 / self._height * self.settings.line_smooth

        for pose in self._cam_poses:
            # Get color for this track from color settings, otherwise use anatomical colors (None)
            color: tuple[float, float, float, float] | None = None
            if self._color_settings is not None:
                colors = self._color_settings.track_color_tuples
                color = colors[pose.track_id % len(colors)]

            # Transform points to image space if use_bbox is enabled
            points: Points2D = pose[Points2D]
            if self.settings.use_bbox:
                points = self._transform_to_image_space(points, pose[BBox].to_rect())

            self._shader.use(points, line_width, line_smooth, color=color, use_scores=self.settings.use_scores)

    def update(self) -> None:
        self._cam_poses = {p for p in self._board.get_frames(self.settings.stage).values() if p.cam_id == self._cam_id}

    @staticmethod
    def _transform_to_image_space(points: Points2D, bbox: Rect) -> Points2D:
        """Transform pose points from bbox-relative [0,1] to image space."""
        x_bbox, y_bbox = points.get_xy_arrays()
        x_image = x_bbox * bbox.width + bbox.x
        y_image = y_bbox * bbox.height + bbox.y
        return Points2D.from_xy_arrays(x_image, y_image, points.scores)
