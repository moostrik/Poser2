""" Renders pose lines for all tracks visible in a camera """

# Local application imports
from modules.settings import Field, BaseSettings
from modules.board import HasFrames
from modules.pose import Frame, Points2D, BBox
from ....layer_base import LayerBase
from ..shaders import PosePointLines
from ....color_settings import ColorSettings


class PoseDrawSettings(BaseSettings):
    line_smooth:Field[float] = Field(0.0, min=0.0, max=10.0, description="Line smoothing/antialiasing width")
    use_scores: Field[bool]  = Field(False, description="Use confidence scores for line opacity")
    use_bbox:   Field[bool]  = Field(True, description="Transform points to image space using bbox")


class PoseDraw(LayerBase):
    """The pose lines of a camera's tracks. `stage` and `line_width` (pixels) are inputs its owner sets."""

    def __init__(self, cam_id: int, board: HasFrames, settings: PoseDrawSettings, color_settings: ColorSettings | None = None,
                 *, stage: int = 0, line_width: float = 1.0) -> None:
        self.settings: PoseDrawSettings = settings
        self.stage: int = stage
        self.line_width: float = line_width
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
        line_width: float = 1.0 / self._height * self.line_width
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
                points = points.to_image(pose[BBox].to_rect())

            self._shader.use(points, line_width, line_smooth, color=color, use_scores=self.settings.use_scores)

    def update(self) -> None:
        self._cam_poses = {p for p in self._board.get_frames(self.stage).values() if p.cam_id == self._cam_id}
