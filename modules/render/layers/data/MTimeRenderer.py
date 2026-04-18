
# Third-party imports
from OpenGL.GL import * # type: ignore

# Local application imports
from modules.settings import Field, BaseSettings
from modules.whiteboard import HasFrames
from modules.pose.frame import Frame
from modules.pose.features import MotionTime
from modules.render.layers.LayerBase import LayerBase
from modules.gl import Text


class MTimeRendererSettings(BaseSettings):
    stage: Field[int] = Field(3, description="Pipeline stage for pose data")


class MTimeRenderer(LayerBase):
    def __init__(self, track_id: int, board: HasFrames, config: MTimeRendererSettings | None = None) -> None:
        self._config: MTimeRendererSettings = config or MTimeRendererSettings()
        self._board: HasFrames = board
        self._track_id: int = track_id
        self._motion_time: str | None = None

        # Text renderer for GPU-based text drawing (allocate later)
        self._text_renderer: Text = Text()
        self._width: int = 0
        self._height: int = 0

    def allocate(self, width: int | None = None, height: int | None = None, internal_format: int | None = None) -> None:
        if width and height:
            self._width = width
            self._height = height
        self._text_renderer.allocate("files/RobotoMono-Regular.ttf", font_size=40)

    def deallocate(self) -> None:
        self._text_renderer.deallocate()

    def draw(self) -> None:
        if self._motion_time is None or self._width == 0:
            return

        # Draw text in bottom-left corner with padding
        # Note: y=0 is top of screen after coordinate flip in shader
        x = 20
        y = 20  # 20 pixels from top

        self._text_renderer.draw_box_text(
            x, y, self._motion_time,
            color=(1.0, 1.0, 1.0, 1.0),
            bg_color=(0.0, 0.0, 0.0, 0.8),
            screen_width=self._width,
            screen_height=self._height
        )

    def update(self) -> None:
        pose: Frame | None = self._board.get_frame(self._config.stage, self._track_id)

        if pose is None:
            self._motion_time = None
        else:
            time_str: str = f"MT: {pose[MotionTime].value:.2f}"
            self._motion_time = time_str
