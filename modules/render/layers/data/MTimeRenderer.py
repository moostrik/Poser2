
# Third-party imports
from OpenGL.GL import * # type: ignore

# Local application imports
from modules.DataHub import DataHub, DataHubType, PoseDataHubTypes
from modules.pose.Frame import Frame
from modules.render.layers.LayerBase import LayerBase, DataCache
from modules.gl import Text

from modules.utils.HotReloadMethods import HotReloadMethods


class MTimeRenderer(LayerBase):
    def __init__(self, track_id: int, data: DataHub, data_type: PoseDataHubTypes) -> None:
        self._data: DataHub = data
        self._track_id: int = track_id
        self._motion_time: str | None = None
        self.data_type: PoseDataHubTypes = data_type

        # Text renderer for GPU-based text drawing (allocate later)
        self._text_renderer: Text = Text()
        self._width: int = 0
        self._height: int = 0

        self.hot_reloader = HotReloadMethods(self.__class__, True, True)

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
        pose: Frame | None = self._data.get_item(DataHubType(self.data_type), self._track_id)

        if pose is None:
            self._motion_time = None
        else:
            time_str: str = f"MT: {pose.motion_time:.2f}"
            self._motion_time = time_str
