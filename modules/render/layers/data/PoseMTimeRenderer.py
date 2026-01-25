
# Third-party imports
from OpenGL.GL import * # type: ignore

# Local application imports

from modules.DataHub import DataHub, DataHubType, PoseDataHubTypes
from modules.pose.Frame import Frame
from modules.render.layers.LayerBase import LayerBase, Rect
from modules.gl.Text import draw_box_string, text_init


class PoseMTimeRenderer(LayerBase):
    def __init__(self, track_id: int, data: DataHub, data_type: PoseDataHubTypes) -> None:
        self._data: DataHub = data
        self._track_id: int = track_id
        self._motion_time: str | None = None

        self.data_type: PoseDataHubTypes = data_type

        text_init()


    def deallocate(self) -> None:
        pass

    def draw(self) -> None:
        if self._motion_time is None:
            return

        draw_box_string(rect.x + 20, rect.y + rect.height - 20, self._motion_time, (1.0, 1.0, 1.0, 1.0), (0.0, 0.0, 0.0, 0.3), True) # type: ignore

    def update(self) -> None:
        pose: Frame | None = self._data.get_item(DataHubType(self.data_type), self._track_id)

        if pose is None:
            self._motion_time = None
        else:
            time_str: str = f"Motion Time: {pose.motion_time:.2f}"
            self._motion_time = time_str
