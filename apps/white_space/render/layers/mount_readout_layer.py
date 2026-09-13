"""MountReadoutLayer — one camera's tilt and roll error, over that camera's own view.

The pinned `camera.mount.status` only says OK or OFF; this says which camera and which axis, where
the operator is already looking while turning the tripod.
"""

import math

from modules.gl import Text
from modules.oak import CameraSettings, MountCheckSettings, mount_deviation
from modules.render.layers import LayerBase

_MARGIN_PX: float = 6.0
_GAP_PX: float = 8.0

_OK_COLOR:       tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
_OFF_COLOR:      tuple[float, float, float, float] = (1.0, 0.25, 0.25, 1.0)
_UNMEASURED_COLOR: tuple[float, float, float, float] = (0.6, 0.6, 0.6, 1.0)
_BACKGROUND:     tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.6)


class MountReadoutLayer(LayerBase):
    """Two boxed labels in the view's top-left corner: the signed tilt error (measured − `tilt`)
    and the roll, each red when past `tolerance`. A camera without an IMU reading shows `no IMU`.

    Owns no FBO: it draws into whatever viewport is current when `draw()` is called, which must be
    this camera's tile, sized by the last `allocate`.
    """

    def __init__(self, camera: CameraSettings, mount: MountCheckSettings) -> None:
        self._camera: CameraSettings = camera
        self._mount: MountCheckSettings = mount
        self._text: Text = Text()
        self._width: int = 1
        self._height: int = 1

    def allocate(self, width: int, height: int, internal_format: int) -> None:
        self._text.allocate()
        self._width = max(1, width)
        self._height = max(1, height)

    def deallocate(self) -> None:
        self._text.deallocate()

    def update(self) -> None:
        pass

    def draw(self) -> None:
        tilt, roll = mount_deviation(self._camera)
        if math.isnan(tilt) and math.isnan(roll):
            self._label(_MARGIN_PX, 'no IMU', _UNMEASURED_COLOR)
            return

        x: float = _MARGIN_PX
        x = self._label(x, self._reading('tilt', tilt), self._color(tilt)) + _GAP_PX
        self._label(x, self._reading('roll', roll), self._color(roll))

    @staticmethod
    def _reading(axis: str, value: float) -> str:
        return f'{axis} --' if math.isnan(value) else f'{axis} {value:+.1f}°'

    def _color(self, value: float) -> tuple[float, float, float, float]:
        if math.isnan(value):
            return _UNMEASURED_COLOR
        return _OFF_COLOR if abs(value) > self._mount.tolerance else _OK_COLOR

    def _label(self, x: float, text: str, color: tuple[float, float, float, float]) -> float:
        """Draw one boxed label with its left edge at ``x``; return its right edge."""
        width, _ = self._text.measure_text(text)
        self._text.draw_box_text(x, _MARGIN_PX, text, color, _BACKGROUND, self._width, self._height)
        return x + width + 2 * Text.BOX_PADDING
