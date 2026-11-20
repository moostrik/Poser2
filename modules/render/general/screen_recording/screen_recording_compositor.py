"""ScreenRecordingCompositor: the screen recording's frame, drawn from the layers an app hands it, and its recorder."""

import logging
from dataclasses import dataclass, field

from OpenGL.GL import glViewport

from modules.gl import Style, Texture, clear_color, fill, fit
from modules.video import ScreenRecorder, ScreenRecorderSettings

from ...layer_base import LayerBase
from .screen_capture import ScreenCapture
from .screen_preview import ScreenPreview

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ScreenRecordingSource:
    """A layer to record and the shape it is drawn at (px)."""
    layer: LayerBase
    width: int
    height: int


@dataclass(frozen=True)
class ScreenRecordingInput:
    """One tick of the recording: the layers in the order they are drawn, the main window's size, the tick's time."""
    sources: list[ScreenRecordingSource] = field(default_factory=list)
    window_width: int = 0
    window_height: int = 0
    timestamp: float = 0.0


class ScreenRecordingCompositor(LayerBase):
    """The screen recording on the render tick: the recorder's tick, the capture made at the recorder's size, the
    finished frames handed to the recorder, and, when the recorder captures this tick, the sources drawn into the
    screen frame, each fit (or filled) at its own shape and added on black as the show draws them, or the main window
    just drawn; while a marker of a held take is dragged, its frame over the main window. Owns the `ScreenCapture`
    and the `ScreenRecorder`; `snap` sets the size to the first source's, or the window's."""

    def __init__(self, settings: ScreenRecorderSettings, folder: str) -> None:
        """`folder`: where the recordings go."""
        self.settings: ScreenRecorderSettings = settings
        self._capture: ScreenCapture = ScreenCapture()
        self._preview: ScreenPreview = ScreenPreview()
        self.recorder: ScreenRecorder = ScreenRecorder(settings, folder)
        self._capture.add_screen_frame_callback(self.recorder.submit_screen_frame)
        self._input: ScreenRecordingInput = ScreenRecordingInput()
        self.settings.bind(ScreenRecorderSettings.snap, self._on_snap)

    @property
    def texture(self) -> Texture:
        """The screen frame."""
        return self._capture.frame

    def allocate(self, width: int | None = None, height: int | None = None, internal_format: int | None = None) -> None:
        """Start the recorder's thread; the capture's GL objects are made when a take begins."""
        self.recorder.start()

    def deallocate(self) -> None:
        """End the recording (its encode finishes, its thread gives every frame back), then free the GL objects."""
        self.settings.unbind(ScreenRecorderSettings.snap, self._on_snap)
        self.recorder.stop()
        self._capture.remove_screen_frame_callback(self.recorder.submit_screen_frame)
        if self._preview.allocated:
            self._preview.deallocate()
        if self._capture.allocated:
            self._capture.deallocate()

    def set_input(self, input: ScreenRecordingInput) -> None:
        self._input = input

    def update(self) -> None:
        """The recording's tick; call it right after the main window is drawn, before it is shown."""
        input = self._input
        self.recorder.update(input.timestamp)
        self._follow_capture_size()
        self._capture.poll()
        if self.recorder.capturing and self._capture.allocated:
            capture = self._capture
            if self.settings.main_window:
                capture.copy_window(input.window_width, input.window_height, self.settings.fill)
            else:
                place = fill if self.settings.fill else fit
                capture.frame.begin()
                clear_color()
                Style.set_blend_mode(Style.BlendMode.DISABLED)
                Style.push_style()
                Style.set_blend_mode(Style.BlendMode.ADD)
                for source in input.sources:
                    x, y, w, h = place(source.width, source.height, capture.width, capture.height)
                    glViewport(round(x), round(y), round(w), round(h))
                    source.layer.draw()
                Style.pop_style()
                capture.frame.end()
            capture.capture(input.timestamp)
            glViewport(0, 0, input.window_width, input.window_height)
        self._draw_preview(input.window_width, input.window_height)

    def _follow_capture_size(self) -> None:
        """Make the capture at the recorder's size while it has a take, and free it after; never while a frame of it
        is lent or on its way."""
        size = self.recorder.capture_size
        capture, preview = self._capture, self._preview
        if capture.lent:
            return
        if size is None:
            if capture.allocated:
                capture.deallocate()
            if preview.allocated:
                preview.deallocate()
            return
        if capture.allocated and (capture.width, capture.height) == size:
            return
        try:
            capture.allocate(*size)
            preview.allocate(*size)
        except Exception as e:
            self.recorder.fail(f'could not make the screen capture at {size[0]}x{size[1]}: {e}')

    def _draw_preview(self, window_width: int, window_height: int) -> None:
        """While a marker of a held take is dragged, its frame over the main window, fit on black: the frame under the
        marker is stored for the preview's thread, and the picture shown is the newest one it has made."""
        preview = self._preview
        if not preview.allocated:
            return
        shown = self.recorder.preview_frame
        if shown is not None:
            preview.set_screen_frame(*shown)
        preview.update()
        if shown is None or preview.shown is None or window_width <= 0 or window_height <= 0:
            return
        glViewport(0, 0, window_width, window_height)
        clear_color(0.0, 0.0, 0.0, 1.0)
        x, y, w, h = fit(preview.width, preview.height, window_width, window_height)
        glViewport(round(x), round(y), round(w), round(h))
        Style.push_style()
        Style.set_blend_mode(Style.BlendMode.DISABLED)
        preview.fbo.draw()
        Style.pop_style()
        glViewport(0, 0, window_width, window_height)

    def _on_snap(self, _: bool) -> None:
        """The recording's size: the first source's shape, or the main window's when it is recorded."""
        input = self._input
        if self.settings.main_window:
            width, height = input.window_width, input.window_height
        elif input.sources:
            width, height = input.sources[0].width, input.sources[0].height
        else:
            return
        self.settings.width = width
        self.settings.height = height
