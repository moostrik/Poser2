"""ScreenRecordingCompositor: the screen recording's frame, drawn from the layers an app hands it, and its recorder."""

from dataclasses import dataclass, field

from OpenGL.GL import glViewport

from modules.gl import ScreenCapture, Style, Texture, clear_color, fill, fit
from modules.video import ScreenRecorder, ScreenRecorderSettings

from ...layer_base import LayerBase


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
    """The screen frame of a recording: the sources, each fit (or filled) at its own shape and added on black as the
    show draws them, or the main window just drawn; then the recorder's tick, which copies it, and while a marker of
    a held take is dragged its frame over the main window. Owns the `ScreenCapture` and the `ScreenRecorder`; `snap`
    sets the size to the first source's, or the window's."""

    def __init__(self, settings: ScreenRecorderSettings, folder: str) -> None:
        """`folder`: where the recordings go."""
        self.settings: ScreenRecorderSettings = settings
        self._capture: ScreenCapture = ScreenCapture()
        self.recorder: ScreenRecorder = ScreenRecorder(settings, self._capture, folder)
        self._input: ScreenRecordingInput = ScreenRecordingInput()
        self.settings.bind(ScreenRecorderSettings.snap, self._on_snap)

    @property
    def texture(self) -> Texture:
        """The screen frame."""
        return self._capture.frame

    def allocate(self, width: int | None = None, height: int | None = None, internal_format: int | None = None) -> None:
        """Start the recorder's thread; the capture's GL objects are made at Start."""
        self.recorder.start()

    def deallocate(self) -> None:
        """End the recording (its encode finishes and frees its memory), then free the GL objects."""
        self.settings.unbind(ScreenRecorderSettings.snap, self._on_snap)
        self.recorder.stop()
        self.recorder.deallocate()

    def set_input(self, input: ScreenRecordingInput) -> None:
        self._input = input

    def update(self) -> None:
        """Draw the screen frame when the recorder records this tick, then run the recorder's tick; call it right after
        the main window is drawn, before it is shown."""
        input = self._input
        if self.recorder.capturing:
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
            glViewport(0, 0, input.window_width, input.window_height)
        self.recorder.update(input.timestamp)
        self._draw_preview(input.window_width, input.window_height)

    def _draw_preview(self, window_width: int, window_height: int) -> None:
        """While a marker of a held take is dragged, its frame over the main window, fit on black."""
        slot = self.recorder.preview_slot
        if slot is None or window_width <= 0 or window_height <= 0:
            return
        capture = self._capture
        capture.show_preview(slot)
        glViewport(0, 0, window_width, window_height)
        clear_color(0.0, 0.0, 0.0, 1.0)
        x, y, w, h = fit(capture.width, capture.height, window_width, window_height)
        glViewport(round(x), round(y), round(w), round(h))
        Style.push_style()
        Style.set_blend_mode(Style.BlendMode.DISABLED)
        capture.preview.draw()
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
