"""Superficial Flow render — split-view layer graph: the inference frame's monitor, the timings and the pose."""

from typing import Callable, cast

from OpenGL.GL import GL_RGBA16F, GL_RGBA, glViewport

from modules.gl import RenderBase, ScreenCapture, Shader, Style, clear_color, fill, fit, Texture, shaders
from modules.video import ScreenRecorder, ScreenRecorderSettings
from modules.inference import InferenceFrame
from modules import render
from modules.utils import Color, Rect, Point2f

from ..board import RenderBoard
from ..settings import PoseFrame, RenderSettings
from .layers import InferenceFrameMonitorLayer, PoseMonitorCompositor
from .chain import ChainLayers, Size, TICK_LAYERS, INFERENCE_FRAME_LAYERS, POSE_SHAPES, shown, size_of, layers_of, shape_of


OUTPUT_WIDTH: int = 1080
OUTPUT_HEIGHT: int = 1920

BORDER: Color = Color(0.4, 0.4, 0.4)    # the border around each layer in the main window, so a black layer shows
BORDER_WIDTH: float = 2.0       # px


class SuperficialFlowRender(RenderBase):
    def __init__(self, board: RenderBoard, settings: RenderSettings, screen_recordings: str,
                 num_cams: int = 1, max_players: int = 1) -> None:
        """`screen_recordings` the folder the screen recordings go to."""
        super().__init__(settings.window)
        self.max_players: int = max_players
        self.num_cams: int = num_cams
        self.settings: RenderSettings = settings

        self.board: RenderBoard = board

        self._main_layers: list[ChainLayers] = shown(settings.layer.main)      # the main panel, the show and the recording
        self._debug_layers: list[ChainLayers] = shown(settings.layer.debug)

        self.L: dict[ChainLayers, dict[int, render.LayerBase]] = {layer: {} for layer in ChainLayers}

        for i in range(self.num_cams):
            cam_image =     self.L[ChainLayers.cam_image][i] =   render.CameraImageLayer(    i, self.board)
            cam_crop =      self.L[ChainLayers.cam_crop][i] =    render.CropImageLayer(      i, self.board)
            image = self.L[ChainLayers.inference_frame_image][i] = render.InferenceFrameImageLayer()
            self.L[ChainLayers.inference_frame_monitor][i] =     InferenceFrameMonitorLayer(image)
            self.L[ChainLayers.pose_monitor][i] =                PoseMonitorCompositor(i, self.board, cam_image, cam_crop, settings.pose, settings.colors)
        self.L[ChainLayers.time_series][0] = render.TimeSeriesLayer(self.board, settings.time_series)      # the board's series, one for all cameras

        # Split view: the left and right panels side by side or stacked, whichever shows what they hold larger
        self._window_size: tuple[int, int] = (settings.window.width, settings.window.height)
        self._panel_aspect_ratios: list[float] = self._read_panel_aspect_ratios()
        self.subdivision: render.Subdivision = self._make_layout()

        # The layers' input, built right before they run, from what ran before them this tick
        self._wirings: dict[ChainLayers, Callable[[], None]] = {}

        # The border around each layer in the main window
        self._border: shaders.StrokeRect = shaders.StrokeRect()

        # The screen recording: the main panel's layers, or the main window, drawn into the capture's screen frame
        self._screen_capture: ScreenCapture = ScreenCapture()
        self.screen_recorder: ScreenRecorder = ScreenRecorder(settings.screen_recorder, self._screen_capture, screen_recordings)
        settings.screen_recorder.bind(ScreenRecorderSettings.snap, self._on_snap)

    def _make_layout(self) -> render.Subdivision:
        width, height = self._window_size
        aspect_ratios: list[float] = self._panel_aspect_ratios
        columns, rows = render.best_grid(aspect_ratios, width, height)
        aspect_ratio: float = render.grid_cell_aspect_ratio(aspect_ratios, columns, rows, width, height)
        panels = render.SubdivisionRow(name='panels', columns=columns, rows=rows, src_aspect_ratio=aspect_ratio, padding=Point2f(1.0, 1.0))
        return render.make_subdivision([panels], width, height, True)

    def _read_panel_aspect_ratios(self) -> list[float]:
        """The shape of each panel, the shape of its first layer."""
        aspect_ratios: list[float] = []
        for layer_types in (self._main_layers, self._debug_layers):
            width, height = self._layer_size(layer_types[0]) if layer_types else (OUTPUT_WIDTH, OUTPUT_HEIGHT)
            aspect_ratios.append(width / height)
        return aspect_ratios

    def _update_layout(self) -> None:
        """Rebuild the layout when a panel's shape has changed."""
        aspect_ratios: list[float] = self._read_panel_aspect_ratios()
        if aspect_ratios != self._panel_aspect_ratios:
            self._panel_aspect_ratios = aspect_ratios
            self.subdivision = self._make_layout()
            self.allocate_window_renders()

    def on_main_window_resize(self, width: int, height: int) -> None:
        self._window_size = (width, height)
        self.subdivision = self._make_layout()
        self.allocate_window_renders()

    def allocate(self) -> None:
        for layer_type, cam_dict in self.L.items():
            for layer in cam_dict.values():
                if size_of(layer_type) == Size.OUTPUT:
                    layer.allocate(OUTPUT_WIDTH, OUTPUT_HEIGHT, GL_RGBA16F)
        self.allocate_window_renders()
        self._border.allocate()
        self.screen_recorder.start()
        Shader.enable_hot_reload()

    def allocate_window_renders(self) -> None:
        # Allocate the panel-sized layers to split-view panel sizes
        for layer_type in layers_of(Size.PANEL):
            for i in range(self.num_cams):
                w, h = self.subdivision.get_allocation_size('panels', 0)
                self.L[layer_type][i].allocate(w, h, GL_RGBA)

    def deallocate(self) -> None:
        # the recording ends first: its encode finishes and frees its CUDA memory, then its GL objects go
        self.settings.screen_recorder.unbind(ScreenRecorderSettings.snap, self._on_snap)
        self.screen_recorder.stop()
        self.screen_recorder.deallocate()
        for cam_dict in self.L.values():
            for layer in cam_dict.values():
                layer.deallocate()
        self._border.deallocate()

    def update(self) -> None:
        self._notify_update()

        self._main_layers = shown(self.settings.layer.main)
        self._debug_layers = shown(self.settings.layer.debug)
        self._update_layout()

        Style.set_blend_mode(Style.BlendMode.ALPHA)
        seen: set[ChainLayers] = set()
        for layer_type in TICK_LAYERS + self._main_layers + self._debug_layers:
            if layer_type in seen:
                continue
            if layer_type in INFERENCE_FRAME_LAYERS:
                seen.update(INFERENCE_FRAME_LAYERS)
                self._run_inference_frames()
                continue
            seen.add(layer_type)
            wiring: Callable[[], None] | None = self._wirings.get(layer_type)
            if wiring is not None:
                wiring()
            for layer in self.L[layer_type].values():
                layer.update()

    # ========== The inference frames ==========

    def _run_inference_frames(self) -> None:
        """Every frame since the last tick, each camera's in order, through the frame layers (`_run_inference_frame`).
        A tick without a frame runs none of them, so no stage works on a frame twice."""
        for i in self.L[ChainLayers.inference_frame_image]:
            for frame in self.board.take_inference_frames(i):
                self._run_inference_frame(i, frame)

    def _run_inference_frame(self, i: int, frame: InferenceFrame) -> None:
        """One frame: its image into GL."""
        image = cast(render.InferenceFrameImageLayer, self.L[ChainLayers.inference_frame_image][i])
        image.set_inference_frame(frame)
        image.update()

    # ========== Drawing ==========

    def draw_main(self, width: int, height: int) -> None:
        if self.settings.debug:
            self._draw_panels(width, height)
        else:
            self._draw_show(width, height)      # the main panel alone, as on a second screen
        self._draw_screen_recording(width, height)

    def _draw_screen_recording(self, width: int, height: int) -> None:
        """The screen frame, when the recorder records this tick: the main panel's layers, each fit and centered at its
        own shape and added on black as the show draws them, or the main window just drawn; then the recorder's
        tick, which copies it."""
        recorder_settings: ScreenRecorderSettings = self.settings.screen_recorder
        if self.screen_recorder.capturing:
            capture: ScreenCapture = self._screen_capture
            if recorder_settings.main_window:
                capture.copy_window(width, height, recorder_settings.fill)
            else:
                place = fill if recorder_settings.fill else fit
                capture.frame.begin()
                clear_color()
                Style.set_blend_mode(Style.BlendMode.DISABLED)
                Style.push_style()
                Style.set_blend_mode(Style.BlendMode.ADD)
                for layer_type in self._main_layers:
                    if 0 in self.L[layer_type]:
                        x, y, w, h = place(*self._layer_size(layer_type), capture.width, capture.height)
                        glViewport(round(x), round(y), round(w), round(h))
                        self.L[layer_type][0].draw()
                Style.pop_style()
                capture.frame.end()
            glViewport(0, 0, width, height)
        self.screen_recorder.update(self.window_manager.frame_time)

    def _on_snap(self, _: bool) -> None:
        """The recording's size: the main panel's first layer's, or the main window's when it is recorded."""
        recorder_settings: ScreenRecorderSettings = self.settings.screen_recorder
        record: list[ChainLayers] = self._main_layers
        if recorder_settings.main_window:
            width, height = self._window_size
        elif record:
            width, height = self._layer_size(record[0])
        else:
            return
        recorder_settings.width = width
        recorder_settings.height = height

    def _draw_panels(self, width: int, height: int) -> None:
        """The main and the debug panel's layers, side by side or stacked."""
        clear_color()

        Style.set_blend_mode(Style.BlendMode.ALPHA)

        main_rect: Rect = self.subdivision.get_rect('panels', 0)
        self._draw_fit(self._main_layers, main_rect.x, height - main_rect.y - main_rect.height, main_rect.width, main_rect.height, True)

        debug_rect: Rect = self.subdivision.get_rect('panels', 1)
        self._draw_fit(self._debug_layers, debug_rect.x, height - debug_rect.y - debug_rect.height, debug_rect.width, debug_rect.height, True)

    def draw_secondary(self, monitor_id: int, width: int, height: int) -> None:
        self._draw_show(width, height)

    def _draw_show(self, width: int, height: int) -> None:
        """The show: the main panel's layers in order, each fit to `width` × `height` at its own shape."""
        clear_color()

        Style.set_blend_mode(Style.BlendMode.DISABLED)
        Style.push_style()
        Style.set_blend_mode(Style.BlendMode.ADD)

        self._draw_fit(self._main_layers, 0, 0, width, height)

        Style.pop_style()

    def _draw_fit(self, layer_types: list[ChainLayers], x: float, y: float, width: float, height: float, border: bool = False) -> None:
        """Draw each layer into a GL rect (origin bottom left) at its own shape, centered; with `border`, a border around
        each."""
        if width <= 0 or height <= 0:
            return
        for layer_type in layer_types:
            layers: dict[int, render.LayerBase] = self.L[layer_type]
            if 0 in layers:
                fit_x, fit_y, fit_width, fit_height = fit(*self._layer_size(layer_type), width, height)
                glViewport(int(x + fit_x), int(y + fit_y), int(fit_width), int(fit_height))
                layers[0].draw()
                if border:
                    self._draw_border(fit_width, fit_height)

    def _draw_border(self, width: float, height: float) -> None:
        """A border around the current viewport, `width` by `height` pixels."""
        if width <= 0 or height <= 0:
            return
        Style.push_style()
        Style.set_blend_mode(Style.BlendMode.ALPHA)
        self._border.use(Rect(0.0, 0.0, 1.0, 1.0), BORDER, BORDER_WIDTH / width, BORDER_WIDTH / height)
        Style.pop_style()

    def _layer_size(self, layer_type: ChainLayers) -> tuple[int, int]:
        """The shape of `layer_type`'s panel: its shape layer's texture, the output's without one."""
        shape: ChainLayers | None = shape_of(layer_type)
        if layer_type == ChainLayers.pose_monitor:
            shape = POSE_SHAPES[PoseFrame(self.settings.pose.frame)]
        if shape is not None and 0 in self.L[shape]:
            texture: Texture = self.L[shape][0].texture
            if texture.allocated:
                return texture.width, texture.height
        return OUTPUT_WIDTH, OUTPUT_HEIGHT
