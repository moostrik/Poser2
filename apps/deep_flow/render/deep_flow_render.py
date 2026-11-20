"""Deep Flow render — split-view layer graph with the 3D flow: view, color, motion, bridge and the volumetric fluid."""

from threading import Event
from typing import Callable, cast

from OpenGL.GL import GL_RGBA16F, GL_RGBA, glViewport

from modules.gl import GpuTimer, RenderBase, Shader, Style, clear_color, fit, Texture, MouseButton, shaders
from modules.video import ScreenRecorderSettings
from modules.flow import Port, hold_tick
from modules.inference import InferenceFrame, Poses
from modules.pose import Frame, FrameDict
from modules import render
from modules.utils import FPS, Color, Rect, Point2f

from ..board import RenderBoard
from ..settings import PoseFrame, RenderSettings, WindowsSettings
from .layers import InferenceFrameMonitorLayer, ViewInput, ViewOutput, ViewLayer, PoseMonitorCompositor
from .chain import ChainLayers, Size, TICK_LAYERS, INFERENCE_FRAME_LAYERS, MONITORS, POSE_SHAPES, shown, size_of, layers_of, shape_of


OUTPUT_WIDTH: int = 1080
OUTPUT_HEIGHT: int = 1920

BORDER: Color = Color(0.4, 0.4, 0.4)    # the border around each layer in the main window, so a black layer shows
BORDER_WIDTH: float = 2.0       # px


class DeepFlowRender(RenderBase):
    def __init__(self, board: RenderBoard, settings: RenderSettings, windows: WindowsSettings, recorder_settings: ScreenRecorderSettings, fov: float, screen_recordings: str,
                 num_cams: int = 1, max_players: int = 1) -> None:
        """`fov` is the camera image's horizontal field (degrees), for the metres a view width spans;
        `recorder_settings` the screen recorder's, `screen_recordings` the folder its recordings go to."""
        super().__init__(windows)
        self.max_players: int = max_players
        self.num_cams: int = num_cams
        self.settings: RenderSettings = settings

        self.board: RenderBoard = board

        self._main_layers: list[ChainLayers] = shown(settings.layer.main)      # the main panel, the show and the recording
        self._debug_layers: list[ChainLayers] = shown(settings.layer.debug)

        self.L: dict[ChainLayers, dict[int, render.LayerBase]] = {layer: {} for layer in ChainLayers}
        self._debug_monitors: dict[ChainLayers, dict[int, render.Monitor2DLayer]] = {}     # each stage's debug monitor

        for i in range(self.num_cams):
            cam_image =     self.L[ChainLayers.cam_image][i] =   render.CameraImageLayer(    i, self.board)
            cam_crop =      self.L[ChainLayers.cam_crop][i] =    render.CropImageLayer(      i, self.board)
            self.L[ChainLayers.inference_frame_image][i] =               render.InferenceFrameImageLayer()
            self.L[ChainLayers.inference_frame_depth][i] =               render.InferenceFrameDepthLayer()

            # The stages, each with its monitor, and the pose's monitor
            view =          self.L[ChainLayers.view][i] =        ViewLayer(           settings.view, fov)
            self.L[ChainLayers.pose_monitor][i] =                PoseMonitorCompositor(i, self.board, cam_image, cam_crop, view, settings.pose, settings.colors)
            self.L[ChainLayers.color][i] =                       render.ColorLayer(          settings.color)
            self.L[ChainLayers.motion][i] =                      render.Motion3DLayer(       i, settings.motion)
            self.L[ChainLayers.bridge][i] =                      render.Bridge3DLayer(       settings.bridge)
            fluid =         self.L[ChainLayers.fluid][i] =       render.Fluid3DLayer(        settings.fluid)
            self.L[ChainLayers.inference_frame_monitor][i] = InferenceFrameMonitorLayer(cast(render.InferenceFrameImageLayer, self.L[ChainLayers.inference_frame_image][i]),
                                                                    cast(render.InferenceFrameDepthLayer, self.L[ChainLayers.inference_frame_depth][i]),
                                                                    settings.inference_frame, settings.view.crop)
            # each stage's main monitor in self.L, shown in the main panel; its debug monitor in self._debug_monitors,
            # shown in the debug panel; the flat stages' flat, the fluid's 3D
            for key, stage, flat in ((ChainLayers.view_monitor, render.FlowStage.VIEW, settings.view),
                                     (ChainLayers.color_monitor, render.FlowStage.COLOR, settings.color),
                                     (ChainLayers.motion_monitor, render.FlowStage.MOTION, settings.motion),
                                     (ChainLayers.bridge_monitor, render.FlowStage.BRIDGE, settings.bridge)):
                self.L[key][i] = render.Monitor2DLayer(stage, flat.monitor_main)
                self._debug_monitors.setdefault(key, {})[i] = render.Monitor2DLayer(stage, flat.monitor_debug)
            self.L[ChainLayers.fluid_monitor][i] = render.Monitor3DLayer(render.FlowStage.FLUID, settings.fluid.monitor_main)
            self._debug_monitors.setdefault(ChainLayers.fluid_monitor, {})[i] = \
                render.Monitor3DLayer(render.FlowStage.FLUID, settings.fluid.monitor_debug)
        self.L[ChainLayers.time_series][0] = render.TimeSeriesLayer(self.board, settings.time_series)      # the board's series, one for all cameras

        # Split view: the left and right panels side by side or stacked, whichever shows what they hold larger
        self._window_size: tuple[int, int]                  # window 0's, from allocate on
        self._panel_aspect_ratios: list[float] = self._read_panel_aspect_ratios()
        self.subdivision: render.Subdivision                # made in allocate, at window 0's size

        # The length of each tick, the flow's dt, held to twice the mean tick (`hold_tick`)
        self._last_tick: float = 0.0
        self._tick_mean: float | None = None
        self._dt: float = 1.0 / 60.0
        self.add_update_callback(self._measure_tick)

        # A reset asked for from another thread, applied at the next update on the GL thread
        self._reset_fluid: Event = Event()

        self._view_timer: GpuTimer = GpuTimer("GPU view")
        self._view_size: tuple[int, int] = (0, 0)       # the view layers' size: the motion's, set in allocate

        # The fluid's and the monitors' input, built right before they run, from what ran before them this tick
        self._wirings: dict[ChainLayers, Callable[[], None]] = {
            ChainLayers.fluid: self._wire_fluid,
            ChainLayers.view_monitor: self._wire_monitors,
        }

        # Each camera's frames the motion measures, for the panel's cadence (read on any thread)
        self.motion_rates: dict[int, FPS] = {}
        self._inference_frame_timestamps: dict[int, float] = {}     # each camera's frame run last

        # The border around each layer in the main window
        self._border: shaders.StrokeRect = shaders.StrokeRect()

        # The mouse in the main window drives the monitors' volume views: a drag that starts anywhere goes to
        # every monitor, so A and B keep one angle and one layer (nothing else in the main window takes it)
        self._drag: tuple[MouseButton, float, float] | None = None     # the button held and where the cursor was
        self._dragged: bool = False
        self.window_manager.input.add_mouse_button_callback(self._on_mouse_button)
        self.window_manager.input.add_mouse_move_callback(self._on_mouse_move)
        self.window_manager.input.add_mouse_scroll_callback(self._on_mouse_scroll)

        # The screen recording: the main panel's layers, or the main window
        self.screen_recording: render.ScreenRecordingCompositor = render.ScreenRecordingCompositor(recorder_settings, screen_recordings)

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

    def on_window_resize(self, window: int, width: int, height: int) -> None:
        if window != 0:
            return
        self._window_size = (width, height)
        self.subdivision = self._make_layout()
        self.allocate_window_renders()

    def allocate(self) -> None:
        self._window_size = self.window_size(0)
        self.subdivision = self._make_layout()
        self._view_size = (self.settings.view.width, self.settings.view.height)
        for layer_type, cam_dict in self.L.items():
            for layer in cam_dict.values():
                size: Size = size_of(layer_type)
                if size == Size.LARGE:
                    layer.allocate(OUTPUT_WIDTH * 2, OUTPUT_HEIGHT * 2, GL_RGBA16F)
                elif size == Size.VIEW:
                    layer.allocate(*self._view_size, GL_RGBA16F)
                elif size == Size.OUTPUT:
                    layer.allocate(OUTPUT_WIDTH, OUTPUT_HEIGHT, GL_RGBA16F)
        for monitors in self._debug_monitors.values():
            for monitor in monitors.values():
                monitor.allocate(OUTPUT_WIDTH * 2, OUTPUT_HEIGHT * 2, GL_RGBA16F)     # a monitor renders at the size it is drawn at
        self.allocate_window_renders()
        self._border.allocate()
        self.screen_recording.allocate()
        Shader.enable_hot_reload()

    def allocate_window_renders(self) -> None:
        # Allocate the panel-sized layers to split-view panel sizes
        for layer_type in layers_of(Size.PANEL):
            for i in range(self.num_cams):
                w, h = self.subdivision.get_allocation_size('panels', 0)
                self.L[layer_type][i].allocate(w, h, GL_RGBA)

    def _follow_view_size(self) -> None:
        """Reallocate the view layers when the view's size changed."""
        size: tuple[int, int] = (self.settings.view.width, self.settings.view.height)
        if size == self._view_size:
            return
        self._view_size = size
        for layer_type in layers_of(Size.VIEW):
            for layer in self.L[layer_type].values():
                layer.deallocate()
                layer.allocate(*size, GL_RGBA16F)

    def deallocate(self) -> None:
        # the recording ends first: its encode finishes and frees its CUDA memory, then its GL objects go
        self.screen_recording.deallocate()
        for cam_dict in self.L.values():
            for layer in cam_dict.values():
                layer.deallocate()
        for monitors in self._debug_monitors.values():
            for monitor in monitors.values():
                monitor.deallocate()
        self._border.deallocate()
        self._view_timer.deallocate()

    def _measure_tick(self, now: float) -> None:
        if self._last_tick > 0.0:
            self._dt, self._tick_mean = hold_tick(now - self._last_tick, self._tick_mean)
        self._last_tick = now

    # ========== The mouse: the monitors' volume views ==========

    def _monitors(self) -> list[render.Monitor2DLayer]:
        return [cast(render.Monitor2DLayer, layer) for key in MONITORS
                for layer in (*self.L[key].values(), *self._debug_monitors.get(key, {}).values())]

    def _volume_monitors(self) -> list[render.Monitor3DLayer]:
        """The monitors with a 3D view, the ones the mouse moves."""
        return [monitor for monitor in self._monitors() if isinstance(monitor, render.Monitor3DLayer)]

    def _on_mouse_button(self, window: int, x: float, y: float, button: MouseButton, pressed: bool) -> None:
        """A press in window 0 starts a drag; a middle click that did not move resets the views."""
        if pressed and window != 0:
            return
        if pressed:
            self._drag, self._dragged = (button, x, y), False
            return
        if self._drag is not None and self._drag[0] == MouseButton.MIDDLE and not self._dragged:
            for monitor in self._volume_monitors():
                monitor.reset_view()
        self._drag = None

    def _on_mouse_move(self, window: int, x: float, y: float) -> None:
        if self._drag is None or window != 0:
            return
        button, last_x, last_y = self._drag
        for monitor in self._volume_monitors():
            monitor.drag(x - last_x, y - last_y, button)
        self._drag, self._dragged = (button, x, y), True

    def _on_mouse_scroll(self, window: int, steps: float) -> None:
        """A layer per step for every monitor, the show's density's layers; over window 0."""
        if window != 0:
            return
        for monitor in self._volume_monitors():
            monitor.scroll(steps, self.settings.fluid.depth)

    def reset_fluid(self) -> None:
        """Clear the optical flow, the bridges and the fluid volume at the next update; any thread."""
        self._reset_fluid.set()

    def update(self) -> None:
        self._notify_update()

        if self._reset_fluid.is_set():
            self._reset_fluid.clear()
            for layer_type in [ChainLayers.motion, ChainLayers.bridge, ChainLayers.fluid]:
                for layer in self.L[layer_type].values():
                    cast(render.StageLayer, layer).reset()

        self._main_layers = shown(self.settings.layer.main)
        self._debug_layers = shown(self.settings.layer.debug)
        self._update_layout()
        self._follow_view_size()

        if self.screen_recording.playing:
            self._hold_show()
            return

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
            for monitor in self._debug_monitors.get(layer_type, {}).values():
                monitor.update()

    def _hold_show(self) -> None:
        """While the held take plays, the show waits: the camera frames are taken and dropped, and no layer works,
        so the fluid goes on where it stopped and the other windows keep its last picture."""
        for i in self.L[ChainLayers.view]:
            self.board.take_inference_frames(i)

    # ========== The flow's wiring ==========


    def _run_inference_frames(self) -> None:
        """Every frame since the last tick, each camera's in order, through the frame layers (`_run_inference_frame`).
        A tick without a frame runs none of them, so no stage works on a frame twice. With `freeze_frame` the frames
        are taken and dropped and only the view and the color run, on the last frame, made again when one of their
        settings changes."""
        for i in self.L[ChainLayers.view]:
            frames: list[InferenceFrame] = self.board.take_inference_frames(i)
            if self.settings.freeze_frame:
                view = cast(ViewLayer, self.L[ChainLayers.view][i])
                view.update()
                self._run_color(i, view)
                continue
            for index, frame in enumerate(frames):
                self._run_inference_frame(i, frame, same_tick=index > 0)

    def _run_inference_frame(self, i: int, frame: InferenceFrame, same_tick: bool) -> None:
        """One frame: its image and depth into GL, the view of it with its player's pose, the density's color from
        the view, the motion on the view's image, and the bridge stepping over the time since the frame before, the
        density's color the color stage's. `same_tick`: another frame in this tick, whose bridge outputs the fluid
        takes with the last's."""
        image = cast(render.InferenceFrameImageLayer, self.L[ChainLayers.inference_frame_image][i])
        depth = cast(render.InferenceFrameDepthLayer, self.L[ChainLayers.inference_frame_depth][i])
        view = cast(ViewLayer, self.L[ChainLayers.view][i])
        motion = cast(render.Motion3DLayer, self.L[ChainLayers.motion][i])
        bridge = cast(render.Bridge3DLayer, self.L[ChainLayers.bridge][i])

        image.set_inference_frame(frame)
        image.update()
        depth.set_inference_frame(frame)
        depth.update()
        poses: FrameDict = frame[Poses].poses
        pose: Frame | None = poses[min(poses)] if poses else None
        view.set_input(ViewInput(image.texture, depth.texture, frame.timestamp, pose))
        self._view_timer.begin()
        view.update()
        self._view_timer.end()
        color: render.ColorLayer = self._run_color(i, view)

        interval: float = render.frame_interval(frame.timestamp, self._inference_frame_timestamps.get(i), self.settings.motion.fps, self._dt, self._dt)
        self._inference_frame_timestamps[i] = frame.timestamp
        self.motion_rates.setdefault(i, FPS()).processed()
        output: ViewOutput = view.output
        motion.set_input(render.MotionInput(output.image, frame.timestamp, interval, output.depth, output.mask))
        motion.update()
        bridge.set_input(render.BridgeInput(motion.output.velocity, interval, color.output.color,
                                        mask=output.mask, z_position=motion.output.z_position, same_tick=same_tick))
        bridge.update()

    def _run_color(self, i: int, view: ViewLayer) -> render.ColorLayer:
        """The density's color from camera `i`'s view, made again when the view was or one of its settings changed."""
        color = cast(render.ColorLayer, self.L[ChainLayers.color][i])
        output: ViewOutput = view.output
        color.set_input(render.ColorInput(output.image, output.mask, view.timestamp, view.made))
        color.update()
        return color

    def _wire_fluid(self) -> None:
        """Each fluid's input, one injection per bridge, at full weight."""
        dt: float = self._dt
        bridges: dict[int, render.Bridge3DLayer] = {i: cast(render.Bridge3DLayer, layer) for i, layer in self.L[ChainLayers.bridge].items()}
        for i, _fluid in self.L[ChainLayers.fluid].items():
            fluid = cast(render.Fluid3DLayer, _fluid)
            injections: list[render.Injection] = []
            for bridge in bridges.values():
                output: render.BridgeOutput = bridge.output
                injections.append(render.Injection(output.velocity, output.density, output.color, output.temperature,
                                               output.acceleration,
                                               velocity_z_position=output.velocity_z_position,
                                               density_z_position=output.density_z_position,
                                               temperature_z_position=output.temperature_z_position,
                                               acceleration_z_position=output.acceleration_z_position,
                                               expansion=output.expansion, expansion_z_position=output.expansion_z_position))
            fluid.set_input(render.FluidInput(dt, injections))

    def _wire_monitors(self) -> None:
        """The monitors' input, every stage's ports; each monitor shows its own stage's."""
        for i, _motion in self.L[ChainLayers.motion].items():
            if i not in self.L[ChainLayers.bridge] or i not in self.L[ChainLayers.fluid] or i not in self.L[ChainLayers.view]:
                continue
            ports: dict[render.FlowStage, dict[str, Port]] = {
                render.FlowStage.VIEW: cast(ViewLayer, self.L[ChainLayers.view][i]).ports,
                render.FlowStage.COLOR: cast(render.ColorLayer, self.L[ChainLayers.color][i]).ports,
                render.FlowStage.MOTION: cast(render.Motion3DLayer, _motion).ports,
                render.FlowStage.BRIDGE: cast(render.Bridge3DLayer, self.L[ChainLayers.bridge][i]).ports,
                render.FlowStage.FLUID: cast(render.Fluid3DLayer, self.L[ChainLayers.fluid][i]).ports,
            }
            for monitor in self._monitors():
                monitor.set_input(render.MonitorInput(ports))

    # ========== Drawing ==========

    def draw(self, window: int, width: int, height: int) -> None:
        if window != 0:
            self._draw_show(width, height)
            return
        if self.screen_recording.playing:
            self._draw_screen_recording(width, height)      # the held take over the main window, the show waiting
            return
        if self.settings.debug:
            self._draw_panels(width, height)
        else:
            self._draw_show(width, height)      # the main panel alone, as on a second screen
        self._draw_screen_recording(width, height)

    def _draw_screen_recording(self, width: int, height: int) -> None:
        """The screen recording's tick, right after the main window is drawn: the main panel's layers at their shapes."""
        sources: list[render.ScreenRecordingSource] = [render.ScreenRecordingSource(self.L[layer_type][0], *self._layer_size(layer_type))
                                                       for layer_type in self._main_layers if 0 in self.L[layer_type]]
        self.screen_recording.set_input(render.ScreenRecordingInput(sources, width, height, self.window_manager.frame_time))
        self.screen_recording.update()

    def _draw_panels(self, width: int, height: int) -> None:
        """The main and the debug panel's layers, side by side or stacked."""
        clear_color()

        Style.set_blend_mode(Style.BlendMode.ALPHA)

        main_rect: Rect = self.subdivision.get_rect('panels', 0)
        self._draw_fit(self._main_layers, main_rect.x, height - main_rect.y - main_rect.height, main_rect.width, main_rect.height, True)

        debug_rect: Rect = self.subdivision.get_rect('panels', 1)
        self._draw_fit(self._debug_layers, debug_rect.x, height - debug_rect.y - debug_rect.height, debug_rect.width, debug_rect.height,
                       True, debug=True)

    def _draw_show(self, width: int, height: int) -> None:
        """The show: the main panel's layers in order, each fit to `width` × `height` at its own shape."""
        clear_color()

        Style.set_blend_mode(Style.BlendMode.DISABLED)
        Style.push_style()
        Style.set_blend_mode(Style.BlendMode.ADD)

        self._draw_fit(self._main_layers, 0, 0, width, height)

        Style.pop_style()

    def _draw_fit(self, layer_types: list[ChainLayers], x: float, y: float, width: float, height: float, border: bool = False,
                  debug: bool = False) -> None:
        """Draw each layer into a GL rect (origin bottom left) at its own shape, centered; with `border`, a border around
        each; in the debug panel (`debug`) a stage's debug monitor, else its main monitor."""
        if width <= 0 or height <= 0:
            return
        for layer_type in layer_types:
            layers: dict[int, render.LayerBase] = self.L[layer_type]
            if debug and layer_type in self._debug_monitors:
                layers = dict(self._debug_monitors[layer_type])
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
