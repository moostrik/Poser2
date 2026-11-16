"""Deep Flow render — split-view layer graph with the 3D flow: motion, bridge and the volumetric fluid."""

from threading import Event
from typing import Callable, cast

from OpenGL.GL import GL_RGBA16F, GL_RGBA, glViewport

from modules.gl import GpuTimer, RenderBase, ScreenCapture, Shader, Style, clear_color, fill, fit, Texture, MonitorId, MouseButton, shaders
from modules.video import ScreenRecorder, ScreenRecorderSettings
from modules.flow import Port, hold_tick
from modules.inference import Capture
from modules.pose import Frame
from modules import render
from modules.utils import FPS, Color, Rect, Point2f

from ..board import RenderBoard
from ..settings import MainView, PoseFrame, RenderSettings
from .layers import CaptureMonitorLayer, ViewInput, ViewOutput, ViewLayer, PoseMonitorCompositor
from .chain import ChainLayers, Size, TICK_LAYERS, CAPTURE_LAYERS, MONITORS, POSE_SHAPES, shown, size_of, layers_of, shape_of


OUTPUT_WIDTH: int = 1080
OUTPUT_HEIGHT: int = 1920

BORDER: Color = Color(0.4, 0.4, 0.4)    # the border around each layer in the main window, so a black layer shows
BORDER_WIDTH: float = 2.0       # px


class DeepFlowRender(RenderBase):
    def __init__(self, board: RenderBoard, settings: RenderSettings, fov: float, screen_recordings: str,
                 num_cams: int = 1, max_players: int = 1) -> None:
        """`fov` is the camera image's horizontal field (degrees), for the metres a view width spans;
        `screen_recordings` the folder the screen recordings go to."""
        super().__init__(settings.window)
        self.max_players: int = max_players
        self.num_cams: int = num_cams
        self.settings: RenderSettings = settings

        self.board: RenderBoard = board

        self._a_layers: list[ChainLayers] = shown(settings.layer.select.a)      # panel a, the show and the recording
        self._b_layers: list[ChainLayers] = shown(settings.layer.select.b)

        self.L: dict[ChainLayers, dict[int, render.LayerBase]] = {layer: {} for layer in ChainLayers}
        self._monitors_b: dict[ChainLayers, dict[int, render.MonitorLayer]] = {}     # each stage's monitor B, panel b's

        for i in range(self.num_cams):
            cam_image =     self.L[ChainLayers.cam_image][i] =   render.CameraImageLayer(    i, self.board)
            cam_crop =      self.L[ChainLayers.cam_crop][i] =    render.CropImageLayer(      i, self.board)
            self.L[ChainLayers.capture_image][i] =               render.CaptureImageLayer()
            self.L[ChainLayers.capture_depth][i] =               render.CaptureDepthLayer()

            # The stages, each with its monitor, and the pose's monitor
            view =          self.L[ChainLayers.view][i] =        ViewLayer(           i, settings.view, fov, settings.colors)
            self.L[ChainLayers.pose_monitor][i] =                PoseMonitorCompositor(i, self.board, cam_image, cam_crop, view, settings.pose, settings.colors)
            self.L[ChainLayers.motion][i] =                      render.Motion3DLayer(       i, settings.motion)
            self.L[ChainLayers.bridge][i] =                      render.Bridge3DLayer(       settings.bridge)
            fluid =         self.L[ChainLayers.fluid][i] =       render.Fluid3DLayer(        settings.fluid)
            self.L[ChainLayers.capture_monitor][i] = CaptureMonitorLayer(cast(render.CaptureImageLayer, self.L[ChainLayers.capture_image][i]),
                                                                    cast(render.CaptureDepthLayer, self.L[ChainLayers.capture_depth][i]),
                                                                    settings.capture.monitor, settings.view)
            # each stage's monitor A in self.L, shown in panel a; its monitor B in self._monitors_b, shown in panel b
            for key, stage, stage_settings in ((ChainLayers.view_monitor, render.FlowStage.VIEW, settings.view),
                                               (ChainLayers.motion_monitor, render.FlowStage.MOTION, settings.motion),
                                               (ChainLayers.bridge_monitor, render.FlowStage.BRIDGE, settings.bridge),
                                               (ChainLayers.fluid_monitor, render.FlowStage.FLUID, settings.fluid)):
                self.L[key][i] = render.MonitorLayer(stage, stage_settings.monitor_a)
                self._monitors_b.setdefault(key, {})[i] = render.MonitorLayer(stage, stage_settings.monitor_b)
        self.L[ChainLayers.time_series][0] = render.TimeSeriesLayer(self.board, settings.time_series)      # the board's series, one for all cameras

        # Split view: the left and right panels side by side or stacked, whichever shows what they hold larger
        self._window_size: tuple[int, int] = (settings.window.width, settings.window.height)
        self._panel_aspect_ratios: list[float] = self._read_panel_aspect_ratios()
        self.subdivision: render.Subdivision = self._make_layout()

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

        # Each camera's captures the motion measures, for the panel's cadence (read on any thread)
        self.motion_rates: dict[int, FPS] = {}
        self._capture_timestamps: dict[int, float] = {}     # each camera's capture run last

        # The border around each layer in the main window
        self._border: shaders.StrokeRect = shaders.StrokeRect()

        # The mouse in the main window drives the monitors' volume views: a drag that starts anywhere goes to
        # every monitor, so A and B keep one angle and one layer (nothing else in the main window takes it)
        self._drag: tuple[MouseButton, float, float] | None = None     # the button held and where the cursor was
        self._dragged: bool = False
        self.window_manager.add_mouse_button_callback(self._on_mouse_button)
        self.window_manager.add_mouse_move_callback(self._on_mouse_move)
        self.window_manager.add_mouse_scroll_callback(self._on_mouse_scroll)

        # The screen recording: panel a's layers, or the main window, drawn into the capture's screen frame
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
        for layer_types in (self._a_layers, self._b_layers):
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
        for monitors in self._monitors_b.values():
            for monitor in monitors.values():
                monitor.allocate(OUTPUT_WIDTH * 2, OUTPUT_HEIGHT * 2, GL_RGBA16F)     # a monitor renders at the size it is drawn at
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
        self.settings.screen_recorder.unbind(ScreenRecorderSettings.snap, self._on_snap)
        self.screen_recorder.stop()
        self.screen_recorder.deallocate()
        for cam_dict in self.L.values():
            for layer in cam_dict.values():
                layer.deallocate()
        for monitors in self._monitors_b.values():
            for monitor in monitors.values():
                monitor.deallocate()
        self._border.deallocate()
        self._view_timer.deallocate()

    def _measure_tick(self, now: float) -> None:
        if self._last_tick > 0.0:
            self._dt, self._tick_mean = hold_tick(now - self._last_tick, self._tick_mean)
        self._last_tick = now

    # ========== The mouse: the monitors' volume views ==========

    def _monitors(self) -> list[render.MonitorLayer]:
        return [cast(render.MonitorLayer, layer) for key in MONITORS
                for layer in (*self.L[key].values(), *self._monitors_b.get(key, {}).values())]

    def _on_mouse_button(self, x: float, y: float, button: MouseButton, pressed: bool) -> None:
        """A press starts a drag; a middle click that did not move resets the views."""
        if pressed:
            self._drag, self._dragged = (button, x, y), False
            return
        if self._drag is not None and self._drag[0] == MouseButton.MIDDLE and not self._dragged:
            for monitor in self._monitors():
                monitor.reset_view()
        self._drag = None

    def _on_mouse_move(self, x: float, y: float) -> None:
        if self._drag is None:
            return
        button, last_x, last_y = self._drag
        for monitor in self._monitors():
            monitor.drag(x - last_x, y - last_y, button)
        self._drag, self._dragged = (button, x, y), True

    def _on_mouse_scroll(self, steps: float) -> None:
        """A layer per step for every monitor, the show's density's layers."""
        for monitor in self._monitors():
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

        select = self.settings.layer.select
        self._a_layers = shown(select.a)
        self._b_layers = shown(select.b)
        self._update_layout()
        self._follow_view_size()

        Style.set_blend_mode(Style.BlendMode.ALPHA)
        seen: set[ChainLayers] = set()
        for layer_type in TICK_LAYERS + self._a_layers + self._b_layers:
            if layer_type in seen:
                continue
            if layer_type in CAPTURE_LAYERS:
                seen.update(CAPTURE_LAYERS)
                self._run_captures()
                continue
            seen.add(layer_type)
            wiring: Callable[[], None] | None = self._wirings.get(layer_type)
            if wiring is not None:
                wiring()
            for layer in self.L[layer_type].values():
                layer.update()
            for monitor in self._monitors_b.get(layer_type, {}).values():
                monitor.update()

    # ========== The flow's wiring ==========


    def _run_captures(self) -> None:
        """Every capture since the last tick, each camera's in order, through the capture layers (`_run_capture`).
        A tick without a capture runs none of them, so no stage works on a frame twice."""
        for i in self.L[ChainLayers.view]:
            for index, capture in enumerate(self.board.take_captures(i)):
                self._run_capture(i, capture, same_tick=index > 0)

    def _run_capture(self, i: int, capture: Capture, same_tick: bool) -> None:
        """One capture: its image and depth into GL, the view of it with its player's pose and its color, the
        motion on the view's image, and the bridge stepping over the time since the capture before, the density's
        color the view's. `same_tick`: another capture in this tick, whose bridge outputs the fluid takes with the
        last's."""
        image = cast(render.CaptureImageLayer, self.L[ChainLayers.capture_image][i])
        depth = cast(render.CaptureDepthLayer, self.L[ChainLayers.capture_depth][i])
        view = cast(ViewLayer, self.L[ChainLayers.view][i])
        motion = cast(render.Motion3DLayer, self.L[ChainLayers.motion][i])
        bridge = cast(render.Bridge3DLayer, self.L[ChainLayers.bridge][i])

        image.set_capture(capture)
        image.update()
        depth.set_capture(capture)
        depth.update()
        pose: Frame | None = capture.poses[min(capture.poses)] if capture.poses else None
        view.set_input(ViewInput(image.texture, depth.texture, capture.timestamp, pose))
        self._view_timer.begin()
        view.update()
        self._view_timer.end()

        interval: float = render.frame_interval(capture.timestamp, self._capture_timestamps.get(i), self.settings.motion.fps, self._dt, self._dt)
        self._capture_timestamps[i] = capture.timestamp
        self.motion_rates.setdefault(i, FPS()).processed()
        output: ViewOutput = view.output
        motion.set_input(render.MotionInput(output.image, capture.timestamp, interval, output.depth, output.mask, output.depth_mask))
        motion.update()
        bridge.set_input(render.BridgeInput(motion.output.velocity, interval, output.color,
                                        mask=output.mask, depth_mask=output.depth_mask, z_position=motion.output.z_position, same_tick=same_tick))
        bridge.update()

    def _wire_fluid(self) -> None:
        """Each fluid's input, one injection per bridge, at full weight."""
        dt: float = self._dt
        bridges: dict[int, render.Bridge3DLayer] = {i: cast(render.Bridge3DLayer, layer) for i, layer in self.L[ChainLayers.bridge].items()}
        for i, _fluid in self.L[ChainLayers.fluid].items():
            fluid = cast(render.Fluid3DLayer, _fluid)
            injections: list[render.Injection] = []
            for bridge in bridges.values():
                output: render.BridgeOutput = bridge.output
                injections.append(render.Injection(output.boundary_velocity, output.boundary_density, output.boundary_temperature,
                                               output.body_force_acceleration, output.body_force_density,
                                               output.body_force_temperature,
                                               boundary_velocity_z_position=output.boundary_velocity_z_position,
                                               boundary_density_z_position=output.boundary_density_z_position,
                                               boundary_temperature_z_position=output.boundary_temperature_z_position,
                                               body_force_acceleration_z_position=output.body_force_acceleration_z_position,
                                               body_force_density_z_position=output.body_force_density_z_position,
                                               body_force_temperature_z_position=output.body_force_temperature_z_position,
                                               expansion=output.expansion, expansion_z_position=output.expansion_z_position))
            fluid.set_input(render.FluidInput(dt, injections))

    def _wire_monitors(self) -> None:
        """The monitors' input, every stage's ports; each monitor shows its own stage's."""
        for i, _motion in self.L[ChainLayers.motion].items():
            if i not in self.L[ChainLayers.bridge] or i not in self.L[ChainLayers.fluid] or i not in self.L[ChainLayers.view]:
                continue
            ports: dict[render.FlowStage, dict[str, Port]] = {
                render.FlowStage.VIEW: cast(ViewLayer, self.L[ChainLayers.view][i]).ports,
                render.FlowStage.MOTION: cast(render.Motion3DLayer, _motion).ports,
                render.FlowStage.BRIDGE: cast(render.Bridge3DLayer, self.L[ChainLayers.bridge][i]).ports,
                render.FlowStage.FLUID: cast(render.Fluid3DLayer, self.L[ChainLayers.fluid][i]).ports,
            }
            for monitor in self._monitors():
                monitor.set_input(render.MonitorInput(ports))

    # ========== Drawing ==========

    def draw_main(self, width: int, height: int) -> None:
        if self.settings.main_view == MainView.SHOW:
            self._draw_show(width, height)      # the show alone, on one screen
        else:
            self._draw_panels(width, height)
        self._draw_screen_recording(width, height)

    def _draw_screen_recording(self, width: int, height: int) -> None:
        """The screen frame, when the recorder records this tick: panel a's layers, each fit and centered at its
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
                for layer_type in self._a_layers:
                    if 0 in self.L[layer_type]:
                        x, y, w, h = place(*self._layer_size(layer_type), capture.width, capture.height)
                        glViewport(round(x), round(y), round(w), round(h))
                        self.L[layer_type][0].draw()
                Style.pop_style()
                capture.frame.end()
            glViewport(0, 0, width, height)
        self.screen_recorder.update(self.window_manager.frame_time)

    def _on_snap(self, _: bool) -> None:
        """The recording's size: panel a's first layer's, or the main window's when it is recorded."""
        recorder_settings: ScreenRecorderSettings = self.settings.screen_recorder
        record: list[ChainLayers] = self._a_layers
        if recorder_settings.main_window:
            width, height = self._window_size
        elif record:
            width, height = self._layer_size(record[0])
        else:
            return
        recorder_settings.width = width
        recorder_settings.height = height

    def _draw_panels(self, width: int, height: int) -> None:
        """The debug panels: panel a's and panel b's layers, side by side or stacked."""
        clear_color()

        Style.set_blend_mode(Style.BlendMode.ALPHA)

        a_rect: Rect = self.subdivision.get_rect('panels', 0)
        self._draw_fit(self._a_layers, a_rect.x, height - a_rect.y - a_rect.height, a_rect.width, a_rect.height, True)

        b_rect: Rect = self.subdivision.get_rect('panels', 1)
        self._draw_fit(self._b_layers, b_rect.x, height - b_rect.y - b_rect.height, b_rect.width, b_rect.height, True, panel_b=True)

    def draw_secondary(self, monitor_id: int, width: int, height: int) -> None:
        self._draw_show(width, height)

    def _draw_show(self, width: int, height: int) -> None:
        """The show: panel a's layers in order, each fit to `width` × `height` at its own shape."""
        clear_color()

        Style.set_blend_mode(Style.BlendMode.DISABLED)
        Style.push_style()
        Style.set_blend_mode(Style.BlendMode.ADD)

        self._draw_fit(self._a_layers, 0, 0, width, height)

        Style.pop_style()

    def _draw_fit(self, layer_types: list[ChainLayers], x: float, y: float, width: float, height: float, border: bool = False,
                  panel_b: bool = False) -> None:
        """Draw each layer into a GL rect (origin bottom left) at its own shape, centered; with `border`, a border around
        each; in `panel_b` a stage's monitor B, else its monitor A."""
        if width <= 0 or height <= 0:
            return
        for layer_type in layer_types:
            layers: dict[int, render.LayerBase] = self.L[layer_type]
            if panel_b and layer_type in self._monitors_b:
                layers = dict(self._monitors_b[layer_type])
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
