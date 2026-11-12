"""Deep Flow render — split-view layer graph with the 3D flow: motion, bridge and the volumetric fluid."""

from threading import Event
from typing import Callable, cast

import numpy as np
from OpenGL.GL import GL_RGBA16F, GL_RGBA, glViewport

from modules.gl import GpuTimer, RenderBase, Shader, Style, clear_color, fit, Texture, MonitorId, MouseButton
from modules.flow import Port, hold_tick
from modules.inference import Capture
from modules.pose import Frame, Similarity, AngleMotion
from modules.render import layers as ls, shaders, make_subdivision, best_grid, grid_cell_aspect_ratio, SubdivisionRow, Subdivision
from modules.utils import FPS, Rect, Point2f

from .render_board import RenderBoard
from .settings import Layers, MainView, RenderSettings, FlowInputSettings


UPDATE_LAYERS: list[Layers] = [
    Layers.cam_image,
    Layers.cam_crop,
    Layers.capture_image,
    Layers.capture_depth,

    Layers.view,                # the flow's view of the capture: image, depth, masks, foreground
    Layers.view_depth,
    Layers.view_pose,
    Layers.foreground_look,     # the density's color, the look of the view's foreground

    Layers.color_mask,
    Layers.motion,
    Layers.bridge,
    Layers.fluid,
    Layers.monitor_a,
    Layers.monitor_b,
    Layers.composite,
]

INTERFACE_LAYERS: list[Layers] = [
    Layers.tracker,
    Layers.pose,
]

LARGE_LAYERS: list[Layers] = [
    Layers.motion,
    Layers.bridge,
    Layers.fluid,
    Layers.monitor_a,
    Layers.monitor_b,
    Layers.composite,
]

# Run once per capture, in this order, in place of the first of them in UPDATE_LAYERS (`_run_captures`)
CAPTURE_LAYERS: list[Layers] = [
    Layers.capture_image,
    Layers.capture_depth,
    Layers.view,
    Layers.foreground_look,
    Layers.motion,
    Layers.bridge,
]

# The view stage and what is drawn from it, at the motion's size
VIEW_LAYERS: list[Layers] = [
    Layers.view,
    Layers.view_image,
    Layers.view_foreground,
    Layers.view_depth,
    Layers.view_mask,
    Layers.view_depth_mask,
    Layers.view_pose,
    Layers.foreground_look,
    Layers.color_mask,
]

OUTPUT_WIDTH: int = 1080
OUTPUT_HEIGHT: int = 1920

BORDER: tuple[float, float, float, float] = (0.4, 0.4, 0.4, 1.0)    # the border around each layer in the main window, so a black layer shows
BORDER_WIDTH: float = 2.0       # px

# The layers that are not output-shaped, each with the layer whose texture holds its shape
SHAPE_LAYERS: dict[Layers, Layers] = {
    Layers.cam_image:      Layers.cam_image,
    Layers.cam_crop:       Layers.cam_crop,
    Layers.capture_image:  Layers.capture_image,
    Layers.capture_depth:  Layers.capture_image,  # on the image's view, at the depth's own grid
    Layers.tracker:        Layers.cam_image,  # camera stretched into a panel-sized FBO
    Layers.pose:           Layers.cam_crop,   # crop stretched into a panel-sized FBO
    **{layer: Layers.view_image for layer in VIEW_LAYERS},
}


class DeepFlowRender(RenderBase):
    def __init__(self, board: RenderBoard, settings: RenderSettings, fov: float, num_cams: int = 1, max_players: int = 1) -> None:
        """`fov` is the camera image's horizontal field (degrees), for the metres a view width spans."""
        super().__init__(settings.window)
        self.max_players: int = max_players
        self.num_cams: int = num_cams
        self.settings: RenderSettings = settings

        self.board: RenderBoard = board

        self._update_layers: list[Layers] = UPDATE_LAYERS
        self._interface_layers: list[Layers] = INTERFACE_LAYERS
        self._left_layers: list[Layers] = settings.layer.select.left
        self._right_layers: list[Layers] = settings.layer.select.right
        self._output_layers: list[Layers] = settings.layer.select.output

        self.L: dict[Layers, dict[int, ls.LayerBase]] = {layer: {} for layer in Layers}

        # Set data.b overrides
        settings.data.b.line_width = 6.0
        settings.data.b.line_smooth = 6.0
        settings.data.b.use_history_color = True

        masks: dict[int, Texture] = {}     # the view's masks, for the color mask
        for i in range(self.num_cams):
            cam_image =     self.L[Layers.cam_image][i] =   ls.ImageSourceLayer(    i, self.board)
            cam_crop =      self.L[Layers.cam_crop][i] =    ls.CropSourceLayer(     i, self.board)
            self.L[Layers.capture_image][i] =               ls.CaptureImageLayer()
            self.L[Layers.capture_depth][i] =               ls.CaptureDepthLayer()

            tracker_compositor = self.L[Layers.tracker][i] = ls.TrackerCompositor(  i, self.board,   cam_image.texture,          settings.preview.tracker,   settings.colors)
            pose_compositor = self.L[Layers.pose][i] =      ls.PoseCompositor(      i, self.board,   cam_image.texture, cam_crop, settings.preview.pose,      settings.colors)

            # The flow's view of the capture, and what is drawn from it
            view =          self.L[Layers.view][i] =        ls.ViewLayer(           settings.view, settings.motion.z, fov)
            output: ls.ViewOutput = view.output
            masks[i] = output.mask
            self.L[Layers.view_image][i] =                  ls.ViewOutputLayer(     view, output.image)
            self.L[Layers.view_foreground][i] =             ls.ViewOutputLayer(     view, output.foreground)
            self.L[Layers.view_mask][i] =                   ls.ViewOutputLayer(     view, output.mask)
            self.L[Layers.view_depth_mask][i] =             ls.ViewOutputLayer(     view, output.depth_mask)
            self.L[Layers.view_depth][i] =                  ls.ViewDepthLayer(      view, settings.view_depth)
            self.L[Layers.view_pose][i] =                   ls.CenterPoseLayer(     i, view,                                     settings.view_pose,         settings.colors)
            view_foreground: ls.LayerBase = self.L[Layers.view_foreground][i]
            foreground_look = self.L[Layers.foreground_look][i] = ls.ForegroundLookLayer(i, view_foreground, output.mask,           settings.foreground_look, settings.colors)

            ms_mask =       self.L[Layers.color_mask][i] =  ls.MSColorMaskLayer(    i, self.board,   foreground_look.texture,   masks, settings.color_mask, settings.colors)
            self.L[Layers.motion][i] =                      ls.Motion3DLayer(       i, settings.motion)
            self.L[Layers.bridge][i] =                      ls.Bridge3DLayer(       settings.bridge)
            fluid =         self.L[Layers.fluid][i] =       ls.Fluid3DLayer(        settings.fluid)
            self.L[Layers.monitor_a][i] =                   ls.MonitorLayer(        settings.monitor_a)
            self.L[Layers.monitor_b][i] =                   ls.MonitorLayer(        settings.monitor_b)

            composite =    self.L[Layers.composite][i] =   ls.CompositeLayer(                          [fluid, ms_mask],         settings.layer.composite)

            self.L[Layers.data_A_W][i]  = ls.FeatureWindowLayer(i, self.board, settings.data.a, settings.colors)
            self.L[Layers.data_A_F][i]  = ls.FeatureFrameLayer( i, self.board, settings.data.a, settings.colors)
            self.L[Layers.data_B_W][i]  = ls.FeatureWindowLayer(i, self.board, settings.data.b, settings.colors)
            self.L[Layers.data_B_F][i]  = ls.FeatureFrameLayer( i, self.board, settings.data.b, settings.colors)
            self.L[Layers.data_time][i] = ls.MotionTimeRenderer(i, self.board)
        self.L[Layers.time_series][0] = ls.TimeSeriesLayer(self.board, settings.time_series)      # the board's series, one for all cameras

        # Split view: the left and right panels side by side or stacked, whichever shows what they hold larger
        self._window_size: tuple[int, int] = (settings.window.width, settings.window.height)
        self._panel_aspect_ratios: list[float] = self._read_panel_aspect_ratios()
        self.subdivision: Subdivision = self._make_layout()

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
        self._wirings: dict[Layers, Callable[[], None]] = {
            Layers.fluid: self._wire_fluid,
            Layers.monitor_a: self._wire_monitors,
        }

        # Each camera's captures the motion measures, for the panel's cadence (read on any thread)
        self.motion_rates: dict[int, FPS] = {}
        self._capture_timestamps: dict[int, float] = {}     # each camera's capture run last

        # The border around each layer in the main window
        self._border: shaders.DrawRectangleOutline = shaders.DrawRectangleOutline()

        # The mouse in the main window drives the monitors' volume views: a drag that starts anywhere goes to
        # every monitor, so A and B keep one angle and one layer (nothing else in the main window takes it)
        self._drag: tuple[MouseButton, float, float] | None = None     # the button held and where the cursor was
        self._dragged: bool = False
        self.window_manager.add_mouse_button_callback(self._on_mouse_button)
        self.window_manager.add_mouse_move_callback(self._on_mouse_move)
        self.window_manager.add_mouse_scroll_callback(self._on_mouse_scroll)

    def _make_layout(self) -> Subdivision:
        width, height = self._window_size
        aspect_ratios: list[float] = self._panel_aspect_ratios
        columns, rows = best_grid(aspect_ratios, width, height)
        aspect_ratio: float = grid_cell_aspect_ratio(aspect_ratios, columns, rows, width, height)
        panels = SubdivisionRow(name='panels', columns=columns, rows=rows, src_aspect_ratio=aspect_ratio, padding=Point2f(1.0, 1.0))
        return make_subdivision([panels], width, height, True)

    def _read_panel_aspect_ratios(self) -> list[float]:
        """The shape of each panel, the shape of its first layer."""
        aspect_ratios: list[float] = []
        for layer_types in (self._left_layers, self._right_layers):
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
        self._view_size = (self.settings.motion.width, self.settings.motion.height)
        for layer_type, cam_dict in self.L.items():
            for layer in cam_dict.values():
                if layer_type in LARGE_LAYERS:
                    layer.allocate(OUTPUT_WIDTH * 2, OUTPUT_HEIGHT * 2, GL_RGBA16F)
                elif layer_type in VIEW_LAYERS:
                    layer.allocate(*self._view_size, GL_RGBA16F)
                else:
                    layer.allocate(OUTPUT_WIDTH, OUTPUT_HEIGHT, GL_RGBA16F)
        self.allocate_window_renders()
        self._border.allocate()
        Shader.enable_hot_reload()

    def allocate_window_renders(self) -> None:
        # Allocate interface layers to split-view panel sizes
        for layer_type in INTERFACE_LAYERS:
            for i in range(self.num_cams):
                w, h = self.subdivision.get_allocation_size('panels', 0)
                self.L[layer_type][i].allocate(w, h, GL_RGBA)

    def _follow_view_size(self) -> None:
        """Reallocate the view layers when the motion's size changed: the view is drawn at the size the
        motion and the bridge read it."""
        size: tuple[int, int] = (self.settings.motion.width, self.settings.motion.height)
        if size == self._view_size:
            return
        self._view_size = size
        for layer_type in VIEW_LAYERS:
            for layer in self.L[layer_type].values():
                layer.deallocate()
                layer.allocate(*size, GL_RGBA16F)

    def deallocate(self) -> None:
        for cam_dict in self.L.values():
            for layer in cam_dict.values():
                layer.deallocate()
        self._border.deallocate()
        self._view_timer.deallocate()

    def _measure_tick(self, now: float) -> None:
        if self._last_tick > 0.0:
            self._dt, self._tick_mean = hold_tick(now - self._last_tick, self._tick_mean)
        self._last_tick = now

    # ========== The mouse: the monitors' volume views ==========

    def _monitors(self) -> list[ls.MonitorLayer]:
        return [cast(ls.MonitorLayer, layer) for key in (Layers.monitor_a, Layers.monitor_b) for layer in self.L[key].values()]

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
            for layer_type in [Layers.motion, Layers.bridge, Layers.fluid]:
                for layer in self.L[layer_type].values():
                    cast(ls.StageLayer, layer).reset()

        select = self.settings.layer.select
        self._left_layers = select.left
        self._right_layers = select.right
        self._output_layers = select.output
        self._update_layout()
        self._follow_view_size()

        for i, _composite_layer in self.L[Layers.composite].items():
            composite_layer = cast(ls.CompositeLayer, _composite_layer)
            composite_layer.set_layers([
                self.L[key][i] for key in self._output_layers
                if key is not Layers.composite and key in self.L and i in self.L[key]   # it cannot draw into itself
            ])

        Style.reset_state()
        Style.set_blend_mode(Style.BlendMode.ALPHA)
        seen: set[Layers] = set()
        for layer_type in self._update_layers + self._interface_layers + self._left_layers + self._right_layers + self._output_layers:
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

    # ========== The flow's wiring ==========

    def _layer(self, key: Layers, cam_id: int) -> ls.LayerBase | None:
        return self.L[key][cam_id] if key in self.L and cam_id in self.L[key] else None

    def _run_captures(self) -> None:
        """Every capture since the last tick, each camera's in order, through the capture layers (`_run_capture`).
        A tick without a capture runs none of them, so no stage works on a frame twice."""
        for i in self.L[Layers.view]:
            for index, capture in enumerate(self.board.take_captures(i)):
                self._run_capture(i, capture, same_tick=index > 0)

    def _run_capture(self, i: int, capture: Capture, same_tick: bool) -> None:
        """One capture: its image and depth into GL, the view of it with its player's pose, the foreground's look,
        the motion, and the bridge stepping over the time since the capture before, from the pictures the panel
        names. `same_tick`: another capture in this tick, whose bridge outputs the fluid takes with the last's."""
        inputs: FlowInputSettings = self.settings.inputs
        image = cast(ls.CaptureImageLayer, self.L[Layers.capture_image][i])
        depth = cast(ls.CaptureDepthLayer, self.L[Layers.capture_depth][i])
        view = cast(ls.ViewLayer, self.L[Layers.view][i])
        motion = cast(ls.Motion3DLayer, self.L[Layers.motion][i])
        bridge = cast(ls.Bridge3DLayer, self.L[Layers.bridge][i])

        image.set_capture(capture)
        image.update()
        depth.set_capture(capture)
        depth.update()
        pose: Frame | None = capture.poses[min(capture.poses)] if capture.poses else None
        view.set_input(ls.ViewInput(image.texture, depth.texture, capture.timestamp, pose))
        self._view_timer.begin()
        view.update()
        self._view_timer.end()
        self.L[Layers.foreground_look][i].update()

        interval: float = ls.frame_interval(capture.timestamp, self._capture_timestamps.get(i), self.settings.motion.fps, self._dt, self._dt)
        self._capture_timestamps[i] = capture.timestamp
        self.motion_rates.setdefault(i, FPS(120)).processed()
        motion_image: ls.LayerBase = self._layer(inputs.motion_image, i) or self.L[Layers.view_image][i]
        color: ls.LayerBase | None = self._layer(inputs.bridge_image, i)
        output: ls.ViewOutput = view.output
        motion.set_input(ls.MotionInput(motion_image.texture, capture.timestamp, interval, output.depth, output.mask, output.depth_mask))
        motion.update()
        bridge.set_input(ls.BridgeInput(motion.output.velocity, interval, color.texture if color is not None else None,
                                        mask=output.mask, z_position=motion.output.z_position, same_tick=same_tick))
        bridge.update()

    def _wire_fluid(self) -> None:
        """Each fluid's input, one injection per bridge, at its weight."""
        dt: float = self._dt
        bridges: dict[int, ls.Bridge3DLayer] = {i: cast(ls.Bridge3DLayer, layer) for i, layer in self.L[Layers.bridge].items()}
        for i, _fluid in self.L[Layers.fluid].items():
            fluid = cast(ls.Fluid3DLayer, _fluid)
            pose: Frame | None = self.board.get_frame(self.settings.stage, i)
            injections: list[ls.Injection] = []
            for j, bridge in bridges.items():
                output: ls.BridgeOutput = bridge.output
                injections.append(ls.Injection(output.boundary_velocity, output.boundary_density, output.boundary_temperature,
                                               output.body_force_acceleration, output.body_force_density,
                                               output.body_force_temperature, weight=self._weight(pose, i, j),
                                               boundary_velocity_z_position=output.boundary_velocity_z_position,
                                               body_force_acceleration_z_position=output.body_force_acceleration_z_position,
                                               z_position=output.z_position))
            fluid.set_input(ls.FluidInput(dt, injections))

    def _wire_monitors(self) -> None:
        """The monitors' input, every stage's ports."""
        for i, _motion in self.L[Layers.motion].items():
            if i not in self.L[Layers.bridge] or i not in self.L[Layers.fluid] or i not in self.L[Layers.view]:
                continue
            ports: dict[ls.FlowStage, dict[str, Port]] = {
                ls.FlowStage.VIEW: cast(ls.ViewLayer, self.L[Layers.view][i]).ports,
                ls.FlowStage.MOTION: cast(ls.Motion3DLayer, _motion).ports,
                ls.FlowStage.BRIDGE: cast(ls.Bridge3DLayer, self.L[Layers.bridge][i]).ports,
                ls.FlowStage.FLUID: cast(ls.Fluid3DLayer, self.L[Layers.fluid][i]).ports,
            }
            for key in (Layers.monitor_a, Layers.monitor_b):
                if i in self.L[key]:
                    cast(ls.MonitorLayer, self.L[key][i]).set_input(ls.MonitorInput(ports))

    def _weight(self, pose: Frame | None, own_cam: int, cam_id: int) -> float:
        """The share of camera `cam_id`'s bridge in camera `own_cam`'s fluid: its player's motion for the
        fluid's own camera, the cross-camera similarity for another; 1 without `scale_by_motion`."""
        if not self.settings.inputs.scale_by_motion:
            return 1.0
        if pose is None:
            return 0.0
        if cam_id == own_cam:
            return float(pose[AngleMotion].value) if AngleMotion in pose else 0.0
        if Similarity in pose:
            similarities: np.ndarray = np.nan_to_num(pose[Similarity].values)
            return float(similarities[cam_id]) if cam_id < len(similarities) else 0.0
        return 0.0

    # ========== Drawing ==========

    def draw_main(self, width: int, height: int) -> None:
        if self.settings.main_view == MainView.SHOW:
            self._draw_show(width, height)      # the show alone, on one screen
            return

        clear_color()

        Style.reset_state()
        Style.set_blend_mode(Style.BlendMode.ALPHA)

        # Left panel
        left_rect: Rect = self.subdivision.get_rect('panels', 0)
        self._draw_fit(self._left_layers, left_rect.x, height - left_rect.y - left_rect.height, left_rect.width, left_rect.height, True)

        # Right panel
        right_rect: Rect = self.subdivision.get_rect('panels', 1)
        self._draw_fit(self._right_layers, right_rect.x, height - right_rect.y - right_rect.height, right_rect.width, right_rect.height, True)

    def draw_secondary(self, monitor_id: int, width: int, height: int) -> None:
        self._draw_show(width, height)

    def _draw_show(self, width: int, height: int) -> None:
        """The show: the composite alone, fit to `width` × `height`."""
        clear_color()

        Style.reset_state()
        Style.push_style()
        Style.set_blend_mode(Style.BlendMode.ADD)

        self._draw_fit([Layers.composite], 0, 0, width, height)

        Style.pop_style()

    def _draw_fit(self, layer_types: list[Layers], x: float, y: float, width: float, height: float, border: bool = False) -> None:
        """Draw each layer into a GL rect (origin bottom left) at its own shape, centered; with `border`, a border around each."""
        if width <= 0 or height <= 0:
            return
        for layer_type in layer_types:
            if 0 in self.L[layer_type]:
                fit_x, fit_y, fit_width, fit_height = fit(*self._layer_size(layer_type), width, height)
                glViewport(int(x + fit_x), int(y + fit_y), int(fit_width), int(fit_height))
                self.L[layer_type][0].draw()
                if border:
                    self._draw_border(fit_width, fit_height)

    def _draw_border(self, width: float, height: float) -> None:
        """A border around the current viewport, `width` by `height` pixels."""
        if width <= 0 or height <= 0:
            return
        Style.push_style()
        Style.set_blend_mode(Style.BlendMode.ALPHA)
        self._border.use(0.0, 0.0, 1.0, 1.0, *BORDER, BORDER_WIDTH / width, BORDER_WIDTH / height)
        Style.pop_style()

    def _layer_size(self, layer_type: Layers) -> tuple[int, int]:
        if layer_type in SHAPE_LAYERS and 0 in self.L[SHAPE_LAYERS[layer_type]]:      # cam_foreground exists with RVM only
            texture: Texture = self.L[SHAPE_LAYERS[layer_type]][0].texture
            if texture.allocated:
                return texture.width, texture.height
        return OUTPUT_WIDTH, OUTPUT_HEIGHT
