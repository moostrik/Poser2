"""Deep Flow render — split-view layer graph with the 3D flow: motion, bridge and the volumetric fluid."""

from threading import Event
from typing import cast

import numpy as np
from OpenGL.GL import GL_RGBA16F, GL_RGBA, glViewport

from modules.gl import RenderBase, Shader, Style, clear_color, fit, Texture, MonitorId, MouseButton
from modules.flow import DepthView, Port, hold_tick
from modules.pose import Frame, Similarity, AngleMotion, Points2D, PointLandmark
from modules.render import layers as ls, shaders, make_subdivision, best_grid, grid_cell_aspect_ratio, SubdivisionRow, Subdivision
from modules.utils import Rect, Point2f

from modules.board import HasFrames

from .capture_poses import CapturePoses
from .render_board import RenderBoard
from .settings import Layers, MaskSource, RenderSettings, FlowInputSettings


UPDATE_LAYERS: list[Layers] = [
    Layers.cam_image,
    Layers.cam_crop,
    Layers.cam_depth,
    Layers.capture_image,
    Layers.cam_mask,            # after the depth it may cut
    Layers.cam_foreground,

    Layers.center_geometry,
    Layers.center_mask,
    Layers.center_capture,      # the flow's input: the capture's image, unmasked
    Layers.center_depth,        # after the image it can lie over
    Layers.center_foreground,
    Layers.center_pose,

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

OUTPUT_WIDTH: int = 1080
OUTPUT_HEIGHT: int = 1920

EYE_EASE: float = 0.4           # s: the parallax eye's ease toward the tracked head, keeping keypoint jitter out
BORDER: tuple[float, float, float, float] = (0.4, 0.4, 0.4, 1.0)    # the border around each layer in the main window, so a black layer shows
BORDER_WIDTH: float = 2.0       # px

# The layers that are not output-shaped, each with the layer whose texture holds its shape
SHAPE_LAYERS: dict[Layers, Layers] = {
    Layers.cam_image:      Layers.cam_image,
    Layers.cam_mask:       Layers.cam_mask,
    Layers.cam_foreground: Layers.cam_foreground,
    Layers.cam_crop:       Layers.cam_crop,
    Layers.cam_depth:      Layers.cam_depth,
    Layers.capture_image:  Layers.capture_image,
    Layers.tracker:        Layers.cam_image,  # camera stretched into a panel-sized FBO
    Layers.pose:           Layers.cam_crop,   # crop stretched into a panel-sized FBO
}


class DeepFlowRender(RenderBase):
    def __init__(self, board: RenderBoard, settings: RenderSettings, mask_source: MaskSource, fov: float, num_cams: int = 1, max_players: int = 1) -> None:
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

        cmt: dict[int, Texture] = {}    # the center masks
        depth_source: bool = mask_source == MaskSource.DEPTH
        space: ls.SourceSpace = ls.SourceSpace.IMAGE if depth_source else ls.SourceSpace.CROP
        for i in range(self.num_cams):
            cam_image =     self.L[Layers.cam_image][i] =   ls.ImageSourceLayer(    i, self.board)
            cam_crop =      self.L[Layers.cam_crop][i] =    ls.CropSourceLayer(     i, self.board)
            cam_depth =     self.L[Layers.cam_depth][i] =   ls.DepthSourceLayer(    i, self.board,   cam_image.texture,          settings.depth)
            capture_image = self.L[Layers.capture_image][i] = ls.CaptureImageLayer(i, self.board)

            # The player's mask and the foreground's source, from the mask source chosen at start-up
            cam_mask: ls.LayerBase
            foreground_source: ls.LayerBase
            if depth_source:
                cam_mask = self.L[Layers.cam_mask][i] =     ls.DepthMaskLayer(      cam_depth)
                foreground_source = capture_image           # masked in the center view (`premultiply`)
            else:
                cam_mask = self.L[Layers.cam_mask][i] =     ls.MaskSourceLayer(     i, self.board)
                cam_foreground = self.L[Layers.cam_foreground][i] = ls.ForegroundSourceLayer(i, self.board)
                foreground_source = cam_foreground          # RVM's, masked already

            tracker_compositor = self.L[Layers.tracker][i] = ls.TrackerCompositor(  i, self.board,   cam_image.texture,          settings.preview.tracker,   settings.colors)
            pose_compositor = self.L[Layers.pose][i] =      ls.PoseCompositor(      i, self.board,   cam_image.texture, cam_crop, settings.preview.pose,      settings.colors)

            # placed by the pose of the capture it shows, so view, image and depth are one moment
            center_geometry = self.L[Layers.center_geometry][i] = ls.CenterGeometry( i, cast(HasFrames, CapturePoses(self.board)), cam_image.texture, settings.center.geometry, fov)
            center_mask =   self.L[Layers.center_mask][i] = ls.CenterMaskLayer(     i, center_geometry, cam_mask,               space, settings.center.mask)
            cmt[i] = center_mask.texture
            center_cam =    self.L[Layers.center_cam][i] =  ls.CenterCamLayer(      i, center_geometry, cam_image,          cmt[i], settings.center.cam)
            center_capture = self.L[Layers.center_capture][i] = ls.CenterCamLayer(  i, center_geometry, capture_image,      cmt[i], settings.center.capture)
            self.L[Layers.center_depth][i] =                ls.CenterDepthLayer(    i, center_geometry, cam_depth, center_capture, cmt[i], settings.center.depth, settings.depth)
            center_foreground = self.L[Layers.center_foreground][i] = ls.CenterForegroundLayer(i, center_geometry, foreground_source, cmt[i], space, depth_source,
                                                                                                settings.center.foreground, settings.colors)
            center_pose =   self.L[Layers.center_pose][i] = ls.CenterPoseLayer(     i, center_geometry,                             settings.center.pose,       settings.colors)

            ms_mask =       self.L[Layers.color_mask][i] =  ls.MSColorMaskLayer(    i, self.board,   center_foreground.texture, cmt, settings.center.color, settings.colors)
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

        # The parallax eye per fluid, eased toward the tracked head
        self._eyes: dict[int, tuple[float, float]] = {}

        # A reset asked for from another thread, applied at the next update on the GL thread
        self._reset_fluid: Event = Event()

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
        for layer_type, cam_dict in self.L.items():
            for layer in cam_dict.values():
                if layer_type in LARGE_LAYERS:
                    layer.allocate(OUTPUT_WIDTH * 2, OUTPUT_HEIGHT * 2, GL_RGBA16F)
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

    def deallocate(self) -> None:
        for cam_dict in self.L.values():
            for layer in cam_dict.values():
                layer.deallocate()
        self._border.deallocate()

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
            monitor.scroll(steps, self.settings.fluid.volume.layers)

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

        self._wire_flow()

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
            if layer_type not in seen:
                seen.add(layer_type)
                for layer in self.L[layer_type].values():
                    layer.update()

    # ========== The flow's wiring ==========

    def _layer(self, key: Layers, cam_id: int) -> ls.LayerBase | None:
        return self.L[key][cam_id] if key in self.L and cam_id in self.L[key] else None

    def _wire_flow(self) -> None:
        """Build each stage's input from the layers the panel names and the stage before: a motion and
        a bridge input per camera, a fluid input per fluid with one injection per bridge, at its weight,
        and the monitors' input from every stage's ports."""
        inputs: FlowInputSettings = self.settings.inputs
        dt: float = self._dt
        bridges: dict[int, ls.Bridge3DLayer] = {i: cast(ls.Bridge3DLayer, layer) for i, layer in self.L[Layers.bridge].items()}

        for i, _motion in self.L[Layers.motion].items():
            motion = cast(ls.Motion3DLayer, _motion)
            image: ls.LayerBase | None = self._layer(inputs.motion_image, i)
            if image is None or i not in bridges:
                continue
            timestamp: float | None = image.timestamp if isinstance(image, ls.HasCaptureTime) else None
            color: ls.LayerBase | None = self._layer(inputs.bridge_image, i)
            mask_layer: ls.LayerBase | None = self._layer(inputs.mask, i)
            mask: Texture | None = mask_layer.texture if mask_layer is not None else None
            depth: DepthView = cast(ls.CenterDepthLayer, self.L[Layers.center_depth][i]).depth_view
            motion.set_input(ls.MotionInput(image.texture, timestamp, dt, depth, mask))
            bridges[i].set_input(ls.BridgeInput(motion.output.velocity, dt, color.texture if color is not None else None, mask=mask))

        for i, _fluid in self.L[Layers.fluid].items():
            fluid = cast(ls.Fluid3DLayer, _fluid)
            pose: Frame | None = self.board.get_frame(self.settings.stage, i)
            injections: list[ls.Injection] = []
            for j, bridge in bridges.items():
                output: ls.BridgeOutput = bridge.output
                injections.append(ls.Injection(output.velocity, output.density, output.temperature, output.onset, output.spin,
                                               self._weight(pose, i, j)))
            fluid.set_input(ls.FluidInput(dt, injections))
            fluid.set_eye(*self._eye(i, pose, dt))

        for i, _motion in self.L[Layers.motion].items():
            if i not in bridges or i not in self.L[Layers.fluid]:
                continue
            ports: dict[ls.FlowStage, dict[str, Port]] = {
                ls.FlowStage.MOTION: cast(ls.Motion3DLayer, _motion).ports,
                ls.FlowStage.BRIDGE: bridges[i].ports,
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

    def _eye(self, cam_id: int, pose: Frame | None, dt: float) -> tuple[float, float]:
        """The parallax eye of camera `cam_id`'s fluid, eased toward the tracked head: the eyes' mean,
        the nose when they are missing, easing back to center when both are. Offsets are ±1 across
        the frame (vertical at half range)."""
        target_x, target_y = 0.0, 0.0
        if pose is not None and Points2D in pose:
            pts = pose[Points2D].values
            head = pts[[PointLandmark.left_eye, PointLandmark.right_eye]].mean(axis=0)
            if np.isnan(head).any():
                head = pts[PointLandmark.nose]
            if not np.isnan(head).any():
                target_x = (float(head[0]) - 0.5) * 2.0
                target_y = float(head[1]) - 0.5
        x, y = self._eyes.get(cam_id, (0.0, 0.0))
        k: float = 1.0 - 0.01 ** (dt / EYE_EASE)
        x += (target_x - x) * k
        y += (target_y - y) * k
        self._eyes[cam_id] = (x, y)
        return x, y

    # ========== Drawing ==========

    def draw_main(self, width: int, height: int) -> None:
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
