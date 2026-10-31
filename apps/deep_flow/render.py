"""Deep Flow render — split-view layer graph with 3D volumetric fluid."""

from threading import Event
from typing import cast

from OpenGL.GL import GL_RGBA16F, GL_RGBA, glViewport

from modules.gl import RenderBase, Shader, Style, clear_color, fit, Texture, MonitorId
from modules.render import layers as ls, make_subdivision, best_grid, SubdivisionRow, Subdivision
from modules.utils import Rect, Point2f, HotReloadMethods

from .render_board import RenderBoard
from .settings import Layers, RenderSettings


UPDATE_LAYERS: list[Layers] = [
    Layers.cam_image,
    Layers.cam_mask,
    Layers.cam_foreground,
    Layers.cam_crop,
    Layers.cam_depth,

    Layers.center_geometry,
    Layers.center_mask,
    Layers.center_depth,
    Layers.center_foreground,
    Layers.center_pose,

    Layers.color_mask,
    Layers.flow,
    Layers.fluid3d,
    Layers.composite,
]

INTERFACE_LAYERS: list[Layers] = [
    Layers.tracker,
    Layers.pose,
]

LARGE_LAYERS: list[Layers] = [
    Layers.flow,
    Layers.fluid3d,
    Layers.composite,
]

OUTPUT_WIDTH: int = 1080
OUTPUT_HEIGHT: int = 1920

# The layers that are not output-shaped, each with the layer whose texture holds its shape
SHAPE_LAYERS: dict[Layers, Layers] = {
    Layers.cam_image:      Layers.cam_image,
    Layers.cam_mask:       Layers.cam_mask,
    Layers.cam_foreground: Layers.cam_foreground,
    Layers.cam_crop:       Layers.cam_crop,
    Layers.cam_depth:      Layers.cam_depth,
    Layers.tracker:        Layers.cam_image,  # camera stretched into a panel-sized FBO
    Layers.pose:           Layers.cam_crop,   # crop stretched into a panel-sized FBO
}


class DeepFlowRender(RenderBase):
    def __init__(self, board: RenderBoard, settings: RenderSettings, num_cams: int = 1, max_players: int = 1) -> None:
        super().__init__(settings.window)
        self.max_players: int = max_players
        self.num_cams: int = num_cams
        self.settings: RenderSettings = settings

        self.board: RenderBoard = board

        self._update_layers: list[Layers] = UPDATE_LAYERS
        self._interface_layers: list[Layers] = INTERFACE_LAYERS
        self._left_layers: list[Layers] = settings.layer.select.left
        self._right_layers: list[Layers] = settings.layer.select.right
        self._draw_layers: list[Layers] = settings.layer.select.final

        self.L: dict[Layers, dict[int, ls.LayerBase]] = {layer: {} for layer in Layers}

        # Set data.b overrides
        settings.data.b.line_width = 6.0
        settings.data.b.line_smooth = 6.0
        settings.data.b.use_history_color = True

        flows: dict[int, ls.FlowLayer] = {}
        cmt: dict[int, Texture] = {}
        for i in range(self.num_cams):
            cam_image =     self.L[Layers.cam_image][i] =   ls.ImageSourceLayer(    i, self.board)
            cam_mask =      self.L[Layers.cam_mask][i] =    ls.MaskSourceLayer(     i, self.board)
            cam_foreground = self.L[Layers.cam_foreground][i] = ls.ForegroundSourceLayer(i, self.board)
            cam_crop =      self.L[Layers.cam_crop][i] =    ls.CropSourceLayer(     i, self.board)
            cam_depth =     self.L[Layers.cam_depth][i] =   ls.DepthSourceLayer(    i, self.board,   cam_image.texture,          settings.depth)

            tracker_compositor = self.L[Layers.tracker][i] = ls.TrackerCompositor(  i, self.board,   cam_image.texture,          settings.preview.tracker,   settings.colors)
            pose_compositor = self.L[Layers.pose][i] =      ls.PoseCompositor(      i, self.board,   cam_image.texture,          settings.preview.pose,      settings.colors)

            center_geometry = self.L[Layers.center_geometry][i] = ls.CenterGeometry( i, self.board,  cam_image.texture,           settings.center.geometry)
            center_mask =   self.L[Layers.center_mask][i] = ls.CenterMaskLayer(     i, center_geometry, cam_mask.texture,           settings.center.mask)
            cmt[i] = center_mask.texture
            center_cam =    self.L[Layers.center_cam][i] =  ls.CenterCamLayer(      i, center_geometry, cam_image.texture,  cmt[i], settings.center.cam)
            self.L[Layers.center_depth][i] =                ls.CenterCamLayer(      i, center_geometry, cam_depth.texture,  cmt[i], settings.center.depth)
            center_foreground = self.L[Layers.center_foreground][i] = ls.CenterForegroundLayer(i, center_geometry, cam_foreground.texture, cmt[i], settings.center.foreground, settings.colors)
            center_pose =   self.L[Layers.center_pose][i] = ls.CenterPoseLayer(     i, center_geometry,                             settings.center.pose,       settings.colors)

            ms_mask =       self.L[Layers.color_mask][i] =  ls.MSColorMaskLayer(    i, self.board,   center_foreground.texture, cmt, settings.center.color,     settings.colors)
            flows[i] =      self.L[Layers.flow][i] =        ls.FlowLayer(           i, self.board,   cam_mask,  center_mask.texture, center_foreground.texture, settings.flow)
            fluid3d =       self.L[Layers.fluid3d][i] =     ls.Fluid3DLayer(        i, self.board,   flows,                      settings.fluid3d,           settings.colors)

            composite =     self.L[Layers.composite][i] =   ls.CompositeLayer(                          [fluid3d, ms_mask],         settings.layer.composite)

            self.L[Layers.data_A_W][i]  = ls.FeatureWindowLayer(i, self.board, settings.data.a, settings.colors)
            self.L[Layers.data_A_F][i]  = ls.FeatureFrameLayer( i, self.board, settings.data.a, settings.colors)
            self.L[Layers.data_B_W][i]  = ls.FeatureWindowLayer(i, self.board, settings.data.b, settings.colors)
            self.L[Layers.data_B_F][i]  = ls.FeatureFrameLayer( i, self.board, settings.data.b, settings.colors)
            self.L[Layers.data_time][i] = ls.MotionTimeRenderer(i, self.board)

        # Split view: the left and right panels side by side or stacked, whichever shows them larger
        self.subdivision: Subdivision = self._make_layout(settings.window.width, settings.window.height)

        # The render rate of each tick feeds the fluid's time step
        self._last_tick: float = 0.0
        self.add_update_callback(self._measure_fps)

        # A reset asked for from another thread, applied at the next update on the GL thread
        self._reset_fluid: Event = Event()

        # hot reloader
        self.hot_reloader = HotReloadMethods(self.__class__, True, True)

    @staticmethod
    def _make_layout(width: int, height: int) -> Subdivision:
        aspect: float = OUTPUT_WIDTH / OUTPUT_HEIGHT
        columns, rows = best_grid(2, aspect, width, height)
        panels = SubdivisionRow(name='panels', columns=columns, rows=rows, src_aspect_ratio=aspect, padding=Point2f(1.0, 1.0))
        return make_subdivision([panels], width, height, True)

    def on_main_window_resize(self, width: int, height: int) -> None:
        self.subdivision = self._make_layout(width, height)
        self.allocate_window_renders()

    def allocate(self) -> None:
        for layer_type, cam_dict in self.L.items():
            for layer in cam_dict.values():
                if layer_type in LARGE_LAYERS:
                    layer.allocate(OUTPUT_WIDTH * 2, OUTPUT_HEIGHT * 2, GL_RGBA16F)
                else:
                    layer.allocate(OUTPUT_WIDTH, OUTPUT_HEIGHT, GL_RGBA16F)
        self.allocate_window_renders()
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

    def _measure_fps(self, now: float) -> None:
        if self._last_tick > 0.0 and now > self._last_tick:
            self.settings.fluid3d.fluid_flow.fps = min(240.0, 1.0 / (now - self._last_tick))
        self._last_tick = now

    def reset_fluid(self) -> None:
        """Clear the optical flow, the bridges and the fluid volume at the next update; any thread."""
        self._reset_fluid.set()

    def update(self) -> None:
        self._notify_update()

        if self._reset_fluid.is_set():
            self._reset_fluid.clear()
            for layer_type in (Layers.flow, Layers.fluid3d):
                for layer in self.L[layer_type].values():
                    cast(ls.FlowLayer | ls.Fluid3DLayer, layer).reset()

        self._left_layers = self.settings.layer.select.left
        self._right_layers = self.settings.layer.select.right
        self._draw_layers = self.settings.layer.select.final

        flow_key: Layers = self.settings.flow.inputs.flow
        density_key: Layers = self.settings.flow.inputs.density
        for i, _flow_layer in self.L[Layers.flow].items():
            flow_layer = cast(ls.FlowLayer, _flow_layer)
            if flow_key in self.L and i in self.L[flow_key]:
                flow_layer.set_flow_input(self.L[flow_key][i].texture)
            if density_key in self.L and i in self.L[density_key]:
                flow_layer.set_density_input(self.L[density_key][i].texture)

        composite_keys: list[Layers] = self.settings.layer.select.composite
        for i, _composite_layer in self.L[Layers.composite].items():
            composite_layer = cast(ls.CompositeLayer, _composite_layer)
            composite_layer.set_layers([
                self.L[key][i] for key in composite_keys
                if key in self.L and i in self.L[key]
            ])

        Style.reset_state()
        Style.set_blend_mode(Style.BlendMode.ALPHA)
        seen: set[Layers] = set()
        for layer_type in self._update_layers + self._interface_layers + self._left_layers + self._right_layers + self._draw_layers:
            if layer_type not in seen:
                seen.add(layer_type)
                for layer in self.L[layer_type].values():
                    layer.update()

    def draw_main(self, width: int, height: int) -> None:
        clear_color()

        Style.reset_state()
        Style.set_blend_mode(Style.BlendMode.ALPHA)

        # Left panel
        left_rect: Rect = self.subdivision.get_rect('panels', 0)
        self._draw_fit(self._left_layers, left_rect.x, height - left_rect.y - left_rect.height, left_rect.width, left_rect.height)

        # Right panel
        right_rect: Rect = self.subdivision.get_rect('panels', 1)
        self._draw_fit(self._right_layers, right_rect.x, height - right_rect.y - right_rect.height, right_rect.width, right_rect.height)

    def draw_secondary(self, monitor_id: int, width: int, height: int) -> None:
        clear_color()

        Style.reset_state()
        Style.push_style()
        Style.set_blend_mode(Style.BlendMode.ADD)

        self._draw_fit(self._draw_layers, 0, 0, width, height)

        Style.pop_style()

    def _draw_fit(self, layer_types: list[Layers], x: float, y: float, width: float, height: float) -> None:
        """Draw each layer into a GL rect (origin bottom left) at its own shape, centered."""
        if width <= 0 or height <= 0:
            return
        for layer_type in layer_types:
            if 0 in self.L[layer_type]:
                fit_x, fit_y, fit_width, fit_height = fit(*self._layer_size(layer_type), width, height)
                glViewport(int(x + fit_x), int(y + fit_y), int(fit_width), int(fit_height))
                self.L[layer_type][0].draw()

    def _layer_size(self, layer_type: Layers) -> tuple[int, int]:
        if layer_type in SHAPE_LAYERS:
            texture: Texture = self.L[SHAPE_LAYERS[layer_type]][0].texture
            if texture.allocated:
                return texture.width, texture.height
        return OUTPUT_WIDTH, OUTPUT_HEIGHT
