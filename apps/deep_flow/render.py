"""Deep Flow render — split-view layer graph with 3D volumetric fluid."""

import math
from threading import Event
from typing import cast

from OpenGL.GL import GL_RGBA16F, GL_RGBA, glViewport

from modules.gl import RenderBase, Shader, Style, clear_color, fit, Texture, MonitorId
from modules.render import layers as ls, make_subdivision, best_grid, grid_cell_aspect_ratio, SubdivisionRow, Subdivision
from modules.utils import Rect, Point2f, HotReloadMethods

from modules.board import HasFrames

from .capture_poses import CapturePoses
from .render_board import RenderBoard
from .settings import Layers, MaskSource, RenderSettings


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

# The source layer that says when a center layer has a new frame; the foreground's is the mask source's (`flow_sources`)
FLOW_SOURCES: dict[Layers, Layers] = {
    Layers.center_cam:        Layers.cam_image,
    Layers.center_capture:    Layers.capture_image,
    Layers.center_depth:      Layers.cam_depth,
    Layers.center_mask:       Layers.cam_mask,
}


def flow_sources(mask_source: MaskSource) -> dict[Layers, Layers]:
    """FLOW_SOURCES with the foreground's source: RVM's foreground layer, or the capture's image."""
    foreground: Layers = Layers.cam_foreground if mask_source == MaskSource.RVM else Layers.capture_image
    return {**FLOW_SOURCES, Layers.center_foreground: foreground}

OUTPUT_WIDTH: int = 1080
OUTPUT_HEIGHT: int = 1920

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
        self._flow_sources: dict[Layers, Layers] = flow_sources(mask_source)
        self._image_span: float = 2.0 * math.tan(math.radians(fov) / 2.0)     # metres the camera image spans per metre of depth

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

        flows: dict[int, ls.FlowLayer] = {}
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
            foreground_source: Texture
            if depth_source:
                cam_mask = self.L[Layers.cam_mask][i] =     ls.DepthMaskLayer(      cam_depth)
                foreground_source = capture_image.texture   # masked in the center view (`premultiply`)
            else:
                cam_mask = self.L[Layers.cam_mask][i] =     ls.MaskSourceLayer(     i, self.board)
                cam_foreground = self.L[Layers.cam_foreground][i] = ls.ForegroundSourceLayer(i, self.board)
                foreground_source = cam_foreground.texture  # RVM's, masked already

            tracker_compositor = self.L[Layers.tracker][i] = ls.TrackerCompositor(  i, self.board,   cam_image.texture,          settings.preview.tracker,   settings.colors)
            pose_compositor = self.L[Layers.pose][i] =      ls.PoseCompositor(      i, self.board,   cam_image.texture, cam_crop, settings.preview.pose,      settings.colors)

            # placed by the pose of the capture it shows, so view, image and depth are one moment
            center_geometry = self.L[Layers.center_geometry][i] = ls.CenterGeometry( i, cast(HasFrames, CapturePoses(self.board)), cam_image.texture, settings.center.geometry)
            center_mask =   self.L[Layers.center_mask][i] = ls.CenterMaskLayer(     i, center_geometry, cam_mask.texture,       space, settings.center.mask)
            cmt[i] = center_mask.texture
            center_cam =    self.L[Layers.center_cam][i] =  ls.CenterCamLayer(      i, center_geometry, cam_image.texture,  cmt[i], settings.center.cam)
            center_capture = self.L[Layers.center_capture][i] = ls.CenterCamLayer(  i, center_geometry, capture_image.texture, cmt[i], settings.center.capture)
            self.L[Layers.center_depth][i] =                ls.CenterDepthLayer(    i, center_geometry, cam_depth, center_capture, cmt[i], settings.center.depth, settings.depth)
            center_foreground = self.L[Layers.center_foreground][i] = ls.CenterForegroundLayer(i, center_geometry, foreground_source, cmt[i], space, depth_source,
                                                                                                settings.center.foreground, settings.colors)
            center_pose =   self.L[Layers.center_pose][i] = ls.CenterPoseLayer(     i, center_geometry,                             settings.center.pose,       settings.colors)

            ms_mask =       self.L[Layers.color_mask][i] =  ls.MSColorMaskLayer(    i, self.board,   center_foreground.texture, cmt, settings.center.color, settings.colors)
            flows[i] =      self.L[Layers.flow][i] =        ls.FlowLayer(           i, self.board,   cam_mask,  center_mask.texture, center_foreground.texture, settings.flow)
            fluid3d =       self.L[Layers.fluid3d][i] =     ls.Fluid3DLayer(        i, self.board,   flows,                      settings.fluid3d,           settings.colors)

            composite =    self.L[Layers.composite][i] =   ls.CompositeLayer(                          [fluid3d, ms_mask],         settings.layer.composite)

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

        # The render rate of each tick feeds the fluid's time step
        self._last_tick: float = 0.0
        self.add_update_callback(self._measure_fps)

        # A reset asked for from another thread, applied at the next update on the GL thread
        self._reset_fluid: Event = Event()

        # hot reloader
        self.hot_reloader = HotReloadMethods(self.__class__, True, True)

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
            for layer_type in [Layers.flow, Layers.fluid3d]:
                for layer in self.L[layer_type].values():
                    cast(ls.FlowLayer | ls.Fluid3DLayer, layer).reset()

        select = self.settings.layer.select
        self._left_layers = select.left
        self._right_layers = select.right
        self._output_layers = select.output
        self._update_layout()

        flow_key: Layers = self.settings.flow.inputs.flow
        density_key: Layers = self.settings.flow.inputs.density
        mask_key: Layers = self.settings.flow.inputs.mask
        flow_source_key: Layers = self._flow_sources.get(flow_key, flow_key)
        for i, _flow_layer in self.L[Layers.flow].items():
            flow_layer = cast(ls.FlowLayer, _flow_layer)
            flow_layer.set_time_step(1.0 / self.settings.fluid3d.fluid_flow.step_fps)
            flow_layer.set_depth_input(cast(ls.CenterDepthLayer, self.L[Layers.center_depth][i]).depth)
            crop: Rect = cast(ls.CenterGeometry, self.L[Layers.center_geometry][i]).image_geometry.crop_roi
            flow_layer.set_view_span(crop.width * self._image_span)
            # one of the depth's own samples in the view; the depth is coarser than the view, stretched onto it
            depth: Texture = cast(ls.DepthSourceLayer, self.L[Layers.cam_depth][i]).depth
            if depth.allocated and crop.width > 0.0 and crop.height > 0.0:
                flow_layer.set_depth_step((1.0 / (depth.width * crop.width), 1.0 / (depth.height * crop.height)))
            if flow_key in self.L and i in self.L[flow_key] and i in self.L[flow_source_key]:
                flow_layer.set_flow_input(self.L[flow_key][i].texture)
                flow_layer.set_flow_source(self.L[flow_source_key][i])
            if density_key in self.L and i in self.L[density_key]:
                flow_layer.set_density_input(self.L[density_key][i].texture)
            flow_layer.set_mask_input(self.L[mask_key][i].texture if mask_key in self.L and i in self.L[mask_key] else None)

        for i, _composite_layer in self.L[Layers.composite].items():
            composite_layer = cast(ls.CompositeLayer, _composite_layer)
            composite_layer.set_layers([
                self.L[key][i] for key in self._output_layers
                if key is not Layers.composite and key in self.L and i in self.L[key]   # it cannot draw into itself
            ])

        Style.reset_state()
        Style.set_blend_mode(Style.BlendMode.ALPHA)
        seen: set[Layers] = set()
        depth_needed: bool = any(cast(ls.FlowLayer, layer).uses_depth for layer in self.L[Layers.flow].values())
        update_layers: list[Layers] = [layer_type for layer_type in self._update_layers if depth_needed or layer_type != Layers.center_depth]
        for layer_type in update_layers + self._interface_layers + self._left_layers + self._right_layers + self._output_layers:
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

        self._draw_fit([Layers.composite], 0, 0, width, height)

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
        if layer_type in SHAPE_LAYERS and 0 in self.L[SHAPE_LAYERS[layer_type]]:      # cam_foreground exists with RVM only
            texture: Texture = self.L[SHAPE_LAYERS[layer_type]][0].texture
            if texture.allocated:
                return texture.width, texture.height
        return OUTPUT_WIDTH, OUTPUT_HEIGHT
