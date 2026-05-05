"""Deep Flow render — split-view layer graph with 3D volumetric fluid."""

from OpenGL.GL import GL_RGBA16F, GL_RGBA, glViewport

from modules.gl import RenderBase, Shader, Style, clear_color, Texture, MonitorId, WindowSettings
from modules.render.layers import LayerBase
from modules.render import layers as ls, make_subdivision, SubdivisionRow, Subdivision
from modules.utils import Rect, Point2f, HotReloadMethods

from .render_board import RenderBoard
from .settings import Layers, RenderSettings


UPDATE_LAYERS: list[Layers] = [
    Layers.cam_image,
    Layers.cam_mask,
    Layers.cam_frg,
    Layers.cam_crop,

    Layers.centre_geom,
    Layers.centre_mask,
    Layers.centre_frg,
    Layers.centre_pose,

    Layers.color_mask,
    Layers.flow,
    Layers.fluid3d,
    Layers.composite,
]

INTERFACE_LAYERS: list[Layers] = [
    Layers.poser,
    Layers.tracker,
]

LARGE_LAYERS: list[Layers] = [
    Layers.flow,
    Layers.fluid3d,
    Layers.composite,
]


class DeepFlowRender(RenderBase):
    def __init__(self, board: RenderBoard, settings: RenderSettings, num_cams: int = 1, num_players: int = 1) -> None:
        super().__init__(settings.window)
        self.num_players: int = num_players
        self.num_cams: int = num_cams
        self.settings: RenderSettings = settings

        self.board: RenderBoard = board

        self._update_layers: list[Layers] = UPDATE_LAYERS
        self._interface_layers: list[Layers] = INTERFACE_LAYERS
        self._left_layers: list[Layers] = settings.layer.select.left
        self._right_layers: list[Layers] = settings.layer.select.right
        self._draw_layers: list[Layers] = settings.layer.select.final

        self.L: dict[Layers, dict[int, LayerBase]] = {layer: {} for layer in Layers}

        # Set data.b overrides
        settings.data.b.line_width = 6.0
        settings.data.b.line_smooth = 6.0
        settings.data.b.use_history_color = True

        flows: dict[int, ls.FlowLayer] = {}
        cmt: dict[int, Texture] = {}
        for i in range(self.num_cams):
            cam_image =     self.L[Layers.cam_image][i] =   ls.ImageSourceLayer(    i, self.board)
            cam_mask =      self.L[Layers.cam_mask][i] =    ls.MaskSourceLayer(     i, self.board)
            cam_frg =       self.L[Layers.cam_frg][i] =     ls.FrgSourceLayer(      i, self.board)
            cam_crop =      self.L[Layers.cam_crop][i] =    ls.CropSourceLayer(     i, self.board)

            cam_comp =      self.L[Layers.poser][i] =       ls.TrackerCompositor(   i, self.board,   cam_image.texture,          settings.preview.tracker,   settings.colors)
            track_comp =    self.L[Layers.tracker][i] =     ls.PoseCompositor(      i, self.board,   cam_image.texture,          settings.preview.poser,     settings.colors)

            centre_gmtry =  self.L[Layers.centre_geom][i] = ls.CentreGeometry(      i, self.board,                               settings.centre.geometry)
            centre_mask =   self.L[Layers.centre_mask][i] = ls.CentreMaskLayer(     i, centre_gmtry,    cam_mask.texture,           settings.centre.mask)
            cmt[i] = centre_mask.texture
            centre_cam =    self.L[Layers.centre_cam][i] =  ls.CentreCamLayer(      i, centre_gmtry,    cam_image.texture,  cmt[i], settings.centre.cam)
            centre_frg =    self.L[Layers.centre_frg][i] =  ls.CentreFrgLayer(      i, centre_gmtry,    cam_frg.texture,    cmt[i], settings.centre.frg,        settings.colors)
            centre_pose =   self.L[Layers.centre_pose][i] = ls.CentrePoseLayer(     i, centre_gmtry,                                settings.centre.pose,       settings.colors)

            ms_mask =       self.L[Layers.color_mask][i] =  ls.MSColorMaskLayer(    i, self.board,   centre_frg.texture, cmt,    settings.centre.color,      settings.colors)
            flows[i] =      self.L[Layers.flow][i] =        ls.FlowLayer(           i, self.board,   cam_mask,  centre_mask.texture, centre_frg.texture,     settings.flow)
            fluid3d =       self.L[Layers.fluid3d][i] =     ls.Fluid3DLayer(        i, self.board,   flows,                      settings.fluid3d,           settings.colors)

            lut =           self.L[Layers.composite][i] =   ls.CompositeLayer(                          [fluid3d, ms_mask],         settings.layer.composite)

            self.L[Layers.data_A_W][i]  = ls.FeatureWindowLayer(i, self.board, settings.data.a, settings.colors)
            self.L[Layers.data_A_F][i]  = ls.FeatureFrameLayer( i, self.board, settings.data.a, settings.colors)
            self.L[Layers.data_B_W][i]  = ls.FeatureWindowLayer(i, self.board, settings.data.b, settings.colors)
            self.L[Layers.data_B_F][i]  = ls.FeatureFrameLayer( i, self.board, settings.data.b, settings.colors)
            self.L[Layers.data_time][i] = ls.MTimeRenderer(     i, self.board)

        # Split-view composition: left panel + right panel
        self.subdivision_rows: list[SubdivisionRow] = [
            SubdivisionRow(name='left',  columns=1, rows=1, src_aspect_ratio=1.0, padding=Point2f(1.0, 1.0)),
            SubdivisionRow(name='right', columns=1, rows=1, src_aspect_ratio=1.0, padding=Point2f(1.0, 1.0)),
        ]
        self.subdivision: Subdivision = make_subdivision(self.subdivision_rows, settings.window.width, settings.window.height, False)

        # Propagate window fps to fluid simulation config
        def _propagate_fps(fps: int) -> None:
            settings.fluid3d.fluid_flow.fps = fps
        settings.window.bind(WindowSettings.avg_fps, _propagate_fps)

        # hot reloader
        self.hot_reloader = HotReloadMethods(self.__class__, True, True)

    def on_main_window_resize(self, width: int, height: int) -> None:
        self.subdivision = make_subdivision(self.subdivision_rows, width, height, True)
        self.allocate_window_renders()

    def allocate(self) -> None:
        for layer_type, cam_dict in self.L.items():
            for layer in cam_dict.values():
                if layer_type in LARGE_LAYERS:
                    layer.allocate(1080 * 2, 1920 * 2, GL_RGBA16F)
                else:
                    layer.allocate(1080, 1920, GL_RGBA16F)
        self.allocate_window_renders()
        Shader.enable_hot_reload()

    def allocate_window_renders(self) -> None:
        # Allocate interface layers to split-view panel sizes
        for layer_type in INTERFACE_LAYERS:
            for i in range(self.num_cams):
                w, h = self.subdivision.get_allocation_size('right', 0)
                self.L[layer_type][i].allocate(w, h, GL_RGBA)

    def deallocate(self) -> None:
        for cam_dict in self.L.values():
            for layer in cam_dict.values():
                layer.deallocate()

    def update(self) -> None:
        self._notify_update()

        self._left_layers = self.settings.layer.select.left
        self._right_layers = self.settings.layer.select.right
        self._draw_layers = self.settings.layer.select.final

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
        left_rect: Rect = self.subdivision.get_rect('left', 0)
        glViewport(int(left_rect.x), int(height - left_rect.y - left_rect.height), int(left_rect.width), int(left_rect.height))
        for layer_type in self._left_layers:
            if 0 in self.L[layer_type]:
                self.L[layer_type][0].draw()

        # Right panel
        right_rect: Rect = self.subdivision.get_rect('right', 0)
        glViewport(int(right_rect.x), int(height - right_rect.y - right_rect.height), int(right_rect.width), int(right_rect.height))
        for layer_type in self._right_layers:
            if 0 in self.L[layer_type]:
                self.L[layer_type][0].draw()

    def draw_secondary(self, monitor_id: int, width: int, height: int) -> None:
        clear_color()

        Style.reset_state()
        Style.push_style()
        Style.set_blend_mode(Style.BlendMode.ADD)

        for layer_type in self._draw_layers:
            if 0 in self.L[layer_type]:
                self.L[layer_type][0].draw()

        Style.pop_style()
