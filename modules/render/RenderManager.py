# TODO
# fix datalayer order / dark
# setvieport more consistent
# fx allocation sizes



# Third-party imports
from OpenGL.GL import GL_RGBA16F, GL_RGBA, glViewport

# Local application imports
from modules.gl import RenderBase, WindowManager, Shader, Style, clear_color, Texture
from modules.gl.WindowManager import MonitorId, WindowSettings
from modules.render.layer_settings import Layers
from modules.render.layers import LayerBase

from modules.data_hub import DataHub
from modules.render.render_settings import RenderSettings
from modules.utils.PointsAndRects import Rect, Point2f

# Render Imports
from modules.render.composition_subdivider import make_subdivision, SubdivisionRow, Subdivision
from modules.render import layers as ls

from modules.utils.HotReloadMethods import HotReloadMethods


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
    Layers.fluid,
    Layers.composite,
]

INTERFACE_LAYERS: list[Layers] = [
    Layers.poser,
]

LARGE_LAYERS: list[Layers] = [
    # Layers.centre_cam,
    # Layers.centre_mask,
    # Layers.ms_mask,
    # Layers.centre_pose,
    Layers.flow,
    Layers.fluid,
    Layers.composite,
]


class RenderManager(RenderBase):
    def __init__(self, data_hub: DataHub, settings: RenderSettings, num_cams: int = 3, num_players: int = 3) -> None:
        self.num_players: int = num_players
        self.num_cams: int =    num_cams
        self.settings: RenderSettings = settings

        # data
        self.data_hub: DataHub = data_hub

        # layers
        self._update_layers: list[Layers] =     UPDATE_LAYERS
        self._interface_layers: list[Layers] =  INTERFACE_LAYERS
        self._preview_layers: list[Layers] =    settings.layer.select.preview
        self._draw_layers: list[Layers] =       settings.layer.select.final

        self.L: dict[Layers, dict[int, LayerBase]] = {layer: {} for layer in Layers}

        # Set data.b overrides (class defaults match data.a)
        settings.data.b.line_width = 6.0
        settings.data.b.line_smooth = 6.0
        settings.data.b.use_history_color = True

        flows: dict[int, ls.FlowLayer] = {}
        cmt: dict[int, Texture] = {}
        for i in range(self.num_cams):
            cam_image =     self.L[Layers.cam_image][i] =   ls.ImageSourceLayer(    i, self.data_hub)
            cam_mask =      self.L[Layers.cam_mask][i] =    ls.MaskSourceLayer(     i, self.data_hub)
            cam_frg =       self.L[Layers.cam_frg][i]=      ls.FrgSourceLayer(      i, self.data_hub)
            cam_crop =      self.L[Layers.cam_crop][i] =    ls.CropSourceLayer(     i, self.data_hub)

            cam_comp =      self.L[Layers.poser][i] =       ls.TrackerCompositor(   i, self.data_hub,   cam_image.texture,          settings.preview.tracker,   settings.colors)
            track_comp =    self.L[Layers.tracker][i] =     ls.PoseCompositor(      i, self.data_hub,   cam_image.texture,          settings.preview.poser,     settings.colors)

            centre_gmtry=   self.L[Layers.centre_geom][i] = ls.CentreGeometry(      i, self.data_hub,                               settings.centre.geometry)
            centre_mask =   self.L[Layers.centre_mask][i] = ls.CentreMaskLayer(     i, centre_gmtry,    cam_mask.texture,           settings.centre.mask)
            cmt[i] = centre_mask.texture
            centre_cam =    self.L[Layers.centre_cam][i] =  ls.CentreCamLayer(      i, centre_gmtry,    cam_image.texture,  cmt[i], settings.centre.cam)
            centre_frg =    self.L[Layers.centre_frg][i] =  ls.CentreFrgLayer(      i, centre_gmtry,    cam_frg.texture,    cmt[i], settings.centre.frg,        settings.colors)
            centre_pose =   self.L[Layers.centre_pose][i] = ls.CentrePoseLayer(     i, centre_gmtry,                                settings.centre.pose,       settings.colors)

            ms_mask =       self.L[Layers.color_mask][i] =  ls.MSColorMaskLayer(    i, self.data_hub,   centre_frg.texture, cmt,    settings.centre.color,      settings.colors)
            flows[i] =      self.L[Layers.flow][i] =        ls.FlowLayer(           i, self.data_hub,   cam_mask,  centre_mask.texture, centre_frg.texture,     settings.flow)
            fluid =         self.L[Layers.fluid][i] =       ls.FluidLayer(          i, self.data_hub,   flows,                      settings.fluid,             settings.colors)

            lut =           self.L[Layers.composite][i] =   ls.CompositeLayer(                          [fluid, ms_mask],           settings.layer.lut)

            self.L[Layers.data_A_W][i]  = ls.FeatureWindowLayer(i, self.data_hub, settings.data.a, settings.colors)
            self.L[Layers.data_A_F][i]  = ls.FeatureFrameLayer( i, self.data_hub, settings.data.a, settings.colors)
            self.L[Layers.data_B_W][i]  = ls.FeatureWindowLayer(i, self.data_hub, settings.data.b, settings.colors)
            self.L[Layers.data_B_F][i]  = ls.FeatureFrameLayer( i, self.data_hub, settings.data.b, settings.colors)
            self.L[Layers.data_time][i] = ls.MTimeRenderer(     i, self.data_hub)

        # composition
        self.subdivision_rows: list[SubdivisionRow] = [
            SubdivisionRow(name='track',        columns=self.num_cams,    rows=1, src_aspect_ratio=1.0,  padding=Point2f(1.0, 1.0)),
            SubdivisionRow(name='preview',      columns=self.num_players, rows=1, src_aspect_ratio=9/16, padding=Point2f(1.0, 1.0)),
        ]
        self.subdivision: Subdivision = make_subdivision(self.subdivision_rows, settings.window.width, settings.window.height, False)
        self.window_manager: WindowManager = WindowManager(self, settings.window)

        # Propagate window fps to fluid simulation configs
        def _propagate_fps(fps: int) -> None:
            settings.fluid.fluid_flow.fps = fps
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
        w, h = self.subdivision.get_allocation_size('similarity', 0)
        for i in range(self.num_cams):
            w, h = self.subdivision.get_allocation_size('track', i)
            self.L[Layers.poser][i].allocate(w , h, GL_RGBA)
            # w, h = self.subdivision.get_allocation_size('preview', i)
            pass
            # self.L[Layers.feature_buf][i].allocate(w, h, GL_RGBA)

    def deallocate(self) -> None:
        # self.pose_sim_layer.deallocate()
        for cam_dict in self.L.values():
            for layer in cam_dict.values():
                layer.deallocate()

    def update(self) -> None:
        self.data_hub.notify_update()

        self._update_layers = UPDATE_LAYERS
        self._draw_layers = self.settings.layer.select.final
        self._preview_layers = self.settings.layer.select.preview

        Style.reset_state()
        Style.set_blend_mode(Style.BlendMode.ALPHA)
        seen: set[Layers] = set()
        for layer_type in self._update_layers + self._interface_layers + self._draw_layers + self._preview_layers:
            if layer_type not in seen:
                seen.add(layer_type)
                for layer in self.L[layer_type].values():
                    layer.update()

    def draw_main(self, width: int, height: int) -> None:
        clear_color()

        Style.reset_state()
        Style.set_blend_mode(Style.BlendMode.ALPHA)

        # Interface layers
        for i in range(self.num_cams):
            track_rect: Rect = self.subdivision.get_rect('track', i)
            glViewport(int(track_rect.x), int(height - track_rect.y - track_rect.height), int(track_rect.width), int(track_rect.height))
            for layer_type in self._interface_layers:
                self.L[layer_type][i].draw()

        # Preview layers
        for i in range(self.num_cams):
            preview_rect: Rect = self.subdivision.get_rect('preview', i)
            glViewport(int(preview_rect.x), int(height - preview_rect.y - preview_rect.height), int(preview_rect.width), int(preview_rect.height))
            for layer_type in self._preview_layers:
                self.L[layer_type][i].draw()


    def draw_secondary(self, monitor_id: int, width: int, height: int) -> None:
        clear_color()

        try:
            camera_id: int = self.settings.window.secondary_list.index(MonitorId(monitor_id))
        except ValueError:
            return

        if camera_id >= self.num_cams:
            return

        Style.reset_state()
        Style.push_style()
        Style.set_blend_mode(Style.BlendMode.ADD)

        for layer_type in self._draw_layers:
            if camera_id in self.L[layer_type]:
                self.L[layer_type][camera_id].draw()

        Style.pop_style()