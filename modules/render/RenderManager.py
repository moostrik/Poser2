# Standard library imports
from enum import IntEnum, auto

# Third-party imports
from OpenGL.GL import * # type: ignore

# Local application imports
from modules.gl import RenderBase, WindowManager, Shader, Style, clear_color, Texture
from modules.render.color_settings import ColorSettings
from modules.render.layers import LayerBase

from modules.DataHub import DataHub
from modules.gui.PyReallySimpleGui import Gui
from modules.render.render_settings import RenderSettings
from modules.utils.PointsAndRects import Rect, Point2f

# Render Imports
from modules.render.CompositionSubdivider import make_subdivision, SubdivisionRow, Subdivision
from modules.render import layers as ls

from modules.utils.HotReloadMethods import HotReloadMethods


class Layers(IntEnum):
    # source layers
    cam_image =     0
    cam_mask =      auto()
    cam_frg=        auto()
    cam_crop =      auto()

    # composite layers
    tracker =       auto()
    poser =         auto()

    # centre layers
    centre_math=    auto()
    centre_cam =    auto()
    centre_mask =   auto()
    centre_frg =    auto()
    centre_pose =   auto()

    # Data layers (configurable slots A and B, all pre-allocated)
    data_B_W =      auto()
    data_B_F =      auto()
    data_A_W =      auto()
    data_A_F =      auto()
    data_time =     auto()

    # composition layers
    motion =        auto()
    flow =          auto()
    fluid =         auto()
    ms_mask =       auto()
    composite =     auto()

    # hdt_prep =      auto()
    # hdt_blend =     auto()


UPDATE_LAYERS: list[Layers] = [
    Layers.cam_image,
    Layers.cam_mask,
    Layers.cam_frg,
    Layers.cam_crop,

    Layers.centre_math,
    # Layers.centre_cam,
    Layers.centre_mask,
    Layers.centre_frg,
    Layers.centre_pose,

    # Layers.motion,
    Layers.ms_mask,
    Layers.flow,
    Layers.fluid,
    Layers.composite,

    # Layers.hdt_prep,
    # Layers.hdt_blend,
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

PREVIEW_CENTRE: list[Layers] = [
    Layers.centre_frg,
    Layers.centre_pose,
    Layers.data_time,
]

SHOW_CAM: list[Layers] = [
    Layers.poser
]

SHOW_POSE: list[Layers] = [
    Layers.tracker,
]

SHOW_CENTRE: list[Layers] = [
    # Layers.centre_cam,
    # Layers.centre_frg,
    Layers.centre_mask,
    # Layers.hdt_prep,
    Layers.centre_pose,
]

SHOW_MASK: list[Layers] = [
    Layers.cam_mask,
    # Layers.centre_mask,
    # Layers.centre_motion,
    Layers.centre_pose,
    # Layers.cam_crop
]

SHOW_COMP: list[Layers] = [
    # Layers.centre_frg,
    # Layers.flow,
    # Layers.centre_mask,
    # Layers.motion,
    # Layers.fluid,
    # Layers.ms_mask,
    # Layers.sim_blend,
    # Layers.centre_pose,
    # Layers.centre_motion,
    # Layers.centre_frg,
    # Layers.fluid,
    Layers.composite,
]

SHOW_DATA: list[Layers] = [
    # Layers.centre_motion,
    Layers.data_B_W,
    Layers.data_B_F,
    Layers.data_A_W,
    Layers.data_A_F,
    # Layers.data_time,
]


PREVIEW_LAYERS: list[Layers] = SHOW_COMP + SHOW_DATA
FINAL_LAYERS: list[Layers] = SHOW_COMP + SHOW_DATA

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
        self._preview_layers: list[Layers] =    PREVIEW_LAYERS
        self._draw_layers: list[Layers] =       FINAL_LAYERS

        self.L: dict[Layers, dict[int, LayerBase]] = {layer: {} for layer in Layers}

        # Set data_b overrides (class defaults match data_a)
        settings.data_b.line_width = 6.0
        settings.data_b.line_smooth = 6.0
        settings.data_b.colors = [settings.colors.history.to_tuple()]
        # Wire color_settings reference to data layer configs
        settings.data_a.color_settings = settings.colors
        settings.data_b.color_settings = settings.colors

        flows: dict[int, ls.FlowLayer] = {}
        cmt: dict[int, Texture] = {}
        for i in range(self.num_cams):
            cam_image =     self.L[Layers.cam_image][i] =   ls.ImageSourceLayer(    i, self.data_hub)
            cam_mask =      self.L[Layers.cam_mask][i] =    ls.MaskSourceLayer(     i, self.data_hub)
            cam_frg =       self.L[Layers.cam_frg][i]=      ls.FrgSourceLayer(      i, self.data_hub)
            cam_crop =      self.L[Layers.cam_crop][i] =    ls.CropSourceLayer(     i, self.data_hub)

            cam_comp =      self.L[Layers.poser][i] =       ls.TrackerCompositor(   i, self.data_hub,   cam_image.texture,          settings.tracker,       settings.colors)
            track_comp =    self.L[Layers.tracker][i] =     ls.PoseCompositor(      i, self.data_hub,   cam_image.texture,          settings.pose_comp,     settings.colors)

            centre_gmtry=   self.L[Layers.centre_math][i] = ls.CentreGeometry(      i, self.data_hub,                               settings.centre_geometry)
            centre_mask =   self.L[Layers.centre_mask][i] = ls.CentreMaskLayer(     i, centre_gmtry,    cam_mask.texture,           settings.centre_mask)
            cmt[i] = centre_mask.texture
            centre_cam =    self.L[Layers.centre_cam][i] =  ls.CentreCamLayer(      i, centre_gmtry,    cam_image.texture,  cmt[i], settings.centre_cam)
            centre_frg =    self.L[Layers.centre_frg][i] =  ls.CentreFrgLayer(      i, centre_gmtry,    cam_frg.texture,    cmt[i], settings.centre_frg,    settings.colors)
            centre_pose =   self.L[Layers.centre_pose][i] = ls.CentrePoseLayer(     i, centre_gmtry,                                settings.centre_pose,   settings.colors)

            motion =        self.L[Layers.motion][i] =      ls.MotionLayer(         i, self.data_hub,                       cmt[i],                         settings.colors)
            ms_mask =       self.L[Layers.ms_mask][i] =     ls.MSColorMaskLayer(    i, self.data_hub,   centre_frg.texture, cmt,    settings.ms_mask,       settings.colors)
            flows[i] =      self.L[Layers.flow][i] =        ls.FlowLayer(           i, self.data_hub,   cam_mask,           cmt[i], settings.flow)
            fluid =         self.L[Layers.fluid][i] =       ls.FluidLayer(          i, self.data_hub,   flows,                      settings.fluid,         settings.colors)

            lut =           self.L[Layers.composite][i] =   ls.CompositeLayer(                          [fluid, ms_mask],           settings.composite)

            self.L[Layers.data_A_W][i]  = ls.FeatureWindowLayer(i, self.data_hub, settings.data_a)
            self.L[Layers.data_A_F][i]  = ls.FeatureFrameLayer( i, self.data_hub, settings.data_a)
            self.L[Layers.data_B_W][i]  = ls.FeatureWindowLayer(i, self.data_hub, settings.data_b)
            self.L[Layers.data_B_F][i]  = ls.FeatureFrameLayer( i, self.data_hub, settings.data_b)
            self.L[Layers.data_time][i] = ls.MTimeRenderer(     i, self.data_hub)

        # composition
        self.subdivision_rows: list[SubdivisionRow] = [
            SubdivisionRow(name='track',        columns=self.num_cams,    rows=1, src_aspect_ratio=1.0,  padding=Point2f(1.0, 1.0)),
            SubdivisionRow(name='preview',      columns=self.num_players, rows=1, src_aspect_ratio=9/16, padding=Point2f(1.0, 1.0)),
        ]
        self.subdivision: Subdivision = make_subdivision(self.subdivision_rows, settings.window.width, settings.window.height, False)
        self.window_manager: WindowManager = WindowManager(self, settings.window)

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
        # self.pose_sim_layer.allocate(w, h, GL_RGBA)
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
        self._draw_layers = FINAL_LAYERS
        self._preview_layers = PREVIEW_LAYERS

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

        Style.reset_state()
        Style.push_style()
        Style.set_blend_mode(Style.BlendMode.ADD)

        camera_id: int = self.settings.window.secondary_list.index(monitor_id)
        for layer_type in self._draw_layers:
            self.L[layer_type][camera_id].draw()

        Style.pop_style()