# Standard library imports
from enum import IntEnum, auto
from typing import cast

# Third-party imports
from OpenGL.GL import * # type: ignore

# Local application imports
from modules.gl import RenderBase, WindowManager, Shader, Style, clear_color, Texture
from modules.render.layers import LayerBase

from modules.DataHub import DataHub, Stage
from modules.gui.PyReallySimpleGui import Gui
from modules.render.Config import Config, DataLayer
from modules.utils.PointsAndRects import Rect, Point2f

# Render Imports
from modules.render.CompositionSubdivider import make_subdivision, SubdivisionRow, Subdivision
from modules.render import layers as ls
from modules.render.layers.data.colors import TRACK_COLORS, HISTORY_COLOR

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
    data_B_AV =     auto()
    data_A_W =      auto()
    data_A_F =      auto()
    data_A_AV =     auto()
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
    Layers.centre_mask,
    Layers.ms_mask,
    Layers.centre_pose,
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
    Layers.centre_frg,
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
    # Layers.cam_frg,
    Layers.composite,
]

SHOW_DATA: list[Layers] = [
    # Layers.centre_motion,
    Layers.data_B_W,
    Layers.data_B_F,
    Layers.data_B_AV,
    Layers.data_A_W,
    Layers.data_A_F,
    Layers.data_A_AV,
    # Layers.data_time,
]


PREVIEW_LAYERS: list[Layers] = PREVIEW_CENTRE
FINAL_LAYERS: list[Layers] = SHOW_COMP + SHOW_DATA

class RenderManager(RenderBase):
    def __init__(self, gui: Gui, data_hub: DataHub, settings: Config) -> None:
        self.num_players: int = settings.num_players
        self.num_cams: int =    settings.num_cams

        # data
        self.data_hub: DataHub = data_hub
        self._settings: Config = settings

        # layers
        self._update_layers: list[Layers] =     UPDATE_LAYERS
        self._interface_layers: list[Layers] =  INTERFACE_LAYERS
        self._preview_layers: list[Layers] =    PREVIEW_LAYERS
        self._draw_layers: list[Layers] =       FINAL_LAYERS

        self.L: dict[Layers, dict[int, LayerBase]] = {layer: {} for layer in Layers}

        # configs
        self.tracker_comp_config =  ls.TrackerCompConfig(   stage=Stage.LERP, pose_line_width=2.0, bbox_line_width=2)
        self.pose_comp_config =     ls.PoseCompConfig(      stage=Stage.LERP, line_width=2.0, line_smooth=0.0, use_gpu_crop=True)

        self.centre_gmtr_config=    ls.CentreGeometryConfig(stage=Stage.LERP, cam_aspect=16/9, target_top_x=0.5, target_top_y=0.33, target_bottom_x=0.5, target_bottom_y=0.6, dst_aspectratio=9/16)
        self.centre_mask_config =   ls.CentreMaskConfig(    blend_factor=0.3, blur_steps=0, blur_radius=1.0, dilation_steps=0)
        self.centre_cam_config =    ls.CentreCamConfig(     blend_factor=0.2, mask_opacity=1.0, use_mask=True)
        self.centre_frg_config =    ls.CentreFrgConfig(     blend_factor=0.2, mask_opacity=1.0, use_mask=True)
        self.centre_pose_config =   ls.CentrePoseConfig(    line_width=3.0, line_smooth=0.0, use_scores=False, draw_anchors=True)

        self.data_A_config =        ls.DataLayerConfig(     stage=Stage.SMOOTH,  line_width=3.0, line_smooth=1.0, use_scores=False, render_labels=True, colors=None)
        self.data_B_config =        ls.DataLayerConfig(     stage=Stage.LERP,    line_width=6.0, line_smooth=6.0, use_scores=False, render_labels=True, colors=[HISTORY_COLOR])
        self.data_time_config =     ls.MTimeRendererConfig( stage=Stage.LERP)
        self.composite_config =     ls.CompositeLayerConfig(lut=settings.lut, lut_strength=settings.lut_strength)

        # Watch settings for LUT changes and propagate to composite config
        settings.watch(lambda v: setattr(self.composite_config, 'lut', v), 'lut')
        settings.watch(lambda v: setattr(self.composite_config, 'lut_strength', v), 'lut_strength')

        flows: dict[int, ls.FlowLayer] = {}
        mask_textures: dict[int, Texture] = {}
        for i in range(self.num_cams):
            color: tuple[float, float, float, float] = TRACK_COLORS[i % len(TRACK_COLORS)]
            cam_image =     self.L[Layers.cam_image][i] =   ls.ImageSourceLayer(    i, self.data_hub)
            cam_mask =      self.L[Layers.cam_mask][i] =    ls.MaskSourceLayer(     i, self.data_hub)
            cam_frg =       self.L[Layers.cam_frg][i]=      ls.FrgSourceLayer(      i, self.data_hub)
            cam_crop =      self.L[Layers.cam_crop][i] =    ls.CropSourceLayer(     i, self.data_hub)

            cam_comp =      self.L[Layers.poser][i] =       ls.TrackerCompositor(   i, self.data_hub,   cam_image.texture,  HISTORY_COLOR, color,  self.tracker_comp_config)
            track_comp =    self.L[Layers.tracker][i] =     ls.PoseCompositor(      i, self.data_hub,   cam_image.texture,  color,                  self.pose_comp_config)

            centre_gmtry=   self.L[Layers.centre_math][i] = ls.CentreGeometry(      i, self.data_hub,                                               self.centre_gmtr_config)
            centre_mask =   self.L[Layers.centre_mask][i] = ls.CentreMaskLayer(        centre_gmtry,    cam_mask.texture,                           self.centre_mask_config)
            mask_textures[i] = centre_mask.texture
            centre_cam =    self.L[Layers.centre_cam][i] =  ls.CentreCamLayer(         centre_gmtry,    cam_image.texture,  centre_mask.texture,    self.centre_cam_config)
            centre_frg =    self.L[Layers.centre_frg][i] =  ls.CentreFrgLayer(         centre_gmtry,    cam_frg.texture,    centre_mask.texture,    self.centre_frg_config)
            centre_pose =   self.L[Layers.centre_pose][i] = ls.CentrePoseLayer(        centre_gmtry,    color,                                      self.centre_pose_config)

            motion =        self.L[Layers.motion][i] =      ls.MotionLayer(         i, self.data_hub,   centre_mask.texture, color)
            ms_mask =       self.L[Layers.ms_mask][i] =     ls.MSColorMaskLayer(    i, self.data_hub, mask_textures, list(TRACK_COLORS))
            flows[i] =      self.L[Layers.flow][i] =        ls.FlowLayer(           i, self.data_hub,   centre_mask.texture, list(TRACK_COLORS))
            fluid =         self.L[Layers.fluid][i] =       ls.FluidLayer(          i, self.data_hub,   flows, list(TRACK_COLORS))

            self.L[Layers.composite][i] = ls.CompositeLayer([ms_mask, fluid], self.composite_config)

            # centre_motion = self.L[Layers.hdt_prep][i]=     ls.HDTPrepare(          i, self.data_hub,   PoseDataHubTypes.pose_I,    centre_mask.texture)
            # sim_blend =     self.L[Layers.hdt_blend][i] =   ls.HDTBlend(            i, self.data_hub,   PoseDataHubTypes.pose_I,    cast(dict[int, ls.HDTPrepare], self.L[Layers.hdt_prep]))

            self.L[Layers.data_A_W][i]  = ls.FeatureWindowLayer(i, self.data_hub, self.data_A_config)
            self.L[Layers.data_A_F][i]  = ls.FeatureFrameLayer( i, self.data_hub, self.data_A_config)
            self.L[Layers.data_A_AV][i] = ls.AngleVelLayer(     i, self.data_hub, self.data_A_config)
            self.L[Layers.data_B_W][i]  = ls.FeatureWindowLayer(i, self.data_hub, self.data_B_config)
            self.L[Layers.data_B_F][i]  = ls.FeatureFrameLayer( i, self.data_hub, self.data_B_config)
            self.L[Layers.data_B_AV][i] = ls.AngleVelLayer(     i, self.data_hub, self.data_B_config)
            self.L[Layers.data_time][i] = ls.MTimeRenderer(     i, self.data_hub, self.data_time_config)

        # Bind data layers (not configs) to settings - propagates active state and shared properties
        settings.bind(
            {i: cast(DataLayer, self.L[Layers.data_A_W][i]) for i in range(self.num_cams)},
            {i: cast(DataLayer, self.L[Layers.data_A_F][i]) for i in range(self.num_cams)},
            {i: cast(DataLayer, self.L[Layers.data_A_AV][i]) for i in range(self.num_cams)},
            {i: cast(DataLayer, self.L[Layers.data_B_W][i]) for i in range(self.num_cams)},
            {i: cast(DataLayer, self.L[Layers.data_B_F][i]) for i in range(self.num_cams)},
            {i: cast(DataLayer, self.L[Layers.data_B_AV][i]) for i in range(self.num_cams)},
        )

        # composition
        self.subdivision_rows: list[SubdivisionRow] = [
            SubdivisionRow(name='track',        columns=self.num_cams,    rows=1, src_aspect_ratio=1.0,  padding=Point2f(1.0, 1.0)),
            SubdivisionRow(name='preview',      columns=self.num_players, rows=1, src_aspect_ratio=9/16, padding=Point2f(1.0, 1.0)),
        ]
        self.subdivision: Subdivision = make_subdivision(self.subdivision_rows, settings.width, settings.height, False)

        # window manager
        self.secondary_order_list: list[int] = settings.secondary_list
        self.window_manager: WindowManager = WindowManager(
            self, self.subdivision.width, self.subdivision.height,
            settings.title, settings.fullscreen,
            settings.v_sync, settings.fps,
            settings.x, settings.y,
            settings.monitor, settings.secondary_list
        )

        # hot reloader
        self.hot_reloader = HotReloadMethods(self.__class__, True, True)

    def on_main_window_resize(self, width: int, height: int) -> None:
        self.subdivision = make_subdivision(self.subdivision_rows, width, height, True)
        self.allocate_window_renders()

    def allocate(self) -> None:
        for layer_type, cam_dict in self.L.items():
            for layer in cam_dict.values():
                if layer_type in LARGE_LAYERS:
                    layer.allocate(1080 * 2, 1920 * 2, GL_RGBA32F)
                else:
                    layer.allocate(1080, 1920, GL_RGBA32F)
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

    def draw_main(self, width: int, height: int) -> None:

        Style.reset_state()
        Style.set_blend_mode(Style.BlendMode.ALPHA)
        self.data_hub.notify_update()
        seen: set[Layers] = set()
        for layer_type in self._update_layers + self._interface_layers + self._draw_layers + self._preview_layers:
            if layer_type not in seen:
                seen.add(layer_type)
                for layer in self.L[layer_type].values():
                    layer.update()

        Style.reset_state()
        Style.set_blend_mode(Style.BlendMode.ALPHA)

        glViewport(0, 0, width, height)
        clear_color()

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

        self._update_layers = UPDATE_LAYERS
        self._draw_layers = FINAL_LAYERS
        self._preview_layers = PREVIEW_LAYERS

        self.centre_mask_config.blend_factor = 0.2


    def draw_secondary(self, monitor_id: int, width: int, height: int) -> None:
        glViewport(0, 0, width, height)
        clear_color()

        Style.reset_state()
        Style.push_style()
        Style.set_blend_mode(Style.BlendMode.ADDITIVE)

        camera_id: int = self.secondary_order_list.index(monitor_id)
        for layer_type in self._draw_layers:
            # print(f"Style for layer type {layer_type}: {Style._current_state}")
            self.L[layer_type][camera_id].draw()

        Style.pop_style()