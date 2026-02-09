# Standard library imports
from enum import IntEnum, auto
from typing import cast

# Third-party imports
from OpenGL.GL import * # type: ignore

# Local application imports
from modules.gl import RenderBase, WindowManager, Shader, Style, clear_color
from modules.render.layers import LayerBase

from modules.DataHub import DataHub, Stage, DataHubType, PoseDataHubTypes
from modules.gui.PyReallySimpleGui import Gui
from modules.pose.Frame import FrameField
from modules.render.Settings import Config, LayerMode
from modules.utils.PointsAndRects import Rect, Point2f

# Render Imports
from modules.render.CompositionSubdivider import make_subdivision, SubdivisionRow, Subdivision
from modules.render import layers as ls

from modules.utils.HotReloadMethods import HotReloadMethods


COLORS: list[tuple[float, float, float, float]] = [
    (1.0, 0.0, 0.0, 1.85),
    (0.0, 0.0, 1.0, 1.85),
    (0.0, 1.0, 0.0, 1.85),
]

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
    centre_motion = auto()

    # Data layers (configurable slots A and B, all pre-allocated)
    data_B_W =      auto()
    data_B_F =      auto()
    data_B_AV =     auto()
    data_A_W =      auto()
    data_A_F =      auto()
    data_A_AV =     auto()
    data_time =     auto()

    # composition layers
    sim_blend =     auto()
    flow =          auto()


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

    Layers.centre_motion,
    Layers.sim_blend,
]

INTERFACE_LAYERS: list[Layers] = [
    Layers.poser,
]

LARGE_LAYERS: list[Layers] = [
    Layers.centre_cam,
    Layers.centre_mask,
    Layers.sim_blend,
    Layers.centre_pose,
]

PREVIEW_CENTRE: list[Layers] = [
    Layers.centre_frg,
    Layers.centre_pose,
    Layers.data_time,
]

SHOW_CAM: list[Layers] = [
    Layers.poser
    # Layers.cam_image,
    # Layers.bbox_bbox,
    # Layers.bbox_pose_A
    # Layers.cam_mask,
    # Layers.cam_frg,
]

SHOW_CENTRE: list[Layers] = [
    Layers.centre_cam,
    Layers.centre_mask,
    # Layers.centre_frg,
    # Layers.centre_motion,
    Layers.centre_pose,
]

SHOW_POSE: list[Layers] = [
    Layers.tracker,
]

SHOW_MASK: list[Layers] = [
    Layers.cam_mask,
    Layers.centre_mask,
    # Layers.centre_motion,
    Layers.centre_pose,
    # Layers.cam_crop
]

SHOW_COMP: list[Layers] = [
    Layers.flow,
    # Layers.centre_pose,
    Layers.sim_blend,
    Layers.cam_frg,
]

SHOW_DATA: list[Layers] = [
    Layers.centre_motion,
    Layers.data_B_W,
    Layers.data_B_F,
    Layers.data_B_AV,
    Layers.data_A_W,
    Layers.data_A_F,
    Layers.data_A_AV,
]


PREVIEW_LAYERS: list[Layers] = PREVIEW_CENTRE
FINAL_LAYERS: list[Layers] = SHOW_POSE

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

        # camera layers
        self.L: dict[Layers, dict[int, LayerBase]] = {layer: {} for layer in Layers}

        # Shared configs for Centre layers
        centre_geometry_config =    ls.CentreGeometryConfig(stage=Stage.LERP, cam_aspect=16/9, target_top_x=0.5, target_top_y=0.33, target_bottom_x=0.5, target_bottom_y=0.6, dst_aspectratio=9/16)
        centre_mask_config =        ls.CentreMaskConfig(    blend_factor=0.2, blur_steps=0, blur_radius=8.0)
        centre_cam_config =         ls.CentreCamConfig(     blend_factor=0.5, mask_opacity=1.0, use_mask=True)
        centre_frg_config =         ls.CentreFrgConfig(     blend_factor=0.2, mask_opacity=1.0, use_mask=True)
        centre_pose_config =        ls.CentrePoseConfig(    line_width=3.0, line_smooth=0.0, use_scores=False, draw_anchors=True)

        # Shared configs for Data layers
        grey: tuple[float, float, float, float] = (0.5, 0.5, 0.5, 1.0)
        ff = ls.ScalarFrameField.angle_motion
        data_A_config = ls.DataLayerConfig(active=False, feature_field=ff, stage=Stage.SMOOTH,  line_width=3.0, line_smooth=1.0, use_scores=False, render_labels=True, colors=None)
        data_B_config = ls.DataLayerConfig(active=False, feature_field=ff, stage=Stage.LERP,    line_width=6.0, line_smooth=6.0, use_scores=False, render_labels=True, colors=[grey])

        # Shared configs for Cam layers
        bbox_config =           ls.BBoxRendererConfig(      stage=Stage.LERP, line_width=2)
        cam_composite_config =  ls.TrackerCompositorConfig( stage=Stage.LERP, track_line_width=2.0, bbox_line_width=2)

        # Shared configs for Pose renderers
        pose_line_A_config =    ls.PoseLineConfig(     stage=Stage.LERP, line_width=3.0, line_smooth=0.0, use_scores=True, use_bbox=False)
        pose_line_B_config =    ls.PoseLineConfig(     stage=Stage.RAW,  line_width=6.0, line_smooth=0.0, use_scores=True, use_bbox=False)
        mtime_config =          ls.MTimeRendererConfig(     stage=Stage.LERP)
        cam_crop_config =       ls.CropConfig(      stage=Stage.LERP)
        track_pose_composite_config = ls.PoseCompositorConfig(stage=Stage.LERP, line_width=2.0, line_smooth=0.0)
        for i in range(self.num_cams):
            color: tuple[float, float, float, float] = COLORS[i % len(COLORS)]
            cam_image =     self.L[Layers.cam_image][i] =   ls.ImageSourceLayer(    i, self.data_hub)
            cam_mask =      self.L[Layers.cam_mask][i] =    ls.MaskSourceLayer(     i, self.data_hub)
            cam_frg =       self.L[Layers.cam_frg][i]=      ls.FrgSourceLayer(      i, self.data_hub)

            cam_comp =      self.L[Layers.poser][i] =   ls.TrackerCompositor(   i, self.data_hub, cam_image.texture, cam_composite_config)
            track_comp =    self.L[Layers.tracker][i] = ls.PoseCompositor(i, self.data_hub, cam_image.texture, color, track_pose_composite_config)

            centre_geometry=self.L[Layers.centre_math][i] = ls.CentreGeometry(      i, self.data_hub,       centre_geometry_config)
            centre_mask =   self.L[Layers.centre_mask][i] = ls.CentreMaskLayer(        centre_geometry,     cam_mask.texture,   centre_mask_config)
            centre_cam =    self.L[Layers.centre_cam][i] =  ls.CentreCamLayer(         centre_geometry,     cam_image.texture,  centre_mask.texture, centre_cam_config)
            centre_frg =    self.L[Layers.centre_frg][i] =  ls.CentreFrgLayer(         centre_geometry,     cam_frg.texture,    centre_mask.texture, centre_frg_config)
            centre_pose =   self.L[Layers.centre_pose][i] = ls.CentrePoseLayer(        centre_geometry,     color,              centre_pose_config)

            centre_motion = self.L[Layers.centre_motion][i]=ls.MotionMultiply(      i, self.data_hub,   PoseDataHubTypes.pose_I,    centre_mask.texture)
            sim_blend =     self.L[Layers.sim_blend][i] =   ls.SimilarityBlend(     i, self.data_hub,   PoseDataHubTypes.pose_I,    cast(dict[int, ls.MotionMultiply], self.L[Layers.centre_motion]))
            flow =          self.L[Layers.flow][i] =        ls.FlowLayer(              sim_blend)

            gpu_crop =      self.L[Layers.cam_crop][i] =    ls.CropSourceLayer(     i, self.data_hub)

            self.L[Layers.data_A_W][i]  = ls.FeatureWindowLayer(i, self.data_hub, data_A_config)
            self.L[Layers.data_A_F][i]  = ls.FeatureFrameLayer( i, self.data_hub, data_A_config)
            self.L[Layers.data_A_AV][i] = ls.AngleVelLayer(     i, self.data_hub, data_A_config)
            self.L[Layers.data_B_W][i]  = ls.FeatureWindowLayer(i, self.data_hub, data_B_config)
            self.L[Layers.data_B_F][i]  = ls.FeatureFrameLayer( i, self.data_hub, data_B_config)
            self.L[Layers.data_B_AV][i] = ls.AngleVelLayer(     i, self.data_hub, data_B_config)

            mtime_data =    self.L[Layers.data_time][i] =  ls.MTimeRenderer(       i, self.data_hub, mtime_config)

        # Bind data config to layer configs - propagates config changes automatically
        settings.bind(
            {i: cast(ls.FeatureWindowLayer, self.L[Layers.data_A_W][i])._config for i in range(self.num_cams)},
            {i: cast(ls.FeatureFrameLayer,  self.L[Layers.data_A_F][i])._config for i in range(self.num_cams)},
            {i: cast(ls.AngleVelLayer,      self.L[Layers.data_A_AV][i])._config for i in range(self.num_cams)},
            {i: cast(ls.FeatureWindowLayer, self.L[Layers.data_B_W][i])._config for i in range(self.num_cams)},
            {i: cast(ls.FeatureFrameLayer,  self.L[Layers.data_B_F][i])._config for i in range(self.num_cams)},
            {i: cast(ls.AngleVelLayer,      self.L[Layers.data_B_AV][i])._config for i in range(self.num_cams)},
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

            # DO TEST SETTINGS HERE
            self.L[Layers.centre_cam][i].use_mask = True    #type: ignore
            self.L[Layers.centre_frg][i].use_mask = True    #type: ignore
            self.L[Layers.centre_mask][i].blur_steps = 0    #type: ignore

        self._update_layers = UPDATE_LAYERS
        self._draw_layers = FINAL_LAYERS
        # self._draw_layers = BOX_LAYERS
        self._preview_layers = PREVIEW_LAYERS

    def draw_secondary(self, monitor_id: int, width: int, height: int) -> None:
        glViewport(0, 0, width, height)
        clear_color()

        Style.reset_state()
        Style.set_blend_mode(Style.BlendMode.ALPHA)

        camera_id: int = self.secondary_order_list.index(monitor_id)
        for layer_type in self._draw_layers:
            self.L[layer_type][camera_id].draw()