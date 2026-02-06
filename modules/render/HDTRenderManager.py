# Standard library imports
from enum import IntEnum, auto
from typing import cast

# Third-party imports
from OpenGL.GL import * # type: ignore

# Local application imports
from modules.gl import RenderBase, WindowManager, Shader, Style, clear_color
from modules.render.layers import LayerBase

from modules.DataHub import DataHub, PoseDataHubTypes, DataHubType
from modules.gui.PyReallySimpleGui import Gui
from modules.pose.Frame import FrameField
from modules.render.Settings import Settings
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
    # image layers
    cam_image =     0
    cam_mask =      auto()
    cam_frg=        auto()
    cam_crop =      auto()

    # bbox layers
    box_cam =       auto()
    box_pose_R =    auto()
    box_pose_S =    auto()
    box_pose_I =    auto()

    # ALTERNATIVE FLOW LAYERS
    dense_flow =    auto()
    centre_D_flow = auto()
    sparse_flow =   auto()
    sparse_images = auto()

    # cam composite layers
    cam_bbox =      auto()
    cam_track =     auto()

    # centre layers
    centre_math=    auto()
    centre_cam =    auto()
    centre_mask =   auto()
    centre_frg =    auto()
    centre_pose =   auto()
    centre_motion = auto()

    # Window layers
    angle_W =       auto()
    angle_vel_W =   auto()
    angle_mtn_W =   auto()
    similarity_W =  auto()
    bbox_W =        auto()

    # Frame layers
    mtime_data =    auto()
    angle_vel_F =   auto()
    angle_mtn_F =   auto()
    similarity_F =  auto()
    bbox_F =        auto()

    # composition layers
    sim_blend =     auto()
    flow =          auto()


UPDATE_LAYERS: list[Layers] = [
    Layers.cam_image,
    Layers.cam_mask,
    Layers.cam_frg,
    Layers.cam_crop,

    Layers.centre_math,
    Layers.centre_cam,
    Layers.centre_mask,
    Layers.centre_frg,
    Layers.centre_pose,
    Layers.centre_motion,

    Layers.sim_blend,
    # Layers.centre_D_flow,

    # Layers.dense_flow,
    # Layers.flow_images,
    # Layers.sparse_flow,
]

INTERFACE_LAYERS: list[Layers] = [
    Layers.cam_track,
    Layers.cam_bbox,
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

    # Layers.angle_W,
    # Layers.angle_vel_W,
    # Layers.angle_mtn_W,
    # Layers.similarity_W,
    Layers.bbox_W,
    Layers.angle_vel_F,
    Layers.mtime_data
]

SHOW_CAM: list[Layers] = [
    Layers.cam_image,
    Layers.cam_bbox,
    # Layers.cam_mask,
    # Layers.cam_frg,
]

SHOW_CENTRE: list[Layers] = [
    Layers.centre_cam,
    Layers.centre_mask,
    Layers.centre_frg,
    # Layers.centre_motion,
    Layers.centre_pose,
]

SHOW_POSE: list[Layers] = [
    Layers.box_cam,
    Layers.box_pose_R,
    Layers.box_pose_S,
    Layers.box_pose_I,
]

SHOW_MASK: list[Layers] = [
    # Layers.cam_mask,
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
    # Layers.angle_W,
    # Layers.angle_vel_W,
    # Layers.angle_mtn_W,
    Layers.similarity_W,
    # Layers.angle_F,
    # Layers.angle_vel_F,
    # Layers.angle_mtn_F,
    Layers.similarity_F,
    # Layers.angle_vel_F,
    # Layers.motion_bar,

    # Layers.motion_sim,
    # Layers.field_bar_R,
    # Layers.field_bar_I,
]


PREVIEW_LAYERS: list[Layers] = PREVIEW_CENTRE
FINAL_LAYERS: list[Layers] = SHOW_DATA

class HDTRenderManager(RenderBase):
    def __init__(self, gui: Gui, data_hub: DataHub, settings: Settings) -> None:
        self.num_players: int = settings.num_players
        self.num_cams: int =    settings.num_cams

        # data
        self.data_hub: DataHub = data_hub

        # layers
        self._update_layers: list[Layers] =     UPDATE_LAYERS
        self._interface_layers: list[Layers] =  INTERFACE_LAYERS
        self._preview_layers: list[Layers] =    PREVIEW_LAYERS
        self._draw_layers: list[Layers] =       FINAL_LAYERS

        # camera layers
        self.L: dict[Layers, dict[int, LayerBase]] = {layer: {} for layer in Layers}

        self.line_width: float = 3.0

        for i in range(self.num_cams):
            cam_image =     self.L[Layers.cam_image][i] =   ls.ImageSourceLayer(    i, self.data_hub)
            cam_mask =      self.L[Layers.cam_mask][i] =    ls.MaskSourceLayer(     i, self.data_hub)
            cam_frg =       self.L[Layers.cam_frg][i]=      ls.FrgSourceLayer(      i, self.data_hub)

            sparse_images = self.L[Layers.sparse_images][i] =  ls.FlowSourceLayer(  i, self.data_hub)
            sparse_flow =   self.L[Layers.sparse_flow][i] = ls.OpticalFlowLayer(       sparse_images)
            dense_flow =    self.L[Layers.dense_flow][i] =  ls.DFlowSourceLayer(    i, self.data_hub)

            cam_bbox =      self.L[Layers.cam_bbox][i] =    ls.BBoxRenderer(        i, self.data_hub,   PoseDataHubTypes.pose_I)
            cam_track =     self.L[Layers.cam_track][i] =   ls.CamCompositeLayer(   i, self.data_hub,   PoseDataHubTypes.pose_R,    cam_image.texture, line_width=2.0)
            mtime_data =    self.L[Layers.mtime_data][i] =  ls.MTimeRenderer(       i, self.data_hub,   PoseDataHubTypes.pose_I)

            box_cam =       self.L[Layers.box_cam][i] =     ls.CamBBoxLayer(        i, self.data_hub,   PoseDataHubTypes.pose_I,    cam_image.texture)
            box_pose_R =    self.L[Layers.box_pose_R][i] =  ls.PoseLineLayer(       i, self.data_hub,   PoseDataHubTypes.pose_R,    3.0, 0.0, True, False, (1.0, 1.0, 1.0, 1.0))
            box_pose_S =    self.L[Layers.box_pose_S][i] =  ls.PoseLineLayer(       i, self.data_hub,   PoseDataHubTypes.pose_S,    3.0, 0.0, True, False, (1.0, 1.0, 1.0, 1.0))
            box_pose_I =    self.L[Layers.box_pose_I][i] =  ls.PoseLineLayer(       i, self.data_hub,   PoseDataHubTypes.pose_I,    6.0, 0.0, True, False)

            centre_math =   self.L[Layers.centre_math][i] = ls.CentreGeometry(      i, self.data_hub,   PoseDataHubTypes.pose_I,    16/9)
            centre_mask =   self.L[Layers.centre_mask][i] = ls.CentreMaskLayer(        centre_math,                                 cam_mask.texture)
            centre_frg =    self.L[Layers.centre_frg][i] =  ls.CentreFrgLayer(         centre_math,                                 cam_frg.texture, centre_mask.texture)
            centre_cam =    self.L[Layers.centre_cam][i] =  ls.CentreCamLayer(         centre_math,                                 cam_image.texture,  centre_mask.texture)
            centre_pose =   self.L[Layers.centre_pose][i] = ls.CentrePoseLayer(        centre_math,                                 line_width=3.0, line_smooth=0.0, use_scores=False, color=COLORS[i % len(COLORS)])
            centre_motion = self.L[Layers.centre_motion][i]=ls.MotionMultiply(      i, self.data_hub,   PoseDataHubTypes.pose_I,    centre_mask.texture)
            centre_D_flow = self.L[Layers.centre_D_flow][i]=ls.CentreDenseFlowLayer(   centre_math,                                 dense_flow.texture, centre_mask.texture)

            sim_blend =     self.L[Layers.sim_blend][i] =   ls.SimilarityBlend(     i, self.data_hub,   PoseDataHubTypes.pose_I,    cast(dict[int, ls.MotionMultiply], self.L[Layers.centre_motion]))
            flow =          self.L[Layers.flow][i] =        ls.FlowLayer(              sim_blend)

            gpu_crop =      self.L[Layers.cam_crop][i] =    ls.CropSourceLayer(     i, self.data_hub)

            angle_W =       self.L[Layers.angle_W][i] =     ls.AngleWindowLayer(    i, self.data_hub, self.line_width)
            angle_vel_W =   self.L[Layers.angle_vel_W][i] = ls.AngleVelWindowLayer( i, self.data_hub, self.line_width)
            angle_mtn_W =   self.L[Layers.angle_mtn_W][i] = ls.AngleMtnWindowLayer( i, self.data_hub, self.line_width)
            similarity_W =  self.L[Layers.similarity_W][i] =ls.SimilarityWindowLayer(  i, self.data_hub, self.line_width)
            bbox_W =        self.L[Layers.bbox_W][i] =      ls.BBoxWindowLayer(     i, self.data_hub, self.line_width)

            angle_vel_F =   self.L[Layers.angle_vel_F][i] = ls.AngleVelLayer(       i, self.data_hub, PoseDataHubTypes.pose_I, line_thickness=1.0, line_smooth=1.0)
            angle_mtn_F =   self.L[Layers.angle_mtn_F][i] = ls.AngleMtnFrameLayer(  i, self.data_hub, PoseDataHubTypes.pose_I, line_thickness=2.0, line_smooth=2.0)
            similarity_F =  self.L[Layers.similarity_F][i] =ls.SimilarityFrameLayer(i, self.data_hub, PoseDataHubTypes.pose_I, line_thickness=4.0, line_smooth=2.0)
            bbox_F =        self.L[Layers.bbox_F][i] =      ls.BBoxFrameLayer(      i, self.data_hub, PoseDataHubTypes.pose_I, line_thickness=2.0, line_smooth=2.0)

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
            self.L[Layers.cam_track][i].allocate(w , h, GL_RGBA)
            w, h = self.subdivision.get_allocation_size('preview', i)
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
            self.L[Layers.box_pose_R][i].color = (1.0, 0.0, 0.0, 1.0)    #type: ignore
            self.L[Layers.box_pose_S][i].color = (0.0, 1.0, 0.0, 1.0)    #type: ignore
            self.L[Layers.centre_cam][i].use_mask = True    #type: ignore
            self.L[Layers.centre_frg][i].use_mask = True    #type: ignore
            self.L[Layers.centre_mask][i].blur_steps = 0    #type: ignore
            self.L[Layers.angle_W][i].line_width = 3.0      #type: ignore

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