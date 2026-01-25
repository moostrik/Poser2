# Standard library imports
from enum import IntEnum, auto
from typing import cast

# Third-party imports
from OpenGL.GL import * # type: ignore

# Local application imports
from modules.gl import RenderBase, WindowManager, Shader, Style
from modules.render.layers import LayerBase

from modules.DataHub import DataHub, PoseDataHubTypes, SimilarityDataHubType
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
    # render layers
    cam_image =     0
    cam_mask =      auto()
    dense_flow =    auto()
    flow_image =    auto()
    sparse_flow =   auto()

    # data layers
    cam_bbox =      auto()
    cam_track =     auto()
    angle_data =    auto()
    mtime_data =    auto()
    field_bar_R =   auto()
    field_bar_I =   auto()
    angle_bar =     auto()
    motion_bar =    auto()
    motion_sim =    auto()

    # bbox layers
    box_cam =       auto()
    box_pose_I =    auto()
    box_pose_R =    auto()

    # centre layers
    centre_math=    auto()
    centre_mask =   auto()
    centre_cam =    auto()
    centre_pose =   auto()
    centre_D_flow = auto()
    centre_motion = auto()

    sim_blend =     auto()

    flow =          auto()

UPDATE_LAYERS: list[Layers] = [
    Layers.cam_image,
    Layers.cam_mask,
    Layers.dense_flow,
    Layers.flow_image,
    Layers.sparse_flow,

    Layers.centre_math,
    Layers.centre_cam,
    Layers.centre_mask,
    Layers.centre_pose,
    Layers.centre_D_flow,
    Layers.centre_motion,
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

PREVIEW_LAYERS: list[Layers] = [
    # Layers.centre_cam,
    Layers.centre_pose,
    # Layers.centre_motion,
    # Layers.sim_blend,
    # Layers.angle_data,
    # Layers.prev_mt,
    # Layers.cam_mask,
    # Layers.flow,

    # Layers.centre_motion,
    # Layers.cam_image,
    # Layers.cam_mask,
    # Layers.dense_flow,


    # Layers.centre_cam,

    # Layers.sim_blend,
    # Layers.flow,
]

FINAL_LAYERS: list[Layers] = [
    # Layers.cam_image,
    # Layers.cam_mask,
    # Layers.dense_flow,
    # Layers.flow_image,
    # Layers.sparse_flow,

    # Layers.cam_bbox,
    # Layers.cam_track,
    # Layers.angle_data,
    # Layers.mtime_data,
    # Layers.field_bar_R,
    # Layers.field_bar_I,
    # Layers.angle_bar,
    # Layers.motion_bar,
    # Layers.motion_sim,

    # Layers.box_cam,
    # Layers.box_pose_I,
    # Layers.box_pose_R,

    # Layers.centre_mask,
    # Layers.centre_cam,
    # Layers.centre_pose,
    # Layers.angle_bar,
    # Layers.centre_D_flow,
    # Layers.centre_motion,
    # Layers.centre_flow,

    # Layers.sim_blend,

    # Layers.centre_cam,
    # Layers.centre_mask,
    # Layers.centre_pose,
    # Layers.sim_blend,
    # Layers.centre_pose,
    Layers.sim_blend,
    Layers.flow,
]

BOX_LAYERS: list[Layers] = [
    Layers.box_cam,
    Layers.box_pose_R,
    Layers.box_pose_I,
]

class HDTRenderManager(RenderBase):
    def __init__(self, gui: Gui, data_hub: DataHub, settings: Settings) -> None:
        self.num_players: int = settings.num_players
        self.num_cams: int =    settings.num_cams
        num_R_streams: int =    settings.num_R
        R_stream_capacity: int= int(settings.stream_capacity)  # 10 seconds buffer

        # data
        self.data_hub: DataHub = data_hub

        # layers
        self._update_layers: list[Layers] =     UPDATE_LAYERS
        self._interface_layers: list[Layers] =  INTERFACE_LAYERS
        self._preview_layers: list[Layers] =    PREVIEW_LAYERS
        self._draw_layers: list[Layers] =       FINAL_LAYERS

        # camera layers
        self.L: dict[Layers, dict[int, LayerBase]] = {layer: {} for layer in Layers}

        for i in range(self.num_cams):
            cam_image =     self.L[Layers.cam_image][i] =   ls.ImageSourceLayer(    i, self.data_hub)
            cam_mask =      self.L[Layers.cam_mask][i] =    ls.MaskSourceLayer(     i, self.data_hub)
            dense_flow =    self.L[Layers.dense_flow][i] =  ls.DFlowSourceLayer(    i, self.data_hub)
            flow_image =    self.L[Layers.flow_image][i] =  ls.FlowSourceLayer(     i, self.data_hub)
            sparse_flow =   self.L[Layers.sparse_flow][i] = ls.OpticalFlowLayer(       flow_image)

            cam_bbox =      self.L[Layers.cam_bbox][i] =    ls.BBoxCamRenderer(     i, self.data_hub,   PoseDataHubTypes.pose_I)
            cam_track =     self.L[Layers.cam_track][i] =   ls.CamCompositeLayer(   i, self.data_hub,   PoseDataHubTypes.pose_R,    cam_image.texture, line_width=2.0)
            angle_data =    self.L[Layers.angle_data][i] =  ls.PDLayer(             i, self.data_hub)
            mtime_data =    self.L[Layers.mtime_data][i] =  ls.PoseMTimeRenderer(   i, self.data_hub,   PoseDataHubTypes.pose_I)
            field_bar_R =   self.L[Layers.field_bar_R][i] = ls.PoseBarScalarLayer(  i, self.data_hub,   PoseDataHubTypes.pose_R,    FrameField.angles, line_thickness=4.0, line_smooth=16.0, color = (0.0, 0.0, 0.0, 0.33))
            field_bar_I =   self.L[Layers.field_bar_I][i] = ls.PoseBarScalarLayer(  i, self.data_hub,   PoseDataHubTypes.pose_I,    FrameField.angles, line_thickness=2.0, line_smooth=2.0)
            angle_bar =     self.L[Layers.angle_bar][i] =   ls.PoseBarADLayer(      i, self.data_hub,   PoseDataHubTypes.pose_I)
            motion_bar =    self.L[Layers.motion_bar][i] =  ls.PoseBarMLayer(       i, self.data_hub,   PoseDataHubTypes.pose_I,    FrameField.angle_motion, line_thickness=2.0, line_smooth=2.0)
            motion_sim =    self.L[Layers.motion_sim][i] =  ls.PoseBarSLayer(       i, self.data_hub,   PoseDataHubTypes.pose_I)

            box_cam =       self.L[Layers.box_cam][i] =     ls.CamBBoxLayer(        i, self.data_hub,   PoseDataHubTypes.pose_I,    cam_image.texture)
            box_pose_I =    self.L[Layers.box_pose_I][i] =  ls.PoseLineLayer(       i, self.data_hub,   PoseDataHubTypes.pose_I,    1.0, 0.0, True, False)
            box_pose_R =    self.L[Layers.box_pose_R][i] =  ls.PoseLineLayer(       i, self.data_hub,   PoseDataHubTypes.pose_R,    0.5, 0.0, True, False, (1.0, 1.0, 1.0, 1.0))

            centre_math =   self.L[Layers.centre_math][i] = ls.CentreGeometry(      i, self.data_hub,   PoseDataHubTypes.pose_I,    16/9)
            centre_mask =   self.L[Layers.centre_mask][i] = ls.CentreMaskLayer(        centre_math,                                 cam_mask.texture)
            centre_cam =    self.L[Layers.centre_cam][i] =  ls.CentreCamLayer(         centre_math,                                 cam_image.texture,  centre_mask.texture)
            centre_D_flow = self.L[Layers.centre_D_flow][i]=ls.CentreDenseFlowLayer(   centre_math,                                 dense_flow.texture, centre_mask.texture)
            centre_pose =   self.L[Layers.centre_pose][i] = ls.CentrePoseLayer(        centre_math,                                 line_width=3.0, line_smooth=0.0, use_scores=False, color=COLORS[i % len(COLORS)])
            centre_motion = self.L[Layers.centre_motion][i]=ls.MotionMultiply(      i, self.data_hub,   PoseDataHubTypes.pose_I,    centre_mask.texture)

            sim_blend =     self.L[Layers.sim_blend][i] =   ls.SimilarityBlend(     i, self.data_hub,   PoseDataHubTypes.pose_I,    cast(dict[int, ls.MotionMultiply], self.L[Layers.centre_motion]))
            flow =          self.L[Layers.flow][i] =        ls.FlowLayer(              sim_blend)

        # global layers
        self.pose_sim_layer =   ls.SimilarityLayer(num_R_streams, R_stream_capacity, self.data_hub, SimilarityDataHubType.sim_P, ls.AggregationMethod.HARMONIC_MEAN, 2.0)

        # composition
        self.subdivision_rows: list[SubdivisionRow] = [
            SubdivisionRow(name='track',        columns=self.num_cams,    rows=1, src_aspect_ratio=1.0,  padding=Point2f(1.0, 1.0)),
            SubdivisionRow(name='preview',      columns=self.num_players, rows=1, src_aspect_ratio=9/16, padding=Point2f(1.0, 1.0)),
            SubdivisionRow(name='similarity',   columns=1,                rows=1, src_aspect_ratio=6.0,  padding=Point2f(1.0, 1.0))
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
        self.pose_sim_layer.allocate(w, h, GL_RGBA)
        for i in range(self.num_cams):
            w, h = self.subdivision.get_allocation_size('track', i)
            self.L[Layers.cam_track][i].allocate(w , h, GL_RGBA)
            w, h = self.subdivision.get_allocation_size('preview', i)
            self.L[Layers.angle_data][i].allocate(w, h, GL_RGBA)

    def deallocate(self) -> None:
        self.pose_sim_layer.deallocate()
        for cam_dict in self.L.values():
            for layer in cam_dict.values():
                layer.deallocate()

    def draw_main(self, width: int, height: int) -> None:

        glViewport(0, 0, width, height)
        self.data_hub.notify_update()
        self.pose_sim_layer.update()
        seen: set[Layers] = set()
        for layer_type in self._update_layers + self._interface_layers + self._draw_layers + self._preview_layers:
            if layer_type not in seen:
                seen.add(layer_type)
                for layer in self.L[layer_type].values():
                    layer.update()

        Style.reset_state()
        Style.set_blend_mode(Style.BlendMode.ALPHA)

        glViewport(0, 0, width, height)

        glClearColor(0.0, 0.0, 0.0, 1.0)
        glClear(GL_COLOR_BUFFER_BIT)

        # Global layer

        sim_rect = self.subdivision.get_rect('similarity', 0)
        glViewport(int(sim_rect.x), int(height - sim_rect.y - sim_rect.height), int(sim_rect.width), int(sim_rect.height))
        self.pose_sim_layer.draw()

        # Interface layers
        for i in range(self.num_cams):
            # View.set_view(width, height)
            track_rect: Rect = self.subdivision.get_rect('track', i)
            glViewport(int(track_rect.x), int(height - track_rect.y - track_rect.height), int(track_rect.width), int(track_rect.height))

            for layer_type in self._interface_layers:
                # glViewport(int(track_rect.x), int(height - track_rect.y - track_rect.height), int(track_rect.width), int(track_rect.height))

                self.L[layer_type][i].draw()

        # Preview layers
        for i in range(self.num_cams):
            preview_rect: Rect = self.subdivision.get_rect('preview', i)
            glViewport(int(preview_rect.x), int(height - preview_rect.y - preview_rect.height), int(preview_rect.width), int(preview_rect.height))

            for layer_type in self._preview_layers:
                # glViewport(int(preview_rect.x), int(height - preview_rect.y - preview_rect.height), int(preview_rect.width), int(preview_rect.height))

                self.L[layer_type][i].draw()
            self.L[Layers.centre_cam][i].use_mask = True #type: ignore
            self.L[Layers.centre_mask][i].blur_steps = 0 #type: ignore

        self._draw_layers = FINAL_LAYERS
        # self._draw_layers = BOX_LAYERS
        self._preview_layers = PREVIEW_LAYERS

    def draw_secondary(self, monitor_id: int, width: int, height: int) -> None:
        Style.reset_state()
        Style.set_blend_mode(Style.BlendMode.ALPHA)

        glViewport(0, 0, width, height)

        glClearColor(0.0, 0.0, 0.0, 0.0)
        glClear(GL_COLOR_BUFFER_BIT)

        camera_id: int = self.secondary_order_list.index(monitor_id)

        for layer_type in self._draw_layers:
            self.L[layer_type][camera_id].draw()