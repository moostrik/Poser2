# Standard library imports
from enum import IntEnum, auto
from time import perf_counter
from typing import cast

# Third-party imports
from OpenGL.GL import * # type: ignore

# Local application imports
from modules.gl.RenderBase import RenderBase
from modules.gl.WindowManager import WindowManager
from modules.gl.LayerBase import LayerBase

from modules.DataHub import DataHub, PoseDataHubTypes, SimilarityDataHubType
from modules.gui.PyReallySimpleGui import Gui
from modules.pose.Frame import FrameField
from modules.render.Settings import Settings
from modules.utils.PointsAndRects import Rect, Point2f

# Render Imports
from modules.render.CompositionSubdivider import make_subdivision, SubdivisionRow, Subdivision
from modules.render import layers

from modules.utils.HotReloadMethods import HotReloadMethods


class Layers(IntEnum):
    cam_image =     0
    cam_bbox =      auto()
    cam_track =     auto()
    prev_angles =   auto()
    prev_mt =       auto()
    box_cam =       auto()
    box_pose_I =    auto()
    box_pose_R =    auto()
    centre_cam =    auto()
    centre_pose =   auto()
    sim_blend =     auto()
    field_bar_R =   auto()
    field_bar_I =   auto()
    angle_bar =     auto()
    motion_bar =    auto()
    motion_sim =    auto()

    mask = auto()

PREVIEW_LAYERS: list[Layers] = [
    Layers.centre_cam,
    # Layers.sim_blend,
    # Layers.centre_pose,
    Layers.prev_angles,
    Layers.prev_mt,
]

BOX_LAYERS: list[Layers] = [
    Layers.box_cam,
    Layers.box_pose_R,
    Layers.box_pose_I,
]

FINAL_LAYERS: list[Layers] = [
    Layers.centre_cam,
    Layers.sim_blend,
    Layers.centre_pose,
    # Layers.angle_bar,
    # Layers.motion_sim,
    # Layers.mask,
]

LARGE_LAYERS: list[Layers] = [
    Layers.centre_cam,
    Layers.sim_blend,
    Layers.centre_pose,
]

DYNAMIC_LAYERS: list[Layers] = [
    Layers.prev_angles,
]

COLORS: list[tuple[float, float, float, float]] = [
    (1.0, 0.0, 0.0, 1.85),
    (0.0, 1.0, 1.0, 1.85),
    (0.0, 1.0, 0.0, 1.85),
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
        self._update_layers: list[Layers] =     [Layers.cam_image, Layers.centre_cam]
        self._interface_layers: list[Layers] =  [Layers.cam_track, Layers.cam_bbox]
        self._preview_layers: list[Layers] =    PREVIEW_LAYERS
        self._draw_layers: list[Layers] =       FINAL_LAYERS

        # camera layers
        self.L: dict[Layers, dict[int, LayerBase]] = {layer: {} for layer in Layers}

        for i in range(self.num_cams):
            self.L[Layers.cam_image][i] =   layers.CamImageRenderer(i, self.data_hub)
            self.L[Layers.cam_bbox][i] =    layers.CamBBoxRenderer(i, self.data_hub, PoseDataHubTypes.pose_I)
            self.L[Layers.cam_track][i] =   layers.CamCompositeLayer(i, self.data_hub, PoseDataHubTypes.pose_R, cast(layers.CamImageRenderer, self.L[Layers.cam_image][i]))
            self.L[Layers.prev_angles][i] = layers.PDLineLayer(i, self.data_hub)
            self.L[Layers.prev_mt][i] =     layers.PoseMotionTimeRenderer(i, self.data_hub, PoseDataHubTypes.pose_I)
            self.L[Layers.box_cam][i] =     layers.PoseCamLayer(i, self.data_hub, PoseDataHubTypes.pose_I, cast(layers.CamImageRenderer, self.L[Layers.cam_image][i]))
            self.L[Layers.box_pose_I][i] =  layers.PoseLineLayer(i, self.data_hub, PoseDataHubTypes.pose_I, 1.0, 0.0, True, False)
            self.L[Layers.box_pose_R][i] =  layers.PoseLineLayer(i, self.data_hub, PoseDataHubTypes.pose_R, 0.5, 0.0, True, False, (1.0, 1.0, 1.0, 1.0))

            self.L[Layers.field_bar_R][i] = layers.PoseScalarBarLayer(i, self.data_hub, PoseDataHubTypes.pose_R, FrameField.angles, 4.0, 16.0, (0.0, 0.0, 0.0, 0.33))
            self.L[Layers.field_bar_I][i] = layers.PoseScalarBarLayer(i, self.data_hub, PoseDataHubTypes.pose_I, FrameField.angles, 2.0, 2.0)
            self.L[Layers.angle_bar][i] =   layers.PoseAngleDeltaBarLayer(i, self.data_hub, PoseDataHubTypes.pose_I)
            self.L[Layers.motion_bar][i] =  layers.PoseMotionBarLayer(i, self.data_hub, PoseDataHubTypes.pose_I, FrameField.angle_motion, 2.0, 2.0)
            self.L[Layers.motion_sim][i]=   layers.PoseMotionSimLayer(i, self.data_hub, PoseDataHubTypes.pose_I)

            self.L[Layers.sim_blend][i] =   layers.SimilarityBlend(i, self.data_hub, PoseDataHubTypes.pose_I, cast(dict[int, layers.CentreCamLayer], self.L[Layers.centre_cam]))
            self.L[Layers.centre_cam][i] =  layers.CentreCamLayer(i, self.data_hub, PoseDataHubTypes.pose_I, cast(layers.CamImageRenderer, self.L[Layers.cam_image][i]))
            self.L[Layers.centre_pose][i] = layers.CentrePoseLayer(i, self.data_hub, PoseDataHubTypes.pose_I, 50.0, 25.0, False, False, COLORS[i % len(COLORS)])
            cast(layers.CentreCamLayer, self.L[Layers.centre_cam][i]).set_points_callback(cast(layers.ElectricLayer, self.L[Layers.centre_pose][i]).setCentrePoints)

            self.L[Layers.mask][i] =        layers.MaskLayer(i, self.data_hub)

        # global layers
        self.pose_sim_layer =   layers.SimilarityLineLayer(num_R_streams, R_stream_capacity, self.data_hub, SimilarityDataHubType.sim_P, layers.AggregationMethod.HARMONIC_MEAN, 2.0)

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

    def allocate_window_renders(self) -> None:
        w, h = self.subdivision.get_allocation_size('similarity', 0)
        self.pose_sim_layer.allocate(w, h, GL_RGBA)

        for i in range(self.num_cams):
            w, h = self.subdivision.get_allocation_size('track', i)
            self.L[Layers.cam_track][i].allocate(w , h, GL_RGBA)
            w, h = self.subdivision.get_allocation_size('preview', i)
            self.L[Layers.prev_angles][i].allocate(w, h, GL_RGBA)

    def deallocate(self) -> None:
        self.pose_sim_layer.deallocate()
        for cam_dict in self.L.values():
            for layer in cam_dict.values():
                layer.deallocate()

    def draw_main(self, width: int, height: int) -> None:

        self.data_hub.notify_update()

        self.pose_sim_layer.update()
        seen: set[Layers] = set()
        for layer_type in self._update_layers + self._interface_layers + self._draw_layers + self._preview_layers:
            if layer_type not in seen:
                seen.add(layer_type)
                for layer in self.L[layer_type].values():
                    layer.update()

        glEnable(GL_TEXTURE_2D)
        glEnable(GL_BLEND)
        self.setView(width, height)
        glClearColor(0.0, 0.0, 0.0, 1.0)
        glClear(GL_COLOR_BUFFER_BIT)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        # Global layer
        self.pose_sim_layer.draw(self.subdivision.get_rect('similarity', 0))

        # Interface layers
        for i in range(self.num_cams):
            track_rect: Rect = self.subdivision.get_rect('track', i)
            for layer_type in self._interface_layers:
                self.L[layer_type][i].draw(track_rect)

        # Draw layers
        for i in range(self.num_cams):
            preview_rect: Rect = self.subdivision.get_rect('preview', i)
            for layer_type in self._preview_layers:
                self.L[layer_type][i].draw(preview_rect)

    def draw_secondary(self, monitor_id: int, width: int, height: int) -> None:
        glEnable(GL_TEXTURE_2D)
        glEnable(GL_BLEND)
        self.setView(width, height)
        glClearColor(0.0, 0.0, 0.0, 1.0)
        glClear(GL_COLOR_BUFFER_BIT)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        camera_id: int = self.secondary_order_list.index(monitor_id)
        draw_rect = Rect(0, 0, width, height)

        for layer_type in self._draw_layers:
            self.L[layer_type][camera_id].draw(draw_rect)
            # cast(layers.PoseLineLayer, self.L[Layers.centre_pose_L][camera_id]).line_width = 50.0
            # cast(layers.PoseLineLayer, self.L[Layers.centre_pose_L][camera_id]).line_smooth = 25.0
            # cast(layers.PoseLineLayer, self.L[Layers.centre_pose_L][camera_id]).color = (1.0, 1.0, 0.0, 0.85)

        self._draw_layers = FINAL_LAYERS
        self._preview_layers = PREVIEW_LAYERS
        # self._draw_layers = BOX_LAYERS