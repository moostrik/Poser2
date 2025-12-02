# Standard library imports
from time import perf_counter

# Third-party imports
from OpenGL.GL import * # type: ignore

# Local application imports
from modules.gl.RenderBase import RenderBase
from modules.gl.WindowManager import WindowManager

from modules.DataHub import DataHub, PoseDataHubTypes, SimilarityDataHubType
from modules.gui.PyReallySimpleGui import Gui
from modules.pose.Frame import FrameField
from modules.render.Settings import Settings
from modules.utils.PointsAndRects import Rect, Point2f

# Render Imports
from modules.render.CompositionSubdivider import make_subdivision, SubdivisionRow, Subdivision
from modules.render.layers import CamCompositeLayer, PoseCamLayer, CentreCamLayer
from modules.render.layers import PoseAngleDeltaBarLayer, PoseScalarBarLayer, PoseMotionBarLayer, PDLineLayer
from modules.render.layers import SimilarityLineLayer, AggregationMethod, SimilarityBlend
from modules.render.renderers import CamImageRenderer, PoseMeshRenderer, CamBBoxRenderer, PoseMotionTimeRenderer

from modules.utils.HotReloadMethods import HotReloadMethods


class HDTRenderManager(RenderBase):
    def __init__(self, gui: Gui, data_hub: DataHub, settings: Settings) -> None:
        self.num_players: int = settings.num_players
        self.num_cams: int =    settings.num_cams
        num_R_streams: int =    settings.num_R
        R_stream_capacity: int= int(settings.stream_capacity)  # 10 seconds buffer

        # data
        self.data_hub: DataHub = data_hub

        # renderers
        self.cam_img_renderers:     dict[int, CamImageRenderer] = {}
        self.mesh_renderers_raw:    dict[int, PoseMeshRenderer] = {}
        self.mesh_renderers:        dict[int, PoseMeshRenderer] = {}
        self.cam_bbox_renderers:    dict[int, CamBBoxRenderer] = {}
        self.motion_time_renderers: dict[int, PoseMotionTimeRenderer] = {}

        # layers
        self.cam_track_layers:      dict[int, CamCompositeLayer] = {}
        self.pose_cam_layers:       dict[int, PoseCamLayer] = {}
        self.centre_cam_layers:     dict[int, CentreCamLayer] = {}
        self.sim_blend_layers:      dict[int, SimilarityBlend] = {}
        self.pd_line_layers:        dict[int, PDLineLayer] = {}
        self.field_bar_layers_raw:  dict[int, PoseScalarBarLayer] = {}
        self.field_bar_layers:      dict[int, PoseScalarBarLayer] = {}
        self.angle_bar_layers:      dict[int, PoseAngleDeltaBarLayer] = {}
        self.motion_bar_layers:     dict[int, PoseMotionBarLayer] = {}
        self.sim_blend_layers:      dict[int, SimilarityBlend] = {}

        # self.line_field_layers:     dict[int, LineFieldLayer] = {}

        self.pose_sim_layer =   SimilarityLineLayer(num_R_streams, R_stream_capacity, self.data_hub, SimilarityDataHubType.sim_P, AggregationMethod.HARMONIC_MEAN, 2.0)
        # self.motion_corr_stream_layer = CorrelationStreamLayer(self.data_hub, num_R_streams, R_stream_capacity, use_motion=True)

        # populate
        for i in range(self.num_cams):
            self.cam_img_renderers[i] = CamImageRenderer(i, self.data_hub)
            self.mesh_renderers[i] =    PoseMeshRenderer(i, self.data_hub,  PoseDataHubTypes.pose_I, 10.0, None)
            self.mesh_renderers_raw[i]= PoseMeshRenderer(i, self.data_hub,  PoseDataHubTypes.pose_R, 10.0, (1.0, 1.0, 1.0, 1.0))
            self.cam_bbox_renderers[i]= CamBBoxRenderer(i, self.data_hub,   PoseDataHubTypes.pose_I)
            self.motion_time_renderers[i] = PoseMotionTimeRenderer(i, self.data_hub, PoseDataHubTypes.pose_I)

            self.cam_track_layers[i] =  CamCompositeLayer(i, self.data_hub, PoseDataHubTypes.pose_R, self.cam_img_renderers[i], 2, None, (1.0, 1.0, 1.0, 0.5))
            self.pose_cam_layers[i] =   PoseCamLayer(i, self.data_hub,      PoseDataHubTypes.pose_I, self.cam_img_renderers[i])
            self.centre_cam_layers[i] = CentreCamLayer(i, self.data_hub,    PoseDataHubTypes.pose_I, self.cam_img_renderers[i])
            self.sim_blend_layers[i] =  SimilarityBlend(i, self.data_hub,   PoseDataHubTypes.pose_I, self.centre_cam_layers)
            self.pd_line_layers[i] =    PDLineLayer(i, self.data_hub)
            self.field_bar_layers[i] =  PoseScalarBarLayer(i, self.data_hub,PoseDataHubTypes.pose_I, FrameField.angles, 2.0, 2.0)
            self.field_bar_layers_raw[i]= PoseScalarBarLayer(i, self.data_hub,PoseDataHubTypes.pose_R, FrameField.angles, 4.0, 16.0, (0.0, 0.0, 0.0, 0.33))
            self.angle_bar_layers[i] =  PoseAngleDeltaBarLayer(i, self.data_hub, PoseDataHubTypes.pose_I)
            self.motion_bar_layers[i] = PoseMotionBarLayer(i, self.data_hub, PoseDataHubTypes.pose_I, FrameField.angle_motion, 2.0, 2.0)
            # self.line_field_layers[i] = LineFieldLayer(self.render_data_old, self.cam_fbos, i)

        # composition
        self.subdivision_rows: list[SubdivisionRow] = [
            SubdivisionRow(name=CamCompositeLayer.__name__,   columns=self.num_cams,    rows=1, src_aspect_ratio=1.0,  padding=Point2f(1.0, 1.0)),
            SubdivisionRow(name=CentreCamLayer.__name__,      columns=self.num_players, rows=1, src_aspect_ratio=9/16, padding=Point2f(1.0, 1.0)),
            SubdivisionRow(name=SimilarityLineLayer.__name__, columns=1,                rows=1, src_aspect_ratio=6.0,  padding=Point2f(1.0, 1.0))
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
        for i in range(self.num_cams):
            self.cam_img_renderers[i].allocate()
            self.mesh_renderers_raw[i].allocate()
            self.mesh_renderers[i].allocate()

            self.pose_cam_layers[i].allocate(1080, 1920, GL_RGBA32F)
            self.centre_cam_layers[i].allocate(1080, 1920, GL_RGBA32F)
            self.sim_blend_layers[i].allocate(1080, 1920, GL_RGBA32F)
            self.field_bar_layers_raw[i].allocate(1080, 1920, GL_RGBA32F)
            self.field_bar_layers[i].allocate(1080, 1920, GL_RGBA32F)
            self.angle_bar_layers[i].allocate(1080, 1920, GL_RGBA32F)
            self.motion_bar_layers[i].allocate(1080, 1920, GL_RGBA32F)
            # self.line_field_layers[i].allocate(2160, 3840, GL_RGBA32F)

        self.allocate_window_renders()
        # self.sound_osc.start()

    def allocate_window_renders(self) -> None:
        w, h = self.subdivision.get_allocation_size(SimilarityLineLayer.__name__, 0)
        self.pose_sim_layer.allocate(w, h, GL_RGBA)
        # self.motion_corr_stream_layer.allocate(w, h, GL_RGBA)

        for i in range(self.num_cams):
            w, h = self.subdivision.get_allocation_size(CamCompositeLayer.__name__, i)
            self.cam_track_layers[i].allocate(w , h, GL_RGBA)
            w, h = self.subdivision.get_allocation_size(CentreCamLayer.__name__, i)
            self.pd_line_layers[i].allocate(w, h, GL_RGBA)

    def deallocate(self) -> None:
        self.pose_sim_layer.deallocate()
        # self.motion_corr_stream_layer.deallocate()

        # for layer in self.line_field_layers.values():
        #     layer.deallocate()
        for layer in self.cam_track_layers.values():
            layer.deallocate()
        for layer in self.centre_cam_layers.values():
            layer.deallocate()
        for layer in self.sim_blend_layers.values():
            layer.deallocate()
        for layer in self.pose_cam_layers.values():
            layer.deallocate()
        for layer in self.pd_line_layers.values():
            layer.deallocate()
        for layer in self.field_bar_layers_raw.values():
            layer.deallocate()
        for layer in self.field_bar_layers.values():
            layer.deallocate()
        for layer in self.angle_bar_layers.values():
            layer.deallocate()
        for layer in self.motion_bar_layers.values():
            layer.deallocate()

        # renderers
        for layer in self.cam_img_renderers.values():
            layer.deallocate()
        for layer in self.mesh_renderers_raw.values():
            layer.deallocate()
        for layer in self.mesh_renderers.values():
            layer.deallocate()

        # self.sound_osc.stop()

    def draw_main(self, width: int, height: int) -> None:
        glEnable(GL_TEXTURE_2D)
        glEnable(GL_BLEND)

        self.data_hub.notify_update()

        self.pose_sim_layer.update()

        for i in range(self.num_cams):
            self.cam_img_renderers[i].update()
            self.mesh_renderers_raw[i].update()
            self.mesh_renderers[i].update()
            self.cam_bbox_renderers[i].update()
            self.motion_time_renderers[i].update()

            self.cam_track_layers[i].update()
            self.pose_cam_layers[i].update()
            self.centre_cam_layers[i].update()
            self.sim_blend_layers[i].update()
            self.pd_line_layers[i].update()
            self.field_bar_layers_raw[i].update()
            self.field_bar_layers[i].update()
            self.angle_bar_layers[i].update()
            self.motion_bar_layers[i].update()

            # self.line_field_layers[i].update()

        self.draw_composition(width, height)

    def draw_composition(self, width:int, height: int) -> None:
        self.setView(width, height)

        glClearColor(0.0, 0.0, 0.0, 1.0)
        glClear(GL_COLOR_BUFFER_BIT)

        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        self.pose_sim_layer.draw(self.subdivision.get_rect(SimilarityLineLayer.__name__, 0))
        # self.motion_corr_stream_layer.draw(self.subdivision.get_rect(CorrelationStreamLayer.__name__, 1))

        for i in range(self.num_cams):
            track_rect: Rect = self.subdivision.get_rect(CamCompositeLayer.__name__, i)
            self.cam_track_layers[i].draw(track_rect)
            self.cam_bbox_renderers[i].draw(track_rect)


        for i in range(self.num_cams):
            preview_rect: Rect = self.subdivision.get_rect(CentreCamLayer.__name__, i)
            # self.pose_cam_layers[i].draw(preview_rect)
            # self.centre_cam_layers[i].draw(preview_rect)
            self.sim_blend_layers[i].draw(preview_rect)

        for i in range(self.num_cams):

            preview_rect: Rect = self.subdivision.get_rect(CentreCamLayer.__name__, i)
            screen_center_rect: Rect = self.centre_cam_layers[i].screen_center_rect
            draw_mesh_rect: Rect = screen_center_rect.affine_transform(preview_rect)
            self.mesh_renderers[i].draw(draw_mesh_rect)
            self.field_bar_layers[i].draw(preview_rect)
            self.pd_line_layers[i].draw(preview_rect)
            self.motion_time_renderers[i].draw(preview_rect)

            # self.line_field_layers[i].draw(self.subdivision.get_rect(PoseStreamLayer.__name__, i))



            self.field_bar_layers[i].feature_type = FrameField.angle_motion
            self.field_bar_layers_raw[i].data_type = PoseDataHubTypes.pose_I
            self.field_bar_layers_raw[i].feature_type = FrameField.similarity


            self.centre_cam_layers[i].data_type = PoseDataHubTypes.pose_I
            self.mesh_renderers[i].data_type = PoseDataHubTypes.pose_I
            self.mesh_renderers_raw[i].color = (0.66, 0.66, 0.66, 0.66)

    def draw_secondary(self, monitor_id: int, width: int, height: int) -> None:
        # return
        glEnable(GL_TEXTURE_2D)
        glEnable(GL_BLEND)
        self.setView(width, height)

        glClearColor(0.0, 0.0, 0.0, 1.0)
        glClear(GL_COLOR_BUFFER_BIT)

        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        camera_id: int = self.secondary_order_list.index(monitor_id)
        draw_rect = Rect(0, 0, width, height)

        draw_raw: bool = False

        if draw_raw:
            self.centre_cam_layers[camera_id].blend_factor = 1.0
            self.centre_cam_layers[camera_id].draw(draw_rect)
            screen_center_rect: Rect = self.centre_cam_layers[camera_id].screen_center_rect
            draw_mesh_rect: Rect = screen_center_rect.affine_transform(draw_rect)
            self.mesh_renderers_raw[camera_id].draw(draw_mesh_rect)
            self.field_bar_layers_raw[camera_id].draw(draw_rect)
            return

        draw_centre: bool = True
        if draw_centre:
            self.centre_cam_layers[camera_id].blend_factor = 0.25
            # self.centre_cam_layers[camera_id].draw(draw_rect)

            self.sim_blend_layers[camera_id].draw(draw_rect)
            screen_center_rect: Rect = self.centre_cam_layers[camera_id].screen_center_rect
            draw_mesh_rect: Rect = screen_center_rect.affine_transform(draw_rect)
            self.mesh_renderers_raw[camera_id].draw(draw_mesh_rect)
            self.mesh_renderers[camera_id].draw(draw_mesh_rect)
        else:
            self.pose_cam_layers[camera_id].draw(draw_rect)
            self.mesh_renderers_raw[camera_id].draw(draw_rect)
            self.mesh_renderers[camera_id].draw(draw_rect)

        # self.field_bar_layers_raw[camera_id].draw(draw_rect)
        # self.field_bar_layers[camera_id].draw(draw_rect)
        self.motion_bar_layers[camera_id].draw(draw_rect)
        self.angle_bar_layers[camera_id].draw(draw_rect)
        self.motion_bar_layers[camera_id].line_smooth = 10.0

        # self.pd_line_layers[camera_id].draw(draw_rect)
        # self.line_field_layers[camera_id].draw(Rect(0, 0, width, height))
