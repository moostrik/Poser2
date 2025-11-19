# Standard library imports
from time import perf_counter

# Third-party imports
from OpenGL.GL import * # type: ignore

# Render Imports
from modules.render.CompositionSubdivider import make_subdivision, SubdivisionRow, Subdivision
from modules.render.layers import CamCompositeLayer, PoseScalarBarLayer, SimilarityLineLayer, PDLineLayer, CentreCamLayer
from modules.render.renderers import CamImageRenderer, PoseMeshRenderer
# from modules.render.layers.HDT.CentreCamLayer import CentreCamLayer
# from modules.render.layers.HDT.CentrePoseRender import CentrePoseRender
# from modules.render.layers.HDT.LineFieldsLayer import LF as LineFieldLayer
# from modules.render.HDTSoundOSC import HDTSoundOSC

# Local application imports
from modules.gl.Fbo import Fbo, SwapFbo
from modules.gl.RenderBase import RenderBase
from modules.gl.WindowManager import WindowManager

from modules.pose.Pose import ScalarPoseField
from modules.Settings import Settings
from modules.utils.PointsAndRects import Rect, Point2f

from modules.gui.PyReallySimpleGui import Gui
from modules.DataHub import DataHub, DataType, PoseDataTypes, SimilarityDataType

from modules.utils.HotReloadMethods import HotReloadMethods

class HDTRenderManager(RenderBase):
    def __init__(self, gui: Gui, data_hub: DataHub, settings: Settings) -> None:
        self.num_players: int = settings.num_players
        self.num_cams: int =    settings.camera_num
        num_R_streams: int =    settings.render_R_num
        R_stream_capacity: int= int(settings.camera_fps * 30)  # 10 seconds buffer

        # data
        self.data_hub: DataHub = data_hub
        # self.sound_osc: HDTSoundOSC =       HDTSoundOSC(self.render_data_old, "localhost", 8000, 60.0)

        # renderers
        self.cam_img_renderers: dict[int, CamImageRenderer] = {}
        self.mesh_renderers_A:  dict[int, PoseMeshRenderer] = {}

        # layers
        self.cam_track_layers:  dict[int, CamCompositeLayer] = {}
        self.centre_cam_layers: dict[int, CentreCamLayer] = {}
        self.pd_line_layers:    dict[int, PDLineLayer] = {}
        self.field_bar_layers:  dict[int, PoseScalarBarLayer] = {}

        # self.line_field_layers:     dict[int, LineFieldLayer] = {}

        self.pose_sim_layer =   SimilarityLineLayer(num_R_streams, R_stream_capacity, self.data_hub, SimilarityDataType.sim_P)
        # self.motion_corr_stream_layer = CorrelationStreamLayer(self.data_hub, num_R_streams, R_stream_capacity, use_motion=True)

        # populate
        for i in range(self.num_cams):
            self.cam_img_renderers[i] = CamImageRenderer(i, self.data_hub)
            self.mesh_renderers_A[i] =  PoseMeshRenderer(i, self.data_hub, PoseDataTypes.pose_R)

            self.cam_track_layers[i] =  CamCompositeLayer(i, self.data_hub, PoseDataTypes.pose_R, self.cam_img_renderers[i], 2, None, (0.0, 0.0, 0.0, 0.5))
            self.centre_cam_layers[i] = CentreCamLayer(i, self.data_hub, PoseDataTypes.pose_R, self.cam_img_renderers[i])
            self.pd_line_layers[i] =    PDLineLayer(i, self.data_hub)
            self.field_bar_layers[i] =  PoseScalarBarLayer(i, self.data_hub, DataType.pose_R, ScalarPoseField.angles)
            # self.line_field_layers[i] = LineFieldLayer(self.render_data_old, self.cam_fbos, i)

        # composition
        self.subdivision_rows: list[SubdivisionRow] = [
            SubdivisionRow(name=CamCompositeLayer.__name__,   columns=self.num_cams,    rows=1, src_aspect_ratio=1.0,  padding=Point2f(1.0, 1.0)),
            SubdivisionRow(name=CentreCamLayer.__name__,      columns=self.num_players, rows=1, src_aspect_ratio=9/16, padding=Point2f(1.0, 1.0)),
            SubdivisionRow(name=SimilarityLineLayer.__name__, columns=1,                rows=1, src_aspect_ratio=6.0,  padding=Point2f(1.0, 1.0))
        ]
        self.subdivision: Subdivision = make_subdivision(self.subdivision_rows, settings.render_width, settings.render_height, False)

        # window manager
        self.secondary_order_list: list[int] = settings.render_secondary_list
        self.window_manager: WindowManager = WindowManager(
            self, self.subdivision.width, self.subdivision.height,
            settings.render_title, settings.render_fullscreen,
            settings.render_v_sync, settings.render_fps,
            settings.render_x, settings.render_y,
            settings.render_monitor, settings.render_secondary_list
        )

        # hot reloader
        self.hot_reloader = HotReloadMethods(self.__class__, True, True)

    def on_main_window_resize(self, width: int, height: int) -> None:
        self.subdivision = make_subdivision(self.subdivision_rows, width, height, True)
        self.allocate_window_renders()

    def allocate(self) -> None:
        for i in range(self.num_cams):
            self.cam_img_renderers[i].allocate()
            self.mesh_renderers_A[i].allocate()

            self.centre_cam_layers[i].allocate(1080, 1920, GL_RGBA32F)
            self.field_bar_layers[i].allocate(1080, 1920, GL_RGBA32F)
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
        for layer in self.pd_line_layers.values():
            layer.deallocate()
        for layer in self.field_bar_layers.values():
            layer.deallocate()

        for layer in self.mesh_renderers_A.values():
            layer.deallocate()

        # renderers
        for layer in self.cam_img_renderers.values():
            layer.deallocate()

        # self.sound_osc.stop()

    def draw_main(self, width: int, height: int) -> None:
        glEnable(GL_TEXTURE_2D)
        glEnable(GL_BLEND)

        self.data_hub.notify_update()

        self.pose_sim_layer.update()

        for i in range(self.num_cams):
            self.cam_img_renderers[i].update()
            self.mesh_renderers_A[i].update()

            self.cam_track_layers[i].update()
            self.centre_cam_layers[i].update()
            self.pd_line_layers[i].update()
            self.field_bar_layers[i].update()

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
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            self.cam_track_layers[i].draw(self.subdivision.get_rect(CamCompositeLayer.__name__, i))

            preview_rect: Rect = self.subdivision.get_rect(CentreCamLayer.__name__, i)
            self.centre_cam_layers[i].draw(preview_rect)
            self.field_bar_layers[i].draw(preview_rect, draw_labels=False)
            self.pd_line_layers[i].draw(preview_rect)

            # self.line_field_layers[i].draw(self.subdivision.get_rect(PoseStreamLayer.__name__, i))

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

        self.centre_cam_layers[camera_id].draw(draw_rect)
        self.field_bar_layers[camera_id].draw(draw_rect)
        # self.pd_line_layers[camera_id].draw(draw_rect)
        # self.line_field_layers[camera_id].draw(Rect(0, 0, width, height))


        centre_rect: Rect = self.centre_cam_layers[camera_id].centre_rect
        pose = self.data_hub.get_item(DataType(PoseDataTypes.pose_R), camera_id)
        if pose is not None:
            bbox = pose.bbox.to_rect()

            # Final transform: screen = ((v * bbox + bbox_pos) - crop_pos) / crop_size * screen_size
            # Mesh does: screen = v * w + x
            # So: w = bbox.size / crop.size * screen.size
            #     x = (bbox.pos - crop.pos) / crop.size * screen.size
            draw_mesh_rect = Rect(
                x=(bbox.x - centre_rect.x) / centre_rect.width * draw_rect.width,
                y=(bbox.y - centre_rect.y) / centre_rect.height * draw_rect.height,
                width=bbox.width / centre_rect.width * draw_rect.width,
                height=bbox.height / centre_rect.height * draw_rect.height
            )
        else:
            draw_mesh_rect = draw_rect
        self.mesh_renderers_A[camera_id].line_width = 10.0
        self.mesh_renderers_A[camera_id].draw(draw_mesh_rect)


