# Standard library imports
from time import perf_counter

# Third-party imports
from OpenGL.GL import * # type: ignore

# Render Imports
from modules.render.CompositionSubdivider import make_subdivision, SubdivisionRow, Subdivision
from modules.render.layers import CamCompositeLayer, PoseScalarBarLayer, SimilarityLineLayer, PDLineLayer
from modules.render.meshes import PoseMesh
# from modules.render.layers.HDT.CentreCamLayer import CentreCamLayer
# from modules.render.layers.HDT.CentrePoseRender import CentrePoseRender
# from modules.render.layers.HDT.LineFieldsLayer import LF as LineFieldLayer
# from modules.render.HDTSoundOSC import HDTSoundOSC

# Local application imports
from modules.gl.Fbo import Fbo, SwapFbo
from modules.gl.RenderBase import RenderBase
from modules.gl.WindowManager import WindowManager
from modules.Settings import Settings
from modules.utils.PointsAndRects import Rect, Point2f

from modules.gui.PyReallySimpleGui import Gui
from modules.DataHub import DataHub, DataType, PoseDataTypes, SimilarityDataType

from modules.utils.HotReloadMethods import HotReloadMethods
from modules.pose.Pose import ScalarPoseField

class HDTRenderManager(RenderBase):
    def __init__(self, gui: Gui, data_hub: DataHub, settings: Settings) -> None:
        self.num_players: int =     settings.num_players
        self.num_cams: int =        settings.camera_num
        num_R_streams: int =   settings.render_R_num
        R_stream_capacity: int = int(settings.camera_fps * 30)  # 10 seconds buffer

        # data
        self.data_hub: DataHub = data_hub
        # self.sound_osc: HDTSoundOSC =       HDTSoundOSC(self.render_data_old, "localhost", 8000, 60.0)

        # meshes
        self.pose_meshes =              PoseMesh(self.num_players, self.data_hub, PoseDataTypes.pose_R)

        # layers
        self.cam_comps:             dict[int, CamCompositeLayer] = {}
        # self.centre_cam_layers:         dict[int, CentreCamLayer] = {}
        # self.centre_pose_layers:        dict[int, CentrePoseRender] = {}
        self.pd_angle_overlay:      dict[int, PDLineLayer] = {}
        self.field_bars:            dict[int, PoseScalarBarLayer] = {}
        # self.line_field_layers:         dict[int, LineFieldLayer] = {}
        self.pose_sim_window =      SimilarityLineLayer(num_R_streams, R_stream_capacity, self.data_hub, SimilarityDataType.sim_P)
        # self.motion_corr_stream_layer = CorrelationStreamLayer(self.data_hub, num_R_streams, R_stream_capacity, use_motion=True)

        # fbos
        self.cam_fbos: dict[int, Fbo] = {}

        # populate
        for i in range(self.num_cams):
            self.cam_comps[i] = CamCompositeLayer(i, self.data_hub, DataType.pose_R, self.pose_meshes, (0.0, 0.0, 0.0, 0.2))
            # self.centre_cam_layers[i] = CentreCamLayer(self.data_hub, i)
            # self.centre_pose_layers[i] = CentrePoseRender(self.data_hub, self.pose_meshes, i)
            # self.centre_pose_layers_fast[i] = CentrePoseRender(self.capture_data, self.render_data_old, self.pose_meshes_fast, i)
            self.pd_angle_overlay[i] = PDLineLayer(i, self.data_hub)
            self.field_bars[i] = PoseScalarBarLayer(i, self.data_hub, DataType.pose_R, ScalarPoseField.angles)
            # self.line_field_layers[i] = LineFieldLayer(self.render_data_old, self.cam_fbos, i)
            # self.cam_fbos[i] = self.centre_cam_layers[i].get_fbo()
            self.cam_fbos[i] = self.cam_comps[i]._fbo
        # composition
        self.subdivision_rows: list[SubdivisionRow] = [
            SubdivisionRow(name=CamCompositeLayer.__name__,      columns=self.num_cams,    rows=1, src_aspect_ratio=1.0,  padding=Point2f(1.0, 1.0)),
            SubdivisionRow(name=PDLineLayer.__name__,        columns=self.num_players, rows=1, src_aspect_ratio=9/16, padding=Point2f(1.0, 1.0)),
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
            settings.render_monitor, settings.render_secondary_list # sorted(settings.render_secondary_list)
        )

        # hot reloader
        self.hot_reloader = HotReloadMethods(self.__class__, True, True)

    def on_main_window_resize(self, width: int, height: int) -> None:
        self.subdivision = make_subdivision(self.subdivision_rows, width, height, True)
        self.allocate_window_renders()

    def allocate(self) -> None:
        for i in range(self.num_cams):
            # self.centre_cam_layers[i].allocate(1080, 1920, GL_RGBA32F)
            # self.centre_pose_layers[i].allocate(1080, 1920, GL_RGBA32F)
            # self.centre_pose_layers_fast[i].allocate(1080, 1920, GL_RGBA32F)
            self.field_bars[i].allocate(1080, 1920, GL_RGBA32F)
            # self.line_field_layers[i].allocate(2160, 3840, GL_RGBA32F)

        self.pose_meshes.allocate()
        # self.pose_meshes_fast.allocate()

        self.allocate_window_renders()
        # self.sound_osc.start()

    def allocate_window_renders(self) -> None:
        w, h = self.subdivision.get_allocation_size(SimilarityLineLayer.__name__, 0)
        self.pose_sim_window.allocate(w, h, GL_RGBA)
        # self.motion_corr_stream_layer.allocate(w, h, GL_RGBA)

        for i in range(self.num_cams):
            w, h = self.subdivision.get_allocation_size(CamCompositeLayer.__name__, i)
            self.cam_comps[i].allocate(w , h, GL_RGBA)
            w, h = self.subdivision.get_allocation_size(PDLineLayer.__name__, i)
            self.pd_angle_overlay[i].allocate(w, h, GL_RGBA)

    def deallocate(self) -> None:
        self.pose_meshes.deallocate()
        self.pose_sim_window.deallocate()
        # self.motion_corr_stream_layer.deallocate()
        for layer in self.cam_comps.values():
            layer.deallocate()
        # for layer in self.centre_cam_layers.values():
        #     layer.deallocate()
        # for layer in self.centre_pose_layers.values():
        #     layer.deallocate()
        # for layer in self.centre_pose_layers_fast.values():
        #     layer.deallocate()
        for layer in self.pd_angle_overlay.values():
            layer.deallocate()
        # for layer in self.line_field_layers.values():
        #     layer.deallocate()
        for layer in self.field_bars.values():
            layer.deallocate()

        # self.sound_osc.stop()

    def draw_main(self, width: int, height: int) -> None:
        glEnable(GL_TEXTURE_2D)
        glEnable(GL_BLEND)

        self.data_hub.notify_update()
        self.pose_meshes.update()
        # self.pose_meshes_fast.update()

        self.pose_sim_window.update()
        # self.motion_corr_stream_layer.update()

        for i in range(self.num_cams):
            self.cam_comps[i].update()
            # self.centre_cam_layers[i].update()
            # self.centre_pose_layers[i].update()
            # self.centre_pose_layers_fast[i].update()
            self.pd_angle_overlay[i].update()
            self.field_bars[i].update()
            # self.line_field_layers[i].update()

        # if (t5-t0) * 1000 > 10:
        #     print(f"t1 {(t1 - t0)*1000:4.0f} | t2 {(t2 - t1)*1000:4.0f} | t3 {(t3 - t2)*1000:4.0f} | t4 {(t4 - t3)*1000:4.0f} | t5 {(t5 - t4)*1000:4.0f} | total {(t5 - t0)*1000:4.0f} ms")

        self.draw_composition(width, height)

    def draw_composition(self, width:int, height: int) -> None:
        self.setView(width, height)

        glClearColor(1.0, 1.0, 1.0, 1.0)
        glClear(GL_COLOR_BUFFER_BIT)

        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        self.pose_sim_window.draw(self.subdivision.get_rect(SimilarityLineLayer.__name__, 0))
        # self.motion_corr_stream_layer.draw(self.subdivision.get_rect(CorrelationStreamLayer.__name__, 1))
        for i in range(self.num_cams):
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            self.cam_comps[i].draw(self.subdivision.get_rect(CamCompositeLayer.__name__, i))

            # glBlendFunc(GL_ONE, GL_ONE)
            # self.centre_cam_layers[i].draw(self.subdivision.get_rect(PoseStreamLayer.__name__, i))
            # self.line_field_layers[i].draw(self.subdivision.get_rect(PoseStreamLayer.__name__, i))
            # self.centre_pose_layers[i].draw(self.subdivision.get_rect(PoseStreamLayer.__name__, i))
            self.pd_angle_overlay[i].draw(self.subdivision.get_rect(PDLineLayer.__name__, i))

            # self.field_bars[i].draw(self.subdivision.get_rect(PDLineLayer.__name__, i))

    def draw_secondary(self, monitor_id: int, width: int, height: int) -> None:
        return
        glEnable(GL_TEXTURE_2D)
        glEnable(GL_BLEND)
        self.setView(width, height)

        glClearColor(0.0, 0.0, 0.0, 1.0)
        glClear(GL_COLOR_BUFFER_BIT)

        glBlendFunc(GL_ONE, GL_ONE)

        camera_id: int = self.secondary_order_list.index(monitor_id)
        self.cam_comps[camera_id].draw(Rect(0, 0, width, height))
        # self.centre_cam_layers[camera_id].draw(Rect(0, 0, width, height))
        self.field_bars[camera_id].draw(Rect(0, 0, width, height))
        self.pd_angle_overlay[camera_id].draw(Rect(0, 0, width, height))
        # self.line_field_layers[camera_id].draw(Rect(0, 0, width, height))

        if self.data_hub.has_item(DataType.pose_I, camera_id): # camera_id is pose id
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            # glBlendEquation(GL_FUNC_REVERSE_SUBTRACT)
            # self.centre_pose_layers[camera_id].draw(Rect(0, 0, width, height))
            # self.centre_pose_layers_fast[camera_id].draw(Rect(0, 0, width, height))
        glBlendFunc(GL_ONE, GL_ONE)

