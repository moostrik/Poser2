# Standard library imports
from time import perf_counter

# Third-party imports
from OpenGL.GL import * # type: ignore

# Local application imports
from modules.gl.Fbo import Fbo, SwapFbo
from modules.gl.RenderBase import RenderBase
from modules.gl.WindowManager import WindowManager
from modules.Settings import Settings
from modules.utils.PointsAndRects import Rect, Point2f

from modules.gui.PyReallySimpleGui import Gui
from modules.CaptureDataHub import CaptureDataHub
from modules.RenderDataHub_Old import RenderDataHub_Old
from modules.render.HDTSoundOSC import HDTSoundOSC

from modules.render.CompositionSubdivider import make_subdivision, SubdivisionRow, Subdivision
from modules.render.layers.Generic.CamTrackPoseLayer import CamTrackPoseLayer
from modules.render.layers.Generic.PoseStreamLayer import PoseStreamLayer
from modules.render.layers.Generic.CorrelationStreamLayer import CorrelationStreamLayer
from modules.render.layers.Generic.PoseFeatureLayer import PoseFeatureLayer
from modules.render.layers.HDT.CentreCamLayer import CentreCamLayer
from modules.render.layers.HDT.CentrePoseRender import CentrePoseRender
from modules.render.layers.HDT.LineFieldsLayer import LF as LineFieldLayer
from modules.render.meshes.PoseMeshes import PoseMeshes

from modules.utils.HotReloadMethods import HotReloadMethods


class HDTRenderManager(RenderBase):
    def __init__(self, gui: Gui, capture_data_hub: CaptureDataHub, render_data_hub: RenderDataHub_Old, settings: Settings) -> None:
        self.num_players: int =     settings.num_players
        self.num_cams: int =        settings.camera_num
        num_R_streams: int =   settings.render_R_num
        R_stream_capacity: int = int(settings.camera_fps * 10)  # 10 seconds buffer

        # data
        self.render_data: RenderDataHub_Old =   render_data_hub
        self.capture_data: CaptureDataHub = capture_data_hub
        self.sound_osc: HDTSoundOSC =       HDTSoundOSC(self.render_data, "localhost", 8000, 60.0)

        # meshes
        self.pose_meshes =          PoseMeshes(self.capture_data, self.num_players)

        # layers
        self.camera_layers:         dict[int, CamTrackPoseLayer] = {}
        self.centre_cam_layers:     dict[int, CentreCamLayer] = {}
        self.centre_pose_layers:    dict[int, CentrePoseRender] = {}
        self.pose_overlays:         dict[int, PoseStreamLayer] = {}
        self.pose_feature_layers:   dict[int, PoseFeatureLayer] = {}
        self.line_field_layers:     dict[int, LineFieldLayer] = {}
        self.pose_corr_stream_layer =   CorrelationStreamLayer(self.capture_data, num_R_streams, R_stream_capacity, use_motion=False)
        self.motion_corr_stream_layer = CorrelationStreamLayer(self.capture_data, num_R_streams, R_stream_capacity, use_motion=True)

        # fbos
        self.cam_fbos: dict[int, Fbo] = {}

        # populate
        for i in range(self.num_cams):
            self.camera_layers[i] = CamTrackPoseLayer(self.capture_data, self.pose_meshes, i)
            self.centre_cam_layers[i] = CentreCamLayer(self.capture_data, self.render_data, i)
            self.centre_pose_layers[i] = CentrePoseRender(self.capture_data, self.render_data, self.pose_meshes, i)
            self.pose_overlays[i] = PoseStreamLayer(self.capture_data, self.pose_meshes, i)
            self.pose_feature_layers[i] = PoseFeatureLayer(self.render_data, self.capture_data, i)
            self.line_field_layers[i] = LineFieldLayer(self.render_data, self.cam_fbos, i)
            self.cam_fbos[i] = self.centre_cam_layers[i].get_fbo()
            # self.cam_fbos[i] = self.camera_layers[i].fbo
        # composition
        self.subdivision_rows: list[SubdivisionRow] = [
            SubdivisionRow(name=CamTrackPoseLayer.__name__,      columns=self.num_cams,    rows=1, src_aspect_ratio=1.0,  padding=Point2f(1.0, 1.0)),
            SubdivisionRow(name=PoseStreamLayer.__name__,        columns=self.num_players, rows=1, src_aspect_ratio=9/16, padding=Point2f(1.0, 1.0)),
            SubdivisionRow(name=CorrelationStreamLayer.__name__, columns=2,                rows=1, src_aspect_ratio=6.0,  padding=Point2f(1.0, 1.0))
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
            self.centre_cam_layers[i].allocate(1080, 1920, GL_RGBA32F)
            self.centre_pose_layers[i].allocate(1080, 1920, GL_RGBA32F)
            self.pose_feature_layers[i].allocate(1080, 1920, GL_RGBA32F)
            self.line_field_layers[i].allocate(2160, 3840, GL_RGBA32F)

        self.pose_meshes.allocate()

        self.allocate_window_renders()
        self.sound_osc.start()

    def allocate_window_renders(self) -> None:
        w, h = self.subdivision.get_allocation_size(CorrelationStreamLayer.__name__, 0)
        self.motion_corr_stream_layer.allocate(w, h, GL_RGBA)
        self.pose_corr_stream_layer.allocate(w, h, GL_RGBA)

        for i in range(self.num_cams):
            w, h = self.subdivision.get_allocation_size(CamTrackPoseLayer.__name__, i)
            self.camera_layers[i].allocate(w , h, GL_RGBA)
            w, h = self.subdivision.get_allocation_size(PoseStreamLayer.__name__, i)
            self.pose_overlays[i].allocate(w, h, GL_RGBA)

    def deallocate(self) -> None:
        self.pose_meshes.deallocate()
        self.motion_corr_stream_layer.deallocate()
        self.pose_corr_stream_layer.deallocate()
        for layer in self.camera_layers.values():
            layer.deallocate()
        for layer in self.centre_cam_layers.values():
            layer.deallocate()
        for layer in self.centre_pose_layers.values():
            layer.deallocate()
        for layer in self.pose_overlays.values():
            layer.deallocate()
        for layer in self.line_field_layers.values():
            layer.deallocate()
        for layer in self.pose_feature_layers.values():
            layer.deallocate()

        self.sound_osc.stop()

    def draw_main(self, width: int, height: int) -> None:
        glEnable(GL_TEXTURE_2D)
        glEnable(GL_BLEND)

        self.render_data.update()
        self.pose_meshes.update()

        self.motion_corr_stream_layer.update()
        self.pose_corr_stream_layer.update()

        for i in range(self.num_cams):
            self.camera_layers[i].update()
            self.centre_cam_layers[i].update()
            self.centre_pose_layers[i].update()
            self.pose_overlays[i].update()
            self.pose_feature_layers[i].update()
            # self.line_field_layers[i].update()

        # if (t5-t0) * 1000 > 10:
        #     print(f"t1 {(t1 - t0)*1000:4.0f} | t2 {(t2 - t1)*1000:4.0f} | t3 {(t3 - t2)*1000:4.0f} | t4 {(t4 - t3)*1000:4.0f} | t5 {(t5 - t4)*1000:4.0f} | total {(t5 - t0)*1000:4.0f} ms")

        self.draw_composition(width, height)

    def draw_composition(self, width:int, height: int) -> None:
        self.setView(width, height)

        glClearColor(0.0, 0.0, 0.0, 1.0)
        glClear(GL_COLOR_BUFFER_BIT)

        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        self.motion_corr_stream_layer.draw(self.subdivision.get_rect(CorrelationStreamLayer.__name__, 0))
        self.pose_corr_stream_layer.draw(self.subdivision.get_rect(CorrelationStreamLayer.__name__, 1))
        for i in range(self.num_cams):
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            self.camera_layers[i].draw(self.subdivision.get_rect(CamTrackPoseLayer.__name__, i))

            glBlendFunc(GL_ONE, GL_ONE)
            self.centre_cam_layers[i].draw(self.subdivision.get_rect(PoseStreamLayer.__name__, i))
            # self.line_field_layers[i].draw(self.subdivision.get_rect(PoseStreamLayer.__name__, i))
            self.centre_pose_layers[i].draw(self.subdivision.get_rect(PoseStreamLayer.__name__, i))
            self.pose_overlays[i].draw(self.subdivision.get_rect(PoseStreamLayer.__name__, i))

    def draw_secondary(self, monitor_id: int, width: int, height: int) -> None:
        glEnable(GL_TEXTURE_2D)
        glEnable(GL_BLEND)
        self.setView(width, height)

        glClearColor(0.0, 0.0, 0.0, 1.0)
        glClear(GL_COLOR_BUFFER_BIT)

        glBlendFunc(GL_ONE, GL_ONE)

        camera_id: int = self.secondary_order_list.index(monitor_id)
        self.centre_cam_layers[camera_id].draw(Rect(0, 0, width, height))
        self.pose_feature_layers[camera_id].draw(Rect(0, 0, width, height))
        # self.pose_overlays[camera_id].draw(Rect(0, 0, width, height))
        # self.line_field_layers[camera_id].draw(Rect(0, 0, width, height))

        if self.render_data.get_is_active(camera_id):
            # glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            # glBlendEquation(GL_FUNC_REVERSE_SUBTRACT)
            self.centre_pose_layers[camera_id].draw(Rect(0, 0, width, height))
        glBlendFunc(GL_ONE, GL_ONE)

