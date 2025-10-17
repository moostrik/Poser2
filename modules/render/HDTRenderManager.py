# Standard library imports

# Third-party imports
from OpenGL.GL import * # type: ignore

# Local application imports
from modules.gl.Fbo import Fbo, SwapFbo
from modules.gl.RenderBase import RenderBase
from modules.gl.WindowManager import WindowManager
from modules.Settings import Settings
from modules.pose.smooth.PoseSmoothDataManager import PoseSmoothDataManager
from modules.utils.PointsAndRects import Rect, Point2f

from modules.render.CompositionSubdivider import make_subdivision, SubdivisionRow, Subdivision
from modules.render.DataManager import DataManager
from modules.render.HDTSoundOSC import HDTSoundOSC

from modules.render.layers.Generic.CamTrackPoseLayer import CamTrackPoseLayer
from modules.render.layers.Generic.PoseStreamLayer import PoseStreamLayer
from modules.render.layers.Generic.RStreamLayer import RStreamLayer
from modules.render.layers.HDT.CentreCamLayer import CentreCamLayer
from modules.render.layers.HDT.CentrePoseRender import CentrePoseRender
from modules.render.layers.HDT.LineFieldsLayer import LF as LineFieldLayer
from modules.render.meshes.PoseMeshes import PoseMeshes

from modules.utils.HotReloadMethods import HotReloadMethods


class HDTRenderManager(RenderBase):
    def __init__(self, settings: Settings) -> None:
        self.num_players: int =     settings.num_players
        self.num_cams: int =        settings.camera_num
        self.num_R_streams: int =   settings.render_R_num

        # data
        self.smooth_data: PoseSmoothDataManager = PoseSmoothDataManager(self.num_players)
        self.data: DataManager =    DataManager(self.smooth_data)
        self.sound_osc: HDTSoundOSC = HDTSoundOSC(self.smooth_data, "10.0.0.65", 8000, 60.0)

        # meshes
        self.pose_meshes =          PoseMeshes(self.data, self.num_players)

        # layers
        self.camera_layers:         dict[int, CamTrackPoseLayer] = {}
        self.centre_cam_layers:     dict[int, CentreCamLayer] = {}
        self.centre_pose_layers:    dict[int, CentrePoseRender] = {}
        self.pose_overlays:         dict[int, PoseStreamLayer] = {}
        self.line_field_layers:     dict[int, LineFieldLayer] = {}
        self.r_stream_layer =       RStreamLayer(self.data, self.num_R_streams)

        # fbos
        self.cam_fbos: dict[int, Fbo] = {}

        # populate
        for i in range(self.num_cams):
            self.camera_layers[i] = CamTrackPoseLayer(self.data, self.pose_meshes, i)
            self.centre_cam_layers[i] = CentreCamLayer(self.data, self.smooth_data, i)
            self.centre_pose_layers[i] = CentrePoseRender(self.data, self.smooth_data, self.pose_meshes, i)
            self.pose_overlays[i] = PoseStreamLayer(self.data, self.pose_meshes, i)
            self.line_field_layers[i] = LineFieldLayer(self.smooth_data, self.cam_fbos, i)
            self.cam_fbos[i] = self.centre_cam_layers[i].get_fbo()
            # self.cam_fbos[i] = self.camera_layers[i].fbo
        # composition
        self.subdivision_rows: list[SubdivisionRow] = [
            SubdivisionRow(name=CamTrackPoseLayer.__name__,       columns=self.num_cams,      rows=1, src_aspect_ratio=1.0,   padding=Point2f(1.0, 1.0)),
            SubdivisionRow(name=PoseStreamLayer.__name__,   columns=self.num_players,   rows=1, src_aspect_ratio=9/16,  padding=Point2f(1.0, 1.0)),
            SubdivisionRow(name=RStreamLayer.__name__,      columns=1,                  rows=1, src_aspect_ratio=12.0,  padding=Point2f(0.0, 1.0))
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
            self.centre_cam_layers[i].allocate(2160, 3840, GL_RGBA32F)
            self.centre_pose_layers[i].allocate(2160, 3840, GL_RGBA32F)
            self.line_field_layers[i].allocate(2160, 3840, GL_RGBA32F)

        self.pose_meshes.allocate()

        self.allocate_window_renders()
        self.sound_osc.start()

    def allocate_window_renders(self) -> None:
        w, h = self.subdivision.get_allocation_size(RStreamLayer.__name__)
        self.r_stream_layer.allocate(w, h, GL_RGBA)

        for i in range(self.num_cams):
            w, h = self.subdivision.get_allocation_size(CamTrackPoseLayer.__name__, i)
            self.camera_layers[i].allocate(w , h, GL_RGBA)
            w, h = self.subdivision.get_allocation_size(PoseStreamLayer.__name__, i)
            self.pose_overlays[i].allocate(w, h, GL_RGBA)

    def deallocate(self) -> None:
        self.pose_meshes.deallocate()
        self.r_stream_layer.deallocate()
        for draw in self.camera_layers.values():
            draw.deallocate()
        for draw in self.centre_cam_layers.values():
            draw.deallocate()
        for draw in self.centre_pose_layers.values():
            draw.deallocate()
        for draw in self.pose_overlays.values():
            draw.deallocate()
        for draw in self.line_field_layers.values():
            draw.deallocate()

        self.sound_osc.stop()

    def draw_main(self, width: int, height: int) -> None:
        glEnable(GL_TEXTURE_2D)
        glEnable(GL_BLEND)

        self.smooth_data.update()
        self.pose_meshes.update()
        self.r_stream_layer.update()

        for i in range(self.num_cams):
            self.camera_layers[i].update()
            self.centre_cam_layers[i].update()
            self.centre_pose_layers[i].update()
            self.pose_overlays[i].update()
            self.line_field_layers[i].update()

        self.draw_composition(width, height)

    def draw_composition(self, width:int, height: int) -> None:
        self.setView(width, height)

        glClearColor(0.0, 0.0, 0.0, 1.0)
        glClear(GL_COLOR_BUFFER_BIT)

        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        self.r_stream_layer.draw(self.subdivision.get_rect(RStreamLayer.__name__))
        for i in range(self.num_cams):
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            self.camera_layers[i].draw(self.subdivision.get_rect(CamTrackPoseLayer.__name__, i))

            glBlendFunc(GL_ONE, GL_ONE)
            self.centre_cam_layers[i].draw(self.subdivision.get_rect(PoseStreamLayer.__name__, i))
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
        # self.centre_cam_layers[camera_id].draw(Rect(0, 0, width, height))
        self.line_field_layers[camera_id].draw(Rect(0, 0, width, height))

        # if self.smooth_data.get_is_active(camera_id):
        #     glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        #     glBlendEquation(GL_FUNC_REVERSE_SUBTRACT)
        #     self.centre_pose_layers[camera_id].draw(Rect(0, 0, width, height))
        #     glBlendEquation(GL_FUNC_ADD)
        #     glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
