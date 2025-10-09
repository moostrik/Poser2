# Standard library imports
from enum import Enum

# Third-party imports
from OpenGL.GL import * # type: ignore

# Local application imports
from modules.gl.WindowManager import WindowManager
from modules.gl.RenderBase import RenderBase
from modules.gl.Fbo import Fbo, SwapFbo

from modules.Settings import Settings

from modules.render.DataManager import DataManager
from modules.render.Subdivision import make_subdivision, SubdivisionRow, Subdivision
from modules.render.meshes.PoseMeshes import PoseMeshes
from modules.render.meshes.AngleMeshes import AngleMeshes

from modules.render.renders.HDT.CentreCameraRender import CentreCameraRender
from modules.render.renders.HDT.CentrePoseRender import CentrePoseRender
from modules.render.renders.HDT.CamOverlayRender import CamOverlayRender
from modules.render.renders.HDT.MovementCamRender import MovementCamRender
from modules.render.renders.HDT.SynchronyCam import SynchronyCam
from modules.render.renders.CameraRender import CameraRender
from modules.render.renders.RStreamRender import RStreamRender
from modules.render.renders.HDT.LineFields import LF

from modules.pose.smooth.PoseSmoothData import PoseSmoothData, OneEuroSettings, PoseSmoothRectSettings

from modules.render.HDTSoundOSC import HDTSoundOSC

from modules.utils.PointsAndRects import Rect, Point2f
from modules.utils.HotReloadMethods import HotReloadMethods

class RenderHDT(RenderBase):
    def __init__(self, settings: Settings) -> None:

        self.num_players: int =     settings.num_players
        self.num_cams: int =        settings.camera_num
        self.num_R_streams: int =   settings.render_R_num
        self.ws_width: int =        settings.light_resolution

        self.OneEuroSettings: OneEuroSettings = OneEuroSettings(25, 1.0, 0.1)
        self.PoseSmoothRectSettings: PoseSmoothRectSettings = PoseSmoothRectSettings(
            smooth_settings=self.OneEuroSettings,
            nose_dest_x=0.5,
            nose_dest_y=0.2,
            height_dest=0.95,
            src_aspectratio=16/9,
            dst_aspectratio=9/16
        )
        self.smooth_data: PoseSmoothData = PoseSmoothData(self.num_players, self.OneEuroSettings, self.PoseSmoothRectSettings)
        self.sound_osc: HDTSoundOSC = HDTSoundOSC(self.smooth_data, "10.0.0.81", 8000, 60.0)

        self.data: DataManager =    DataManager(self.smooth_data)

        # meshes
        self.pose_meshes =          PoseMeshes(self.data, self.num_players)

        # drawers
        self.camera_renders:        dict[int, CameraRender] = {}
        self.centre_cam_renders:    dict[int, CentreCameraRender] = {}
        self.centre_pose_renders:   dict[int, CentrePoseRender] = {}
        self.movement_cam_renders:  dict[int, MovementCamRender] = {}
        self.line_field_renders:    dict[int, LF] = {}

        self.sync_renders:          dict[int, SynchronyCam] = {}
        self.overlay_renders:       dict[int, CamOverlayRender] = {}
        self.r_stream_render =      RStreamRender(self.data, self.num_R_streams)

        self.cam_fbos: list[Fbo] = []

        for i in range(self.num_cams):
            self.camera_renders[i] = CameraRender(self.data, self.pose_meshes, i)
            self.centre_cam_renders[i] = CentreCameraRender(self.data, self.smooth_data, i)
            self.centre_pose_renders[i] = CentrePoseRender(self.data, self.smooth_data, self.pose_meshes, i)
            self.movement_cam_renders[i] = MovementCamRender(self.data, i)
            self.line_field_renders[i] = LF(self.smooth_data, i)
            self.overlay_renders[i] = CamOverlayRender(self.data, self.pose_meshes, i)
            self.sync_renders[i] = SynchronyCam(self.data, i)
            self.cam_fbos.append(self.movement_cam_renders[i].get_fbo())

        # composition
        self.subdivision_rows: list[SubdivisionRow] = [
            SubdivisionRow(name=CameraRender.key(),         columns=self.num_cams,      rows=1, src_aspect_ratio=1.0,  padding=Point2f(1.0, 1.0)),
            SubdivisionRow(name=CamOverlayRender.key(),     columns=self.num_players,   rows=1, src_aspect_ratio=9/16,  padding=Point2f(1.0, 1.0)),
            SubdivisionRow(name=RStreamRender.key(),        columns=1,                  rows=1, src_aspect_ratio=12.0,  padding=Point2f(0.0, 1.0))
        ]
        self.subdivision: Subdivision = make_subdivision(self.subdivision_rows, settings.render_width, settings.render_height, False)

        # window manager
        self.secondary_order_list: list[int] = settings.render_secondary_list
        self.window_manager: WindowManager = WindowManager(
            self, self.subdivision.width, self.subdivision.height,
            settings.render_title, settings.render_fullscreen,
            settings.render_v_sync, settings.render_fps,
            settings.render_x, settings.render_y,
            # settings.render_monitor, sorted(settings.render_secondary_list)
            settings.render_monitor, settings.render_secondary_list
        )

        # hot reloader
        self.hot_reloader = HotReloadMethods(self.__class__, True, True)

    def allocate(self) -> None:
        version = glGetString(GL_VERSION)
        if isinstance(version, bytes):
            opengl_version: str = version.decode("utf-8")
            print("OpenGL version:", opengl_version)
        else:
            raise RuntimeError("OpenGL context is not valid")

        glEnable(GL_TEXTURE_2D)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        self.pose_meshes.allocate()
        for key in self.centre_cam_renders.keys():
            self.centre_cam_renders[key].allocate(2160, 3840, GL_RGBA32F)
            self.centre_pose_renders[key].allocate(2160, 3840, GL_RGBA32F)
            self.movement_cam_renders[key].allocate(2160, 3840, GL_RGBA32F) #??
            self.line_field_renders[key].allocate(2160, 3840, GL_RGBA32F)
            self.sync_renders[key].allocate(2160, 3840, GL_RGBA32F)

        self.allocate_window_renders()
        self.sound_osc.start()

    def allocate_window_renders(self) -> None:
        w, h = self.subdivision.get_allocation_size(RStreamRender.key())
        self.r_stream_render.allocate(w, h, GL_RGBA)

        for i in range(self.num_cams):
            w, h = self.subdivision.get_allocation_size(CameraRender.key(), i)
            self.camera_renders[i].allocate(w , h, GL_RGBA)
            w, h = self.subdivision.get_allocation_size(CamOverlayRender.key(), i)
            self.overlay_renders[i].allocate(w, h, GL_RGBA)

    def deallocate(self) -> None:
        self.r_stream_render.deallocate()
        for draw in self.camera_renders.values():
            draw.deallocate()
        for draw in self.centre_cam_renders.values():
            draw.deallocate()
        for draw in self.centre_pose_renders.values():
            draw.deallocate()
        for draw in self.movement_cam_renders.values():
            draw.deallocate()
        for draw in self.line_field_renders.values():
            draw.deallocate()
        for draw in self.sync_renders.values():
            draw.deallocate()
        for draw in self.overlay_renders.values():
            draw.deallocate()

        self.pose_meshes.deallocate()
        self.sound_osc.stop()

    def draw_main(self, width: int, height: int) -> None:

        glEnable(GL_TEXTURE_2D)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        
        self.OneEuroSettings.min_cutoff = 0.2
        self.OneEuroSettings.beta = 0.5

        self.PoseSmoothRectSettings.nose_dest_y = 0.25
        self.PoseSmoothRectSettings.height_dest = 0.8


        self.pose_meshes.update()

        self.r_stream_render.update()
        for i in range(self.num_cams):
            self.camera_renders[i].update()
            self.overlay_renders[i].update()
            self.centre_cam_renders[i].update()
            self.centre_pose_renders[i].update()
            self.movement_cam_renders[i].update(self.centre_cam_renders[i].get_fbo())
            self.line_field_renders[i].update()
            self.sync_renders[i].update(self.cam_fbos, self.movement_cam_renders[i].movement_for_synchrony)

        self.draw_composition(width, height)

    def draw_composition(self, width:int, height: int) -> None:
        
        glEnable(GL_TEXTURE_2D)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        
        self.setView(width, height)

        glClearColor(0.0, 0.0, 0.0, 1.0)
        glClear(GL_COLOR_BUFFER_BIT)

        self.r_stream_render.draw(self.subdivision.get_rect(RStreamRender.key()))
        for i in range(self.num_cams):
            self.camera_renders[i].draw(self.subdivision.get_rect(CameraRender.key(), i))

            # ADDITIVE
            glBlendFunc(GL_ONE, GL_ONE)
            self.centre_cam_renders[i].draw(self.subdivision.get_rect(CamOverlayRender.key(), i))
            self.overlay_renders[i].draw(self.subdivision.get_rect(CamOverlayRender.key(), i))
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    def draw_secondary(self, monitor_id: int, width: int, height: int) -> None:
        

        glEnable(GL_TEXTURE_2D)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        
        self.setView(width, height)
        glEnable(GL_TEXTURE_2D)

        glClearColor(0.0, 0.0, 0.0, 1.0)
        glClear(GL_COLOR_BUFFER_BIT)
        glEnable(GL_BLEND)
        glBlendFunc(GL_ONE, GL_ONE)

        camera_id: int = self.secondary_order_list.index(monitor_id)

        # self.sync_renders[camera_id].draw(Rect(0, 0, width, height))
        self.centre_cam_renders[camera_id].draw(Rect(0, 0, width, height))
        self.centre_pose_renders[camera_id].draw(Rect(0, 0, width, height))
        self.line_field_renders[camera_id].draw(Rect(0, 0, width, height))
        # self.overlay_renders[camera_id].draw(Rect(0, 0, width, height))
        # self.movement_cam_renders[camera_id].draw(Rect(0, 0, width, height))
        
        # self.camera_renders[camera_id].draw(Rect(0, 0, width, height))

    def on_main_window_resize(self, width: int, height: int) -> None:
        self.subdivision = make_subdivision(self.subdivision_rows, width, height, True)
        self.allocate_window_renders()