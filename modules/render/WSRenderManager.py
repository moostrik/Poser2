# Standard library imports
from enum import Enum

# Third-party imports
from OpenGL.GL import * # type: ignore

# Local application imports
from modules.gl.WindowManager import WindowManager
from modules.gl.RenderBase import RenderBase

from modules.Settings import Settings

from modules.DataHub import DataHub
from modules.render.CompositionSubdivider import make_subdivision, SubdivisionRow, Subdivision
from modules.render.meshes.PoseMesh import PoseMesh
from modules.render.meshes.AngleMeshes import AngleMeshes

from modules.render.layers.Depricated.CamTrackPoseLayer import CamTrackPoseLayer
from modules.render.layers.Depricated.PoseLayer import PoseLayer
from modules.render.layers.Generic.SimilarityLineLayer import SimilarityLineLayer
from modules.render.layers.WS.TrackerPanoramicLayer import TrackerPanoramicLayer as TrackerLayer
from modules.render.layers.WS.WSLightLayer import WSLightLayer
from modules.render.layers.WS.WSLinesLayer import WSLinesLayer

from modules.utils.PointsAndRects import Rect, Point2f
from modules.utils.HotReloadMethods import HotReloadMethods

class WSRenderManager(RenderBase):
    def __init__(self, settings: Settings) -> None:
        self.data: DataHub =    DataHub()
        self.data_key: str = self.data.get_unique_consumer_key()

        self.max_players: int =     settings.num_players
        self.num_cams: int =        settings.camera_num
        self.num_R_streams: int =   settings.render_R_num
        self.ws_width: int =        settings.light_resolution

        # meshes
        self.pose_meshes =          PoseMesh(self.data, self.max_players)
        self.angle_meshes =         AngleMeshes(self.data, self.max_players)

        # drawers
        self.ws_light_render =      WSLightLayer(self.data)
        self.ws_lines_render =      WSLinesLayer(self.data)
        self.r_stream_render =      SimilarityLineLayer(self.num_R_streams)
        self.tracker_render =       TrackerLayer(self.data, self.num_cams)
        self.camera_renders:        dict[int, CamTrackPoseLayer] = {}
        self.pose_renders:          dict[int, PoseLayer] = {}

        for i in range(self.num_cams):
            self.camera_renders[i] = CamTrackPoseLayer(self.data, self.pose_meshes, i)
        for i in range(self.max_players):
            self.pose_renders[i] = PoseLayer(self.data, self.pose_meshes, self.angle_meshes, i)

        # composition
        self.subdivision_rows: list[SubdivisionRow] = [
            SubdivisionRow(name=CamTrackPoseLayer.__name__,   columns=self.num_cams,      rows=1, src_aspect_ratio=16/9,  padding=Point2f(1.0, 1.0)),
            SubdivisionRow(name=TrackerLayer.__name__,  columns=1,                  rows=1, src_aspect_ratio=12.0,  padding=Point2f(0.0, 1.0)),
            SubdivisionRow(name=WSLinesLayer.__name__,  columns=1,                  rows=1, src_aspect_ratio=40.0,  padding=Point2f(0.0, 1.0)),
            SubdivisionRow(name=WSLightLayer.__name__,  columns=1,                  rows=1, src_aspect_ratio=10.0,  padding=Point2f(0.0, 1.0)),
            SubdivisionRow(name=PoseLayer.__name__,  columns=self.max_players,   rows=1, src_aspect_ratio=0.75,  padding=Point2f(1.0, 1.0)),
            SubdivisionRow(name=SimilarityLineLayer.__name__,  columns=1,                  rows=1, src_aspect_ratio=12.0,  padding=Point2f(0.0, 1.0)),
        ]
        self.subdivision: Subdivision = make_subdivision(self.subdivision_rows, settings.render_width, settings.render_height, False)

        # window manager
        self.window_manager: WindowManager = WindowManager(
            self, self.subdivision.width, self.subdivision.height,
            settings.render_title, settings.render_fullscreen,
            settings.render_v_sync, settings.render_fps,
            settings.render_x, settings.render_y,
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
        self.angle_meshes.allocate()

        self.allocate_window_renders()
        self.ws_light_render.allocate(self.ws_width, 1, GL_RGBA32F)
        self.ws_lines_render.allocate(self.ws_width, 100, GL_RGBA32F)

    def allocate_window_renders(self) -> None:
        w, h = self.subdivision.get_allocation_size(SimilarityLineLayer.__name__)
        self.r_stream_render.allocate(w, h, GL_RGBA)
        w, h = self.subdivision.get_allocation_size(TrackerLayer.__name__)
        self.tracker_render.allocate(w, h, GL_RGBA)
        for key in self.camera_renders.keys():
            w, h = self.subdivision.get_allocation_size(CamTrackPoseLayer.__name__, key)
            self.camera_renders[key].allocate(w, h, GL_RGBA)
        for key in self.pose_renders.keys():
            w, h = self.subdivision.get_allocation_size(PoseLayer.__name__, key)
            self.pose_renders[key].allocate(w, h, GL_RGBA)

    def deallocate(self) -> None:
        self.ws_light_render.deallocate()
        self.ws_lines_render.deallocate()
        self.r_stream_render.deallocate()
        self.tracker_render.deallocate()
        for draw in self.camera_renders.values():
            draw.deallocate()
        for draw in self.pose_renders.values():
            draw.deallocate()

        self.pose_meshes.deallocate()
        self.angle_meshes.deallocate()

    def draw_main(self, width: int, height: int) -> None:
        self.pose_meshes.update()
        # self.angle_meshes.update(False)

        self.ws_light_render.update()
        self.ws_lines_render.update()
        self.r_stream_render.update(self.data.get_correlation_streams(True, self.data_key))
        self.tracker_render.update()
        for i in range(self.num_cams):
            self.camera_renders[i].update()
        for i in range(self.max_players):
            self.pose_renders[i].update()

        self.draw_composition(width, height)

    def draw_composition(self, width: int, height: int) -> None:
        self.setView(width, height)

        self.ws_light_render.draw(self.subdivision.get_rect(WSLightLayer.__name__))
        self.ws_lines_render.draw(self.subdivision.get_rect(WSLinesLayer.__name__))
        self.tracker_render.draw(self.subdivision.get_rect(TrackerLayer.__name__))
        self.r_stream_render.draw(self.subdivision.get_rect(SimilarityLineLayer.__name__))
        for i in range(self.num_cams):
            self.camera_renders[i].draw(self.subdivision.get_rect(CamTrackPoseLayer.__name__, i))
        for i in range(self.max_players):
            self.pose_renders[i].draw(self.subdivision.get_rect(PoseLayer.__name__, i))

    def draw_secondary(self, monitor_id: int, width: int, height: int) -> None:
        self.setView(width, height)
        glEnable(GL_TEXTURE_2D)
        line_height: int = 256
        helper_width: int = int(192 * 2.5)
        helper_height: int = int(256 * 2.5)
        self.ws_lines_render.draw(Rect(0, 0, width * 4, line_height))
        self.ws_light_render.draw(Rect(0, line_height, width * 4, height - line_height))
        self.pose_renders[0].draw(Rect(0, height-helper_height, helper_width, helper_height))


    def on_main_window_resize(self, width: int, height: int) -> None:
        self.subdivision = make_subdivision(self.subdivision_rows, width, height, True)
        self.allocate_window_renders()


