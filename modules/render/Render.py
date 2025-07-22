# Standard library imports
from enum import Enum
from marshal import version

# Third-party imports
from OpenGL.GL import * # type: ignore

# Local application imports
from modules.gl.WindowManager import WindowManager
from modules.gl.RenderBase import RenderBase

from modules.Settings import Settings

from modules.render.DataManager import DataManager
from modules.render.CompositionSubdivision import make_subdivision, SubdivisionRow, Subdivision
from modules.render.meshes.PoseMeshes import PoseMeshes
from modules.render.meshes.AngleMeshes import AngleMeshes

from modules.render.renders.WhiteSpaceRender import WhiteSpaceRender
from modules.render.renders.PanoramicTrackerRender import PanoramicTrackerRender
from modules.render.renders.CameraRender import CameraRender
from modules.render.renders.PoseRender import PoseRender
from modules.render.renders.RStreamRender import RStreamRender

from modules.utils.PointsAndRects import Rect, Point2f
from modules.utils.HotReloadMethods import HotReloadMethods


class SubdivisionType(Enum):
    CAM = "CAM"
    TRK = "TRK"
    RST = "RST"
    PSN = "PSN"
    WS = "WS"

class Render(RenderBase):
    def __init__(self, settings: Settings) -> None:
        self.data: DataManager =    DataManager()

        self.max_players: int =     settings.max_players
        self.num_cams: int =        settings.camera_num
        self.num_R_streams: int =   settings.render_R_num
        self.ws_width: int =        settings.light_resolution

        # meshes
        self.pose_meshes =          PoseMeshes(self.data, self.max_players)
        self.angle_meshes =         AngleMeshes(self.data, self.max_players)

        # drawers
        self.white_space_render =   WhiteSpaceRender(self.data)
        self.r_stream_render =      RStreamRender(self.data, self.num_R_streams)
        self.tracker_render =       PanoramicTrackerRender(self.data, self.num_cams)
        self.camera_renders:        dict[int, CameraRender] = {}
        self.pose_renders:          dict[int, PoseRender] = {}

        for i in range(self.num_cams):
            self.camera_renders[i] = CameraRender(self.data, self.pose_meshes, i)
        for i in range(self.max_players):
            self.pose_renders[i] = PoseRender(self.data, self.pose_meshes, self.angle_meshes, i)

        # composition
        self.subdivision_rows: list[SubdivisionRow] = [
            SubdivisionRow(name=SubdivisionType.CAM.value, columns=self.num_cams, rows=1, src_aspect_ratio=16/9, padding=Point2f(1.0, 1.0)),
            SubdivisionRow(name=SubdivisionType.TRK.value, columns=1, rows=1, src_aspect_ratio=12.0, padding=Point2f(0.0, 1.0)),
            SubdivisionRow(name=SubdivisionType.RST.value, columns=1, rows=1, src_aspect_ratio=12.0, padding=Point2f(0.0, 1.0)),
            SubdivisionRow(name=SubdivisionType.PSN.value, columns=self.max_players, rows=1, src_aspect_ratio=1.0, padding=Point2f(1.0, 1.0)),
            SubdivisionRow(name=SubdivisionType.WS.value, columns=1, rows=1, src_aspect_ratio=10.0, padding=Point2f(0.0, 1.0)),
        ]
        self.subdivision: Subdivision = make_subdivision(self.subdivision_rows, settings.render_width, settings.render_height)

        # window manager
        secondary_monitor_ids: list[int] = [i for i in range(1, self.num_cams + 1)]
        self.window_manager: WindowManager = WindowManager(
            self, self.subdivision.width, self.subdivision.height,
            settings.render_title, settings.render_fullscreen,
            settings.render_v_sync, settings.render_fps,
            settings.render_x, settings.render_y,
            settings.render_monitor, secondary_monitor_ids
        )

        # hot reloader
        self.hot_reloader = HotReloadMethods(self.__class__, True, True)

    def allocate(self) -> None:
        version = glGetString(GL_VERSION)
        if isinstance(version, bytes):
            opengl_version: str = version.decode("utf-8")
            print("OpenGL version:", opengl_version)
        else:
            raise RuntimeError("OpenGL context is not valid or not current!")

        glEnable(GL_TEXTURE_2D)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        self.pose_meshes.allocate()
        self.angle_meshes.allocate()

        self.allocate_window_renders()
        self.white_space_render.allocate(self.ws_width, 1, GL_RGBA32F)

    def allocate_window_renders(self) -> None:
        w, h = self.subdivision.get_allocation_size(SubdivisionType.RST.value)
        self.r_stream_render.allocate(w, h, GL_RGBA)
        w, h = self.subdivision.get_allocation_size(SubdivisionType.TRK.value)
        self.tracker_render.allocate(w, h, GL_RGBA)
        for key in self.camera_renders.keys():
            w, h = self.subdivision.get_allocation_size(SubdivisionType.CAM.value, key)
            self.camera_renders[key].allocate(w, h, GL_RGBA)
        for key in self.pose_renders.keys():
            w, h = self.subdivision.get_allocation_size(SubdivisionType.PSN.value, key)
            self.pose_renders[key].allocate(w, h, GL_RGBA)

    def deallocate(self) -> None:
        self.white_space_render.deallocate()
        self.r_stream_render.deallocate()
        self.tracker_render.deallocate()
        for draw in self.camera_renders.values():
            draw.deallocate()
        for draw in self.pose_renders.values():
            draw.deallocate()

        self.pose_meshes.deallocate()
        self.angle_meshes.deallocate()

    def draw_main(self, width: int, height: int) -> None:
        self.pose_meshes.update(True)
        # self.angle_meshes.update(False)

        self.white_space_render.update(True)
        self.r_stream_render.update(False)
        self.tracker_render.update(False)
        for i in range(self.num_cams):
            self.camera_renders[i].update(False)
        for i in range(self.max_players):
            self.pose_renders[i].update(False)

        self.draw_composition()

    def draw_composition(self) -> None:
        self.setView(self.subdivision.width, self.subdivision.height)

        self.white_space_render.draw(self.subdivision.get_rect(SubdivisionType.WS.value))
        self.tracker_render.draw(self.subdivision.get_rect(SubdivisionType.TRK.value))
        self.r_stream_render.draw(self.subdivision.get_rect(SubdivisionType.RST.value))
        for i in range(self.num_cams):
            self.camera_renders[i].draw(self.subdivision.get_rect(SubdivisionType.CAM.value, i))
        for i in range(self.max_players):
            self.pose_renders[i].draw(self.subdivision.get_rect(SubdivisionType.PSN.value, i))

    def draw_secondary(self, monitor_id: int, width: int, height: int) -> None: # override
        self.setView(width, height)
        glEnable(GL_TEXTURE_2D)
        self.white_space_render.draw(Rect(0, 0, width, height))

    def on_main_window_resize(self, width: int, height: int) -> None: # override
        self.subdivision = make_subdivision(self.subdivision_rows, width, height)
        self.allocate_window_renders()


