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
from modules.render.RenderCompositionSubdivision import make_subdivision, SubdivisionRow, Subdivision
from modules.render.Mesh.PoseMeshes import PoseMeshes
from modules.render.Mesh.AngleMeshes import AngleMeshes
from modules.render.Draw.DrawPanoramicTracker import DrawPanoramicTracker
from modules.render.Draw.DrawCamera import DrawCamera
from modules.render.Draw.DrawPose import DrawPose
from modules.render.Draw.DrawCorrelationStream import DrawRStream
from modules.render.Draw.DrawWhiteSpace import DrawWhiteSpace

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
        self.draw_white_space =     DrawWhiteSpace(self.data)
        self.draw_r_stream =        DrawRStream(self.data, self.num_R_streams)
        self.draw_tracker =         DrawPanoramicTracker(self.data, self.num_cams)
        self.draw_cameras:          dict[int, DrawCamera] = {}
        self.draw_poses:            dict[int, DrawPose] = {}

        for i in range(self.num_cams):
            self.draw_cameras[i] = DrawCamera(self.data, self.pose_meshes, i)
        for i in range(self.max_players):
            self.draw_poses[i] = DrawPose(self.data, self.pose_meshes, self.angle_meshes, i)

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

        self.allocate_window_draws()
        self.draw_white_space.allocate(self.ws_width, 1, GL_RGBA32F)

    def allocate_window_draws(self) -> None:
        w, h = self.subdivision.get_allocation_size(SubdivisionType.RST.value)
        self.draw_r_stream.allocate(w, h, GL_RGBA)
        w, h = self.subdivision.get_allocation_size(SubdivisionType.TRK.value)
        self.draw_tracker.allocate(w, h, GL_RGBA)
        for key in self.draw_cameras.keys():
            w, h = self.subdivision.get_allocation_size(SubdivisionType.CAM.value, key)
            self.draw_cameras[key].allocate(w, h, GL_RGBA)
        for key in self.draw_poses.keys():
            w, h = self.subdivision.get_allocation_size(SubdivisionType.PSN.value, key)
            self.draw_poses[key].allocate(w, h, GL_RGBA)

    def deallocate(self) -> None:
        self.draw_white_space.deallocate()
        self.draw_r_stream.deallocate()
        self.draw_tracker.deallocate()
        for draw in self.draw_cameras.values():
            draw.deallocate()
        for draw in self.draw_poses.values():
            draw.deallocate()

        self.pose_meshes.deallocate()
        self.angle_meshes.deallocate()

    def draw_main(self, width: int, height: int) -> None:
        self.pose_meshes.update(True)
        # self.angle_meshes.update(False)

        self.draw_white_space.update(True)
        self.draw_r_stream.update(False)
        self.draw_tracker.update(False)
        for i in range(self.num_cams):
            self.draw_cameras[i].update(False)
        for i in range(self.max_players):
            self.draw_poses[i].update(False)

        self.draw_composition()

    def draw_composition(self) -> None:
        self.setView(self.subdivision.width, self.subdivision.height)

        self.draw_white_space.draw(self.subdivision.get_rect(SubdivisionType.WS.value))
        self.draw_tracker.draw(self.subdivision.get_rect(SubdivisionType.TRK.value))
        self.draw_r_stream.draw(self.subdivision.get_rect(SubdivisionType.RST.value))
        for i in range(self.num_cams):
            self.draw_cameras[i].draw(self.subdivision.get_rect(SubdivisionType.CAM.value, i))
        for i in range(self.max_players):
            self.draw_poses[i].draw(self.subdivision.get_rect(SubdivisionType.PSN.value, i))

    def draw_secondary(self, monitor_id: int, width: int, height: int) -> None: # override
        self.setView(width, height)
        glEnable(GL_TEXTURE_2D)
        self.draw_white_space.draw(Rect(0, 0, width, height))

    def on_main_window_resize(self, width: int, height: int) -> None: # override
        self.subdivision = make_subdivision(self.subdivision_rows, width, height)
        self.allocate_window_draws()


