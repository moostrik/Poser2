# Standard library imports
import numpy as np
from enum import Enum
from typing import Optional, Tuple

# Third-party imports
from OpenGL.GL import * # type: ignore
# import glfw
# import OpenGL.GLUT as glut

# Local application imports
from modules.gl.Fbo import Fbo
from modules.gl.Image import Image
from modules.gl.Mesh import Mesh
from modules.gl.WindowManager import WindowManager
# from modules.gl.RenderWindowGLUT import RenderWindow
from modules.gl.Shader import Shader
from modules.gl.Text import draw_string, draw_box_string, text_init
from modules.gl.RenderBase import RenderBase

from modules.av.Definitions import AvOutput
from modules.cam.depthcam.Definitions import Tracklet as CamTracklet
from modules.tracker.TrackerBase import TrackerType, TrackerMetadata
from modules.tracker.Tracklet import Tracklet, TrackletIdColor, TrackingStatus
from modules.correlation.PairCorrelationStream import PairCorrelationStreamData
from modules.pose.PoseDefinitions import Pose, PosePoints, PoseEdgeIndices, PoseAngleNames
from modules.pose.PoseStream import PoseStreamData
from modules.Settings import Settings
from modules.utils.PointsAndRects import Rect, Point2f

from modules.render.RenderCompositionSubdivision import make_subdivision, SubdivisionRow, Subdivision
from modules.render.DataManager import DataManager
from modules.render.Mesh.AngleMeshes import AngleMeshes
from modules.render.DrawMethods import DrawMethods

from modules.render.Mesh.PoseMeshes import PoseMeshes
from modules.render.Mesh.AngleMeshes import AngleMeshes
from modules.render.Draw.DrawPanoramicTracker import DrawPanoramicTracker

from modules.utils.HotReloadMethods import HotReloadMethods

# Shaders
from modules.gl.shaders.WS_Angles import WS_Angles
from modules.gl.shaders.WS_Lines import WS_Lines
from modules.gl.shaders.WS_PoseStream import WS_PoseStream
from modules.gl.shaders.WS_RStream import WS_RStream


from time import time


class LightVisType(Enum):
    LINES = 0
    LIGHTS = 1

class SubdivisionType(Enum):
    CAM = "CAM"
    TRK = "TRK"
    RST = "RST"
    PSN = "PSN"
    VIS = "VIS"

Composition_Subdivision = dict[SubdivisionType, dict[int, tuple[int, int, int, int]]]


class Render(RenderBase):
    def __init__(self, settings: Settings) -> None:
        self.data: DataManager = DataManager()

        self.max_players: int = settings.max_players
        self.num_cams: int = settings.camera_num
        self.num_viss: int = len(LightVisType)
        self.vis_width: int = settings.light_resolution
        self.num_r_streams: int = 3
        self.r_stream_width: int = settings.corr_stream_capacity

        # images
        self.avi_image = Image()
        self.cam_images: dict[int, Image] = {}
        self.pse_images: dict[int, Image] = {}
        self.a_s_images: dict[int, Image] = {}
        self.r_s_image = Image()
        for i in range(self.num_cams):
            self.cam_images[i] = Image()
        for i in range(self.max_players):
            self.pse_images[i] = Image()
            self.a_s_images[i] = Image()

        # fbos
        # self.trk_fbo: Fbo = Fbo()
        self.rst_fbo: Fbo = Fbo()
        self.cam_fbos: dict[int, Fbo] = {}
        self.vis_fbos: dict[int, Fbo] = {}
        self.pse_fbos: dict[int, Fbo] = {}
        for i in range(self.num_cams):
            self.cam_fbos[i] = Fbo()
        for i in range(self.num_viss):
            self.vis_fbos[i] = Fbo()
        for i in range(self.max_players):
            self.pse_fbos[i] = Fbo()

        # meshes
        self.pose_meshes = PoseMeshes(self.data, self.max_players)
        self.angle_meshes = AngleMeshes(self.data, self.max_players)

        # drawers
        self.draw_tracker = DrawPanoramicTracker(self.data, self.num_cams)

        # shaders
        self.vis_line_shader = WS_Lines()
        self.vis_angle_shader = WS_Angles()
        self.pose_stream_shader = WS_PoseStream()
        self.r_stream_shader = WS_RStream()
        self.all_shaders: list[Shader] = [self.vis_line_shader, self.vis_angle_shader, self.pose_stream_shader, self.r_stream_shader]

        # composition
        self.subdivision_rows: list[SubdivisionRow] = [
            SubdivisionRow(name=SubdivisionType.CAM.value, columns=self.num_cams, rows=1, src_aspect_ratio=16/9, padding=Point2f(1.0, 1.0)),
            SubdivisionRow(name=SubdivisionType.TRK.value, columns=1, rows=1, src_aspect_ratio=12.0, padding=Point2f(0.0, 1.0)),
            SubdivisionRow(name=SubdivisionType.RST.value, columns=1, rows=1, src_aspect_ratio=12.0, padding=Point2f(0.0, 1.0)),
            SubdivisionRow(name=SubdivisionType.PSN.value, columns=self.max_players, rows=1, src_aspect_ratio=1.0, padding=Point2f(1.0, 1.0)),
            SubdivisionRow(name=SubdivisionType.VIS.value, columns=1, rows=self.num_viss, src_aspect_ratio=20.0, padding=Point2f(0.0, 1.0)),
        ]
        self.subdivision: Subdivision = make_subdivision(self.subdivision_rows, settings.render_width, settings.render_height)

        # window manager
        secondary_monitor_ids: list[int] = [i for i in range(1, self.num_cams + 1)]
        self.window_manager: WindowManager = WindowManager(
            self,
            self.subdivision.width, self.subdivision.height,
            settings.render_title,
            settings.render_fullscreen,
            settings.render_v_sync, settings.render_fps,
            settings.render_x, settings.render_y,
            settings.render_monitor,
            secondary_monitor_ids
        )

        # text
        text_init()

        # hot reloader
        self.hot_reloader = HotReloadMethods(self.__class__, True, True)

    def allocate(self) -> None: # override
        glEnable(GL_TEXTURE_2D)
        glClearColor(0.0, 0.0, 0.0, 1.0)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        # self.setView(self.window_width, self.window_height)

        version = glGetString(GL_VERSION)
        opengl_version = version.decode("utf-8")  # type: ignore
        print("OpenGL version:", opengl_version)

        self.on_main_window_resize(self.subdivision.width, self.subdivision.height) # allocated fbos

        self.pose_meshes.allocate()
        self.angle_meshes.allocate()

        for s in self.all_shaders:
            s.allocate(True) # type: ignore

        for fbo in self.vis_fbos.values():
            fbo.allocate(self.vis_width, 1, GL_RGBA32F)
        self.allocated = True

    def on_main_window_resize(self, width: int, height: int) -> None: # override
        self.subdivision = make_subdivision(self.subdivision_rows, width, height)

        rect = self.subdivision.rows[SubdivisionType.RST.value][0]
        w: int = self.r_stream_width
        h: int = int(rect.height)
        self.rst_fbo.allocate(w, h, GL_RGBA)

        rect: Rect = self.subdivision.rows[SubdivisionType.TRK.value][0]
        w, h = int(rect.width), int(rect.height)
        self.draw_tracker.allocate(w, h, GL_RGBA)

        for key in self.cam_fbos.keys():
            rect = self.subdivision.rows[SubdivisionType.CAM.value][key]
            w, h = int(rect.width), int(rect.height)
            self.cam_fbos[key].allocate(w, h, GL_RGBA)

        for key in self.pse_fbos.keys():
            rect = self.subdivision.rows[SubdivisionType.PSN.value][key]
            w, h = int(rect.width), int(rect.height)
            self.pse_fbos[key].allocate(w, h, GL_RGBA)

    def deallocate(self) -> None: # override
        self.rst_fbo.deallocate()
        self.draw_tracker.deallocate()

        for fbo in self.cam_fbos.values():
            fbo.deallocate()
        for fbo in self.pse_fbos.values():
            fbo.deallocate()
        for fbo in self.vis_fbos.values():
            fbo.deallocate()


        self.pose_meshes.deallocate()
        self.angle_meshes.deallocate()

        for shader in self.all_shaders:
            shader.deallocate()

        self.allocated = False

    def draw_main(self, width: int, height: int) -> None:
        try:
            self.pose_meshes.update(True)
            # self.angle_meshes.update(False)

            DrawMethods.draw_cameras(self.data, self.cam_fbos, self.cam_images, self.pose_meshes.meshes)
            DrawMethods.draw_poses(self.data, self.pse_fbos, self.pse_images, self.pose_meshes.meshes, self.a_s_images, self.angle_meshes.meshes, self.pose_stream_shader)
            DrawMethods.draw_correlations(self.data, self.rst_fbo, self.r_s_image, self.r_stream_shader, self.num_r_streams)
            self.draw_tracker.update(False)

            self.draw_lights()
            self.draw_composition()

        except Exception as e:
            print(f"Error in draw: {e}")

    def draw_secondary(self, monitor_id: int, width: int, height: int) -> None: # override
        self.setView(width, height)
        glEnable(GL_TEXTURE_2D)
        self.vis_fbos[0].draw(0, 0, width, height)

    # DRAW METHODS
    def draw_lights(self) -> None:
        light_image: AvOutput | None = self.data.get_light_image()
        if light_image is None:
            return

        self.avi_image.set_image(light_image.img)
        self.avi_image.update()

        for i in range(self.num_viss):
            fbo: Fbo = self.vis_fbos[i]
            self.setView(fbo.width, fbo.height)
            fbo.begin()
            if i == LightVisType.LINES.value:
                self.vis_line_shader.use(fbo.fbo_id, self.avi_image.tex_id)
            elif i == LightVisType.LIGHTS.value:
                self.vis_angle_shader.use(fbo.fbo_id, self.avi_image.tex_id, light_image.resolution)
            glFlush()
            fbo.end()

    def draw_composition(self) -> None:
        self.setView(self.subdivision.width, self.subdivision.height)

        rect = self.subdivision.rows[SubdivisionType.TRK.value][0]
        x, y, w, h = rect.x, rect.y, rect.width, rect.height
        self.draw_tracker.draw(x, y, w, h)

        rect = self.subdivision.rows[SubdivisionType.RST.value][0]
        x, y, w, h = rect.x, rect.y, rect.width, rect.height
        self.rst_fbo.draw(x, y, w, h)

        for i in range(self.num_cams):
            rect = self.subdivision.rows[SubdivisionType.CAM.value][i]
            x, y, w, h = rect.x, rect.y, rect.width, rect.height
            self.cam_fbos[i].draw(x, y, w, h)

        for i in range(self.max_players):
            rect = self.subdivision.rows[SubdivisionType.PSN.value][i]
            x, y, w, h = rect.x, rect.y, rect.width, rect.height
            self.pse_fbos[i].draw(x, y, w, h)

        for i in range(self.num_viss):
            rect = self.subdivision.rows[SubdivisionType.VIS.value][i]
            x, y, w, h = rect.x, rect.y, rect.width, rect.height
            self.vis_fbos[i].draw(x, y, w, h)
