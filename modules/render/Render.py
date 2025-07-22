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
from modules.render.RenderMeshMethods import RenderMeshMethods as Meshmethods

from modules.utils.HotReloadMethods import HotReloadMethods

# Shaders
from modules.gl.shaders.WS_Angles import WS_Angles
from modules.gl.shaders.WS_Lines import WS_Lines
from modules.gl.shaders.WS_PoseStream import WS_PoseStream
from modules.gl.shaders.WS_RStream import WS_RStream


from time import time

class DataVisType(Enum):
    TRACKING = 0
    R_PAIRS = 1

class LightVisType(Enum):
    LINES = 0
    LIGHTS = 1

class SubdivisionType(Enum):
    CAM = "CAM"
    SIM = "SIM"
    PSN = "PSN"
    VIS = "VIS"

Composition_Subdivision = dict[SubdivisionType, dict[int, tuple[int, int, int, int]]]


class Render(RenderBase):
    def __init__(self, settings: Settings) -> None:
        self.data: DataManager = DataManager()

        self.max_players: int = settings.max_players
        self.num_cams: int = settings.camera_num
        self.num_sims: int = len(DataVisType)
        self.num_viss: int = len(LightVisType)
        self.vis_width: int = settings.light_resolution
        self.num_r_streams: int = 3

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
        self.cam_fbos: dict[int, Fbo] = {}
        self.sim_fbos: dict[int, Fbo] = {}
        self.vis_fbos: dict[int, Fbo] = {}
        self.pse_fbos: dict[int, Fbo] = {}
        for i in range(self.num_cams):
            self.cam_fbos[i] = Fbo()
        for i in range(self.num_sims):
            self.sim_fbos[i] = Fbo()
        for i in range(self.num_viss):
            self.vis_fbos[i] = Fbo()
        for i in range(self.max_players):
            self.pse_fbos[i] = Fbo()

        # meshes
        self.pose_meshes: dict[int, Mesh] = {}
        self.angle_meshes: dict[int, Mesh] = {}
        self.R_meshes: dict[Tuple[int, int], Mesh] = {}
        for i in range(self.max_players):
            self.pose_meshes[i] = Mesh()
            self.pose_meshes[i].set_indices(PoseEdgeIndices)
            self.angle_meshes[i] = Mesh()

        # shaders
        self.vis_line_shader = WS_Lines()
        self.vis_angle_shader = WS_Angles()
        self.pose_stream_shader = WS_PoseStream()
        self.r_stream_shader = WS_RStream()
        self.all_shaders: list[Shader] = [self.vis_line_shader, self.vis_angle_shader, self.pose_stream_shader, self.r_stream_shader]

        # composition
        self.subdivision_rows: list[SubdivisionRow] = [
            SubdivisionRow(name=SubdivisionType.CAM.value, columns=self.num_cams, rows=1, src_aspect_ratio=16/9, padding=Point2f(1.0, 1.0)),
            SubdivisionRow(name=SubdivisionType.SIM.value, columns=1, rows=self.num_sims, src_aspect_ratio=12.0, padding=Point2f(0.0, 1.0)),
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
        for s in self.all_shaders:
            s.allocate(True) # type: ignore
        for fbo in self.vis_fbos.values():
            fbo.allocate(self.vis_width, 1, GL_RGBA32F)
        self.allocated = True

    def deallocate(self) -> None: # override
        for fbo in self.cam_fbos.values():
            fbo.deallocate()
        for fbo in self.sim_fbos.values():
            fbo.deallocate()
        for fbo in self.pse_fbos.values():
            fbo.deallocate()
        for fbo in self.vis_fbos.values():
            fbo.deallocate()

        for mesh in self.pose_meshes.values():
            mesh.deallocate()
        self.pose_meshes.clear()
        for mesh in self.angle_meshes.values():
            mesh.deallocate()
        self.angle_meshes.clear()

        for shader in self.all_shaders:
            shader.deallocate()

        self.allocated = False

    def draw_main(self, width: int, height: int) -> None:
        try:
            # update meshes
            Meshmethods.update_pose_meshes(self.data, self.max_players, self.pose_meshes)
            Meshmethods.update_angle_meshes(self.data, self.angle_meshes, self.max_players)

            self.draw_cameras()
            self.draw_sims()
            self.draw_poses()
            self.draw_lights()

            self.draw_composition()
        except Exception as e:
            print(f"Error in draw: {e}")

    def draw_secondary(self, monitor_id: int, width: int, height: int) -> None: # override
        self.setView(width, height)
        glEnable(GL_TEXTURE_2D)
        self.vis_fbos[0].draw(0, 0, width, height)

    def on_main_window_resize(self, width: int, height: int) -> None: # override
        self.subdivision = make_subdivision(self.subdivision_rows, width, height)

        for key in self.cam_fbos.keys():
            rect = self.subdivision.rows[SubdivisionType.CAM.value][key]
            w, h = int(rect.width), int(rect.height)
            self.cam_fbos[key].allocate(w, h, GL_RGBA)

        for key in self.sim_fbos.keys():
            rect = self.subdivision.rows[SubdivisionType.SIM.value][key]
            w, h = int(rect.width), int(rect.height)
            self.sim_fbos[key].allocate(w, h, GL_RGBA)

        for key in self.pse_fbos.keys():
            rect = self.subdivision.rows[SubdivisionType.PSN.value][key]
            w, h = int(rect.width), int(rect.height)
            self.pse_fbos[key].allocate(w, h, GL_RGBA)

    # DRAW METHODS

    def draw_cameras(self) -> None:
        for i in range(self.num_cams):
            frame: np.ndarray | None = self.data.get_cam_image(i)
            image: Image = self.cam_images[i]
            if frame is not None:
                image.set_image(frame)
                image.update()
            fbo: Fbo = self.cam_fbos[i]
            depth_tracklets: list[CamTracklet] | None = self.data.get_depth_tracklets(i, False)
            poses: list[Pose] = self.data.get_poses_for_cam(i)

            self.setView(fbo.width, fbo.height)
            fbo.begin()
            glClearColor(0.0, 0.0, 0.0, 1.0)
            image.draw(0, 0, fbo.width, fbo.height)
            Render.draw_camera_overlay(depth_tracklets, poses, self.pose_meshes, 0, 0, fbo.width, fbo.height)
            # glFlush()
            fbo.end()

    def draw_poses(self) -> None:
        for i in range(self.max_players):
            fbo: Fbo = self.pse_fbos[i]
            pose: Pose | None = self.data.get_pose(i, False)
            if pose is None:
                continue #??
            pose_image: Image = self.pse_images[i]
            pose_image_np: np.ndarray | None = pose.image
            if pose_image_np is not None:
                pose_image.set_image(pose_image_np)
                pose_image.update()
            pose_mesh: Mesh = self.pose_meshes[pose.id]
            pose_stream: PoseStreamData | None = self.data.get_pose_stream(i)
            a_s_image: Image = self.a_s_images[i]
            if pose_stream is not None:
                a_s_image_np: np.ndarray = WS_PoseStream.pose_stream_to_image(pose_stream)
                a_s_image.set_image(a_s_image_np)
                a_s_image.update()

            angle_mesh: Mesh = self.angle_meshes[pose.id]

            self.setView(fbo.width, fbo.height)
            Render.draw_pose(fbo, pose_image, pose, pose_mesh, a_s_image, angle_mesh, self.pose_stream_shader)
            fbo.end()

    def draw_sims(self) -> None:
        for i in range(self.num_sims):
            fbo: Fbo = self.sim_fbos[i]
            if i == DataVisType.TRACKING.value:
                self.setView(fbo.width, fbo.height)
                fbo.begin()
                self.draw_map_positions(self.data.get_tracklets(), self.num_cams, 0, 0, fbo.width, fbo.height)
                # glFlush()
                fbo.end()
            elif i == DataVisType.R_PAIRS.value:
                self.draw_correlations(fbo)

    def draw_correlations(self, fbo: Fbo) -> None:
        correlation_streams: PairCorrelationStreamData | None = self.data.get_correlation_streams()
        if correlation_streams is None:
            return

        pairs: list[Tuple[int, int]] = correlation_streams.get_top_pairs(self.num_r_streams)
        num_pairs: int = len(pairs)

        image_np: np.ndarray = WS_RStream.r_stream_to_image(correlation_streams, self.num_r_streams)
        self.r_s_image.set_image(image_np)
        self.r_s_image.update()

        self.setView(fbo.width, fbo.height)
        self.r_stream_shader.use(fbo.fbo_id, self.r_s_image.tex_id, self.r_s_image.width, self.r_s_image.height, 1.5 / fbo.height)

        step: float = fbo.height / self.num_r_streams

        fbo.begin()
        glColor4f(1.0, 0.5, 0.5, 1.0)  # Set color to white
        for i in range(num_pairs):
            pair: Tuple[int, int] = pairs[i]
            string: str = f'{pair[0]} | {pair[1]}'
            x: int = fbo.width - 100
            y: int = fbo.height - (int(fbo.height - (i + 0.5) * step) - 12)
            draw_box_string(x, y, string, big=True) # type: ignore
        glColor4f(1.0, 1.0, 1.0, 1.0)  # Set color to white
        fbo.end()

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
        for i in range(self.num_cams):
            rect = self.subdivision.rows[SubdivisionType.CAM.value][i]
            x, y, w, h = rect.x, rect.y, rect.width, rect.height
            self.cam_fbos[i].draw(x, y, w, h)

        for i in range(self.num_sims):
            rect = self.subdivision.rows[SubdivisionType.SIM.value][i]
            x, y, w, h = rect.x, rect.y, rect.width, rect.height
            self.sim_fbos[i].draw(x, y, w, h)

        for i in range(self.max_players):
            rect = self.subdivision.rows[SubdivisionType.PSN.value][i]
            x, y, w, h = rect.x, rect.y, rect.width, rect.height
            self.pse_fbos[i].draw(x, y, w, h)

        for i in range(self.num_viss):
            rect = self.subdivision.rows[SubdivisionType.VIS.value][i]
            x, y, w, h = rect.x, rect.y, rect.width, rect.height
            self.vis_fbos[i].draw(x, y, w, h)

    # STATIC METHODS
    @staticmethod
    def draw_camera_overlay(depth_tracklets: list[CamTracklet], poses: list[Pose], pose_meshes: dict[int, Mesh], x: float, y: float, width: float, height: float) -> None:
        for pose in poses:
            tracklet: Tracklet | None = pose.tracklet
            if tracklet is None or tracklet.is_removed or tracklet.is_lost:
                continue
            roi: Rect | None = pose.crop_rect
            mesh: Mesh = pose_meshes[pose.id]
            if roi is None or not mesh.isInitialized():
                continue
            roi_x, roi_y, roi_w, roi_h = roi.x, roi.y, roi.width, roi.height
            roi_x: float = x + roi_x * width
            roi_y: float = y + roi_y * height
            roi_w: float = roi_w * width
            roi_h: float = roi_h * height
            Render.draw_tracklet(tracklet, mesh, roi_x, roi_y, roi_w, roi_h, True, True, False)

        for depth_tracklet in depth_tracklets:
            Render.draw_depth_tracklet(depth_tracklet, 0, 0, width, height)

    @staticmethod
    def draw_depth_tracklet(tracklet: CamTracklet, x: float, y: float, width: float, height: float) -> None:
        if tracklet.status == CamTracklet.TrackingStatus.REMOVED:
            return

        t_x = x + tracklet.roi.x * width
        t_y = y + tracklet.roi.y * height
        t_w = tracklet.roi.width * width
        t_h = tracklet.roi.height * height

        r: float = 1.0
        g: float = 1.0
        b: float = 1.0
        a: float = min(tracklet.age / 100.0, 0.33)
        if tracklet.status == CamTracklet.TrackingStatus.NEW:
            r, g, b, a = (1.0, 1.0, 1.0, 1.0)
        if tracklet.status == CamTracklet.TrackingStatus.TRACKED:
            r, g, b, a = (0.0, 1.0, 0.0, a)
        if tracklet.status == CamTracklet.TrackingStatus.LOST:
            r, g, b, a = (1.0, 0.0, 0.0, a)
        if tracklet.status == CamTracklet.TrackingStatus.REMOVED:
            r, g, b, a = (1.0, 0.0, 0.0, 1.0)

        glColor4f(r, g, b, a)   # Set color
        glBegin(GL_QUADS)       # Start drawing a quad
        glVertex2f(t_x, t_y)        # Bottom left
        glVertex2f(t_x, t_y + t_h)    # Bottom right
        glVertex2f(t_x + t_w, t_y + t_h)# Top right
        glVertex2f(t_x + t_w, t_y)    # Top left
        glEnd()                 # End drawing
        glColor4f(1.0, 1.0, 1.0, 1.0)  # Reset color

        string: str
        t_x += t_w -6
        if t_x + 66 > width:
            t_x: float = width - 66
        t_y += 22
        string = f'ID: {tracklet.id}'
        draw_box_string(t_x, t_y, string)
        t_y += 22
        string = f'Age: {tracklet.age}'
        draw_box_string(t_x, t_y, string)

        # glFlush()               # Render now

    @staticmethod
    def draw_tracklet(tracklet: Tracklet, pose_mesh: Mesh, x: float, y: float, width: float, height: float, draw_box = False, draw_pose = False, draw_text = False) -> None:
        if draw_box:
            glColor4f(0.0, 0.0, 0.0, 0.1)
            glBegin(GL_QUADS)
            glVertex2f(x, y)        # Bottom left
            glVertex2f(x, y + height)    # Bottom right
            glVertex2f(x + width, y + height)# Top right
            glVertex2f(x + width, y)    # Top left
            glEnd()                 # End drawing
            glColor4f(1.0, 1.0, 1.0, 1.0)  # Reset color

        if draw_pose and pose_mesh.isInitialized():
            pose_mesh.draw(x, y, width, height)

        if draw_text:
            string: str = f'ID: {tracklet.id} Cam: {tracklet.cam_id} Age: {tracklet.age_in_seconds:.2f}'
            x += 9
            y += 12
            draw_box_string(x, y, string)

    @staticmethod
    def draw_pose(fbo: Fbo, pose_image: Image, pose: Pose, pose_mesh: Mesh, angle_image: Image, angle_mesh: Mesh, shader: WS_PoseStream) -> None:
        fbo.begin()

        if pose.is_final:
            glClearColor(0.0, 0.0, 0.0, 1.0)
            glClear(GL_COLOR_BUFFER_BIT)
            return

        pose_image.draw(0, 0, fbo.width, fbo.height)
        tracklet: Tracklet | None = pose.tracklet
        if tracklet is not None:
            draw_box: bool = tracklet.is_lost
            Render.draw_tracklet(tracklet, pose_mesh, 0, 0, fbo.width, fbo.height, draw_box, True, True)
        # if angle_mesh.isInitialized():
        #     angle_mesh.draw(0, 0, fbo.width, fbo.height)
        glFlush()
        fbo.end()
        shader.use(fbo.fbo_id, angle_image.tex_id, angle_image.width, angle_image.height, 1.5 / fbo.height)


        angle_num: int = len(PoseAngleNames)
        step: float = fbo.height / angle_num
        fbo.begin()

        # yellow and light blue
        colors: list[tuple[float, float, float, float]] = [(1.0, 0.5, 0.0, 1.0), (0.0, 0.8, 1.0, 1.0)]

        for i in range(angle_num):
            string: str = PoseAngleNames[i]
            x: int = 10
            y: int = fbo.height - (int(fbo.height - (i + 0.5) * step) - 12)
            clr: int = i % 2

            draw_box_string(x, y, string, colors[clr], (0.0, 0.0, 0.0, 0.3)) # type: ignore

    @staticmethod
    def draw_map_positions(tracklets: dict[int, Tracklet], num_cams: int, x: float, y: float, width: float, height: float) -> None:
        glClearColor(0.0, 0.0, 0.0, 1.0)
        glClear(GL_COLOR_BUFFER_BIT)

        for tracklet in tracklets.values():
            if tracklet is None:
                continue
            if tracklet.status != TrackingStatus.TRACKED and tracklet.status != TrackingStatus.NEW:
                continue

            tracklet_metadata: TrackerMetadata | None = tracklet.metadata
            if tracklet_metadata is None or tracklet_metadata.tracker_type != TrackerType.PANORAMIC:
                continue

            world_angle: float = getattr(tracklet.metadata, "world_angle", 0.0)
            local_angle: float = getattr(tracklet.metadata, "local_angle", 0.0)
            overlap: bool = getattr(tracklet.metadata, "overlap", False)

            roi_width: float = tracklet.roi.width * width / num_cams
            roi_height: float = tracklet.roi.height * height
            roi_x: float = world_angle / 360.0 * width + x
            roi_y: float = tracklet.roi.y * height + y

            color: list[float] = TrackletIdColor(tracklet.id, aplha=0.9)
            if overlap == True:
                color[3] = 0.3
            if tracklet.status == TrackingStatus.NEW:
                color = [1.0, 1.0, 1.0, 1.0]

            glColor4f(*color)  # Reset color
            glBegin(GL_QUADS)       # Start drawing a quad
            glVertex2f(roi_x, roi_y)        # Bottom left
            glVertex2f(roi_x, roi_y + roi_height)    # Bottom right
            glVertex2f(roi_x + roi_width, roi_y + roi_height)# Top right
            glVertex2f(roi_x + roi_width, roi_y)    # Top left
            glEnd()                 # End drawing
            glColor4f(1.0, 1.0, 1.0, 1.0)  # Reset color

            string: str
            roi_x += 9
            roi_y += 22
            string = f'W: {world_angle:.1f}'
            draw_box_string(roi_x, roi_y, string)
            roi_y += 22
            string = f'L: {local_angle:.1f}'
            draw_box_string(roi_x, roi_y, string)

