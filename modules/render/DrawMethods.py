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
from modules.render.Mesh.AngleMeshes import AngleMeshes as Meshmethods

from modules.utils.HotReloadMethods import HotReloadMethods

# Shaders
from modules.gl.shaders.WS_Angles import WS_Angles
from modules.gl.shaders.WS_Lines import WS_Lines
from modules.gl.shaders.WS_PoseStream import WS_PoseStream
from modules.gl.shaders.WS_RStream import WS_RStream


class DrawMethods:
    @staticmethod
    def setView(width, height) -> None:
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(0, width, height, 0, -1, 1)

        glMatrixMode(GL_MODELVIEW)
        glViewport(0, 0, width, height)

    @staticmethod
    def draw_cameras(data: DataManager, cam_fbos: dict[int, Fbo], cam_images: dict[int, Image], pose_meshes: dict[int, Mesh]) -> None:
        for key in cam_fbos.keys():
            frame: np.ndarray | None = data.get_cam_image(key)
            image: Image = cam_images[key]
            if frame is not None:
                image.set_image(frame)
                image.update()
            fbo: Fbo = cam_fbos[key]
            depth_tracklets: list[CamTracklet] | None = data.get_depth_tracklets(key, False)
            poses: list[Pose] = data.get_poses_for_cam(key)

            DrawMethods.setView(fbo.width, fbo.height)
            fbo.begin()
            glClearColor(0.0, 0.0, 0.0, 1.0)
            image.draw(0, 0, fbo.width, fbo.height)
            DrawMethods.draw_camera_overlay(depth_tracklets, poses, pose_meshes, 0, 0, fbo.width, fbo.height)
            # glFlush()
            fbo.end()

    @staticmethod
    def draw_poses(data: DataManager, pse_fbos: dict[int, Fbo], pse_images: dict[int, Image], pose_meshes: dict[int, Mesh], a_s_images: dict[int, Image], angle_meshes: dict[int, Mesh], pose_stream_shader: WS_PoseStream) -> None:
        for key in pse_fbos.keys():
            fbo: Fbo = pse_fbos[key]
            pose: Pose | None = data.get_pose(key, False)
            if pose is None:
                continue #??
            pose_image: Image = pse_images[key]
            pose_image_np: np.ndarray | None = pose.image
            if pose_image_np is not None:
                pose_image.set_image(pose_image_np)
                pose_image.update()
            pose_mesh: Mesh = pose_meshes[pose.id]
            pose_stream: PoseStreamData | None = data.get_pose_stream(key, only_if_dirty=True)
            a_s_image: Image = a_s_images[key]
            if pose_stream is not None:
                a_s_image_np: np.ndarray = WS_PoseStream.pose_stream_to_image(pose_stream)
                a_s_image.set_image(a_s_image_np)
                a_s_image.update()

            angle_mesh: Mesh = angle_meshes[pose.id]

            DrawMethods.setView(fbo.width, fbo.height)
            DrawMethods.draw_pose(fbo, pose_image, pose, pose_mesh, a_s_image, angle_mesh, pose_stream_shader)
            fbo.end()

    @staticmethod
    def draw_correlations(data: DataManager, fbo: Fbo, r_s_image: Image, r_stream_shader: WS_RStream, num_r_streams: int) -> None:
        correlation_streams: PairCorrelationStreamData | None = data.get_correlation_streams()
        if correlation_streams is None:
            return

        pairs: list[Tuple[int, int]] = correlation_streams.get_top_pairs(num_r_streams)
        num_pairs: int = len(pairs)

        image_np: np.ndarray = WS_RStream.r_stream_to_image(correlation_streams, num_r_streams)
        r_s_image.set_image(image_np)
        r_s_image.update()

        DrawMethods.setView(fbo.width, fbo.height)
        r_stream_shader.use(fbo.fbo_id, r_s_image.tex_id, r_s_image.width, r_s_image.height, 1.5 / fbo.height)

        step: float = fbo.height / num_r_streams

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
            DrawMethods.draw_tracklet(tracklet, mesh, roi_x, roi_y, roi_w, roi_h, True, True, False)

        for depth_tracklet in depth_tracklets:
            DrawMethods.draw_depth_tracklet(depth_tracklet, 0, 0, width, height)

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
            DrawMethods.draw_tracklet(tracklet, pose_mesh, 0, 0, fbo.width, fbo.height, draw_box, True, True)
        if angle_mesh.isInitialized():
            angle_mesh.draw(0, 0, fbo.width, fbo.height)
        # glFlush()
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
    def draw_map_positions(data: DataManager, fbo: Fbo, num_cams: int) -> None:
        fbo.begin()
        glClearColor(0.0, 0.0, 0.0, 1.0)
        glClear(GL_COLOR_BUFFER_BIT)

        tracklets: dict[int, Tracklet] = data.get_tracklets()

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

            roi_width: float = tracklet.roi.width * fbo.width / num_cams
            roi_height: float = tracklet.roi.height * fbo.height
            roi_x: float = world_angle / 360.0 * fbo.width
            roi_y: float = tracklet.roi.y * fbo.height

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

        glFlush()  # Render now
        fbo.end()

