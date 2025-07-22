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
from modules.cam.depthcam.Definitions import Tracklet as DepthTracklet
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
    def draw_depth_tracklet(tracklet: DepthTracklet, x: float, y: float, width: float, height: float) -> None:
        if tracklet.status == DepthTracklet.TrackingStatus.REMOVED:
            return

        t_x = x + tracklet.roi.x * width
        t_y = y + tracklet.roi.y * height
        t_w = tracklet.roi.width * width
        t_h = tracklet.roi.height * height

        r: float = 1.0
        g: float = 1.0
        b: float = 1.0
        a: float = min(tracklet.age / 100.0, 0.33)
        if tracklet.status == DepthTracklet.TrackingStatus.NEW:
            r, g, b, a = (1.0, 1.0, 1.0, 1.0)
        if tracklet.status == DepthTracklet.TrackingStatus.TRACKED:
            r, g, b, a = (0.0, 1.0, 0.0, a)
        if tracklet.status == DepthTracklet.TrackingStatus.LOST:
            r, g, b, a = (1.0, 0.0, 0.0, a)
        if tracklet.status == DepthTracklet.TrackingStatus.REMOVED:
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
