# Standard library imports
import numpy as np

# Third-party imports
from OpenGL.GL import * # type: ignore

# Local application imports
from modules.gl.Fbo import Fbo
from modules.gl.Image import Image
from modules.gl.Mesh import Mesh
from modules.gl.Text import draw_box_string, text_init

from modules.tracker.Tracklet import Tracklet
from modules.pose.PoseDefinitions import Pose, PoseAngleNames
from modules.pose.PoseStream import PoseStreamData

from modules.render.DataManager import DataManager
from modules.render.Draw.DrawBase import DrawBase, Rect
from modules.render.Mesh.PoseMeshes import PoseMeshes
from modules.render.Mesh.AngleMeshes import AngleMeshes

from modules.utils.HotReloadMethods import HotReloadMethods

# Shaders
from modules.gl.shaders.WS_PoseStream import WS_PoseStream

class DrawPose(DrawBase):
    pose_stream_shader = WS_PoseStream()

    def __init__(self, data: DataManager, pose_meshes: PoseMeshes, angle_meshes: AngleMeshes, cam_id: int) -> None:
        self.data: DataManager = data
        self.fbo: Fbo = Fbo()
        self.image: Image = Image()
        self.cam_id: int = cam_id
        self.pose_stream_image: Image = Image()
        self.pose_meshes: PoseMeshes = pose_meshes
        self.angle_meshes: AngleMeshes = angle_meshes
        text_init()

    def allocate(self, width: int, height: int, internal_format: int) -> None:
        self.fbo.allocate(width, height, internal_format)
        if not DrawPose.pose_stream_shader.allocated:
            DrawPose.pose_stream_shader.allocate(monitor_file=False)

    def deallocate(self) -> None:
        self.fbo.deallocate()
        self.image.deallocate()
        if DrawPose.pose_stream_shader.allocated:
            DrawPose.pose_stream_shader.deallocate()

    def draw(self, rect: Rect) -> None:
        self.fbo.draw(rect.x, rect.y, rect.width, rect.height)

    def update(self, only_if_dirty: bool) -> None:
        key = self.cam_id
        pose: Pose | None = self.data.get_pose(key, only_if_dirty)
        if pose is None:
            return #??
        pose_image_np: np.ndarray | None = pose.image
        if pose_image_np is not None:
            self.image.set_image(pose_image_np)
            self.image.update()
        pose_mesh: Mesh = self.pose_meshes.meshes[pose.id]
        pose_stream: PoseStreamData | None = self.data.get_pose_stream(key, only_if_dirty=True)
        if pose_stream is not None:
            stream_image: np.ndarray = WS_PoseStream.pose_stream_to_image(pose_stream)
            self.pose_stream_image.set_image(stream_image)
            self.pose_stream_image.update()

        angle_mesh: Mesh = self.angle_meshes.meshes[pose.id]

        DrawBase.setView(self.fbo.width, self.fbo.height)
        DrawPose.draw_pose(self.fbo, self.image, pose, pose_mesh, self.pose_stream_image, angle_mesh, DrawPose.pose_stream_shader)
        self.fbo.end()

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
            DrawPose.draw_pose_box(tracklet, pose_mesh, 0, 0, fbo.width, fbo.height, draw_box)
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
    def draw_pose_box(tracklet: Tracklet, pose_mesh: Mesh, x: float, y: float, width: float, height: float, draw_box = False) -> None:
        if draw_box:
            glColor4f(0.0, 0.0, 0.0, 0.1)
            glBegin(GL_QUADS)
            glVertex2f(x, y)        # Bottom left
            glVertex2f(x, y + height)    # Bottom right
            glVertex2f(x + width, y + height)# Top right
            glVertex2f(x + width, y)    # Top left
            glEnd()                 # End drawing
            glColor4f(1.0, 1.0, 1.0, 1.0)  # Reset color

        pose_mesh.draw(x, y, width, height)

        string: str = f'ID: {tracklet.id} Cam: {tracklet.cam_id} Age: {tracklet.age_in_seconds:.2f}'
        x += 9
        y += 12
        draw_box_string(x, y, string)
