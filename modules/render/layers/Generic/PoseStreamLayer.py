# Standard library imports
import numpy as np

# Third-party imports
from OpenGL.GL import * # type: ignore

# Local application imports
from modules.gl.Fbo import Fbo
from modules.gl.Image import Image
from modules.gl.Mesh import Mesh
from modules.gl.LayerBase import LayerBase, Rect
from modules.gl.Text import draw_box_string, text_init
from modules.gl.Texture import Texture

from modules.tracker.Tracklet import Tracklet
from modules.pose.Pose import Pose

from modules.pose.features.PosePoints import POSE_COLOR_LEFT, POSE_COLOR_RIGHT
from modules.pose.features.PoseAngles import ANGLE_NUM_JOINTS, ANGLE_JOINT_NAMES
from modules.pose.PoseStream import PoseStreamData

from modules.CaptureDataHub import CaptureDataHub
from modules.render.meshes.PoseMeshes import PoseMeshes
from modules.render.meshes.AngleMeshes import AngleMeshes

from modules.utils.HotReloadMethods import HotReloadMethods

# Shaders
from modules.gl.shaders.StreamPose import StreamPose

class PoseStreamLayer(LayerBase):
    pose_stream_shader = StreamPose()

    def __init__(self, data: CaptureDataHub, pose_meshes: PoseMeshes, cam_id: int) -> None:
        self.data: CaptureDataHub = data
        self.data_consumer_key: str = data.get_unique_consumer_key()
        self.fbo: Fbo = Fbo()
        self.cam_id: int = cam_id
        self.pose_stream_image: Image = Image()
        self.pose_meshes: PoseMeshes = pose_meshes
        text_init()

        hot_reload = HotReloadMethods(self.__class__, True, True)

    def allocate(self, width: int, height: int, internal_format: int) -> None:
        self.fbo.allocate(width, height, internal_format)
        if not PoseStreamLayer.pose_stream_shader.allocated:
            PoseStreamLayer.pose_stream_shader.allocate(monitor_file=True)

    def deallocate(self) -> None:
        self.fbo.deallocate()
        if PoseStreamLayer.pose_stream_shader.allocated:
            PoseStreamLayer.pose_stream_shader.deallocate()

    def draw(self, rect: Rect) -> None:
        self.fbo.draw(rect.x, rect.y, rect.width, rect.height)

    def update(self) -> None:
        key: int = self.cam_id
        pose: Pose | None = self.data.get_smooth_pose(key, True, self.data_consumer_key)
        if pose is None:
            return #??
        pose_mesh: Mesh = self.pose_meshes.meshes[pose.tracklet.id]
        pose_stream: PoseStreamData | None = self.data.get_pose_stream(key, True, self.data_consumer_key)
        if pose_stream is not None:
            stream_image: np.ndarray = StreamPose.pose_stream_to_image(pose_stream)
            self.pose_stream_image.set_image(stream_image)
            self.pose_stream_image.update()

        # shader gets reset on hot reload, so we need to check if it's allocated
        if not PoseStreamLayer.pose_stream_shader.allocated:
            PoseStreamLayer.pose_stream_shader.allocate(monitor_file=False)

        LayerBase.setView(self.fbo.width, self.fbo.height)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        PoseStreamLayer.draw_pose(self.fbo, pose, pose_mesh, self.pose_stream_image, PoseStreamLayer.pose_stream_shader)


    @staticmethod
    def draw_pose(fbo: Fbo, pose: Pose, pose_mesh: Mesh, angle_image: Image, shader: StreamPose) -> None:
        fbo.begin()

        glClearColor(0.0, 0.0, 0.0, 1.0)
        glClear(GL_COLOR_BUFFER_BIT)
        if pose.tracklet.is_removed:
            return

        tracklet: Tracklet | None = pose.tracklet
        if tracklet is not None:
            draw_box: bool = tracklet.is_lost
            PoseStreamLayer.draw_pose_box(tracklet, pose_mesh, 0, 0, fbo.width, fbo.height, draw_box)
        fbo.end()
        shader.use(fbo.fbo_id, angle_image.tex_id, angle_image.width, angle_image.height, line_width=1.5 / fbo.height)


        angle_num: int = ANGLE_NUM_JOINTS
        step: float = fbo.height / angle_num
        fbo.begin()

        # yellow and light blue
        colors: list[tuple[float, float, float, float]] = [(*POSE_COLOR_LEFT, 1.0), (*POSE_COLOR_RIGHT, 1.0)]

        for i in range(angle_num):
            string: str = ANGLE_JOINT_NAMES[i]
            x: int = 10
            y: int = fbo.height - (int(fbo.height - (i + 0.5) * step) - 12)
            clr: int = i % 2

            draw_box_string(x, y, string, colors[clr], (0.0, 0.0, 0.0, 0.3)) # type: ignore
        fbo.end()


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

        # if pose_mesh.isInitialized():
        #     pose_mesh.draw(x, y, width, height)

        string: str = f'ID: {tracklet.id} Cam: {tracklet.cam_id} Age: {tracklet.age_in_seconds:.2f}'
        x += 9
        y += 12
        draw_box_string(x, y, string)
