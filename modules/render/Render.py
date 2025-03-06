from OpenGL.GL import * # type: ignore
from threading import Lock
import numpy as np
from enum import Enum

from modules.gl.RenderWindow import RenderWindow
from modules.gl.Texture import Texture
from modules.gl.Fbo import Fbo, SwapFbo
from modules.gl.Mesh import Mesh
from modules.gl.Utils import lfo, fit, fill

from modules.pose.PoseConstants import PoseMessage, PoseIndicesFlat

class ImageType(Enum):
    VIDEO = 0
    POSE = 1

class MeshType(Enum):
    POSE_0 = 0
    POSE_1 = 1

class Render(RenderWindow):
    def __init__(self, window_width: int, window_height: int, name: str, fullscreen: bool = False, v_sync = False) -> None:
        super().__init__(window_width, window_height, name, fullscreen, v_sync)

        self.input_mutex: Lock = Lock()

        self.width: int = window_width
        self.height: int = window_height

        self._images: dict[ImageType, np.ndarray | None] = {}
        self.textures: dict[ImageType, Texture] = {}

        for image_type in ImageType:
            self._images[image_type] = None
            self.textures[image_type] = Texture()

        self._vertices: dict[MeshType, np.ndarray | None] = {}
        self._colors: dict[MeshType, np.ndarray | None] = {}
        self.meshes: dict[MeshType, Mesh] = {}
        for mesh_type in MeshType:
            self._vertices[mesh_type] = None
            self._colors[mesh_type] = None
            self.meshes[mesh_type] = Mesh()
            self.meshes[mesh_type].set_indices(PoseIndicesFlat)

        self.fbos: list[Fbo | SwapFbo] = []


    def allocate(self) -> None:
        for fbo in self.fbos: fbo.allocate(self.width, self.height, GL_RGBA32F)
        for mesh in self.meshes.values(): mesh.allocate()

    def draw(self) -> None: # override
        if not self.is_allocated: self.allocate()

        self.update_textures()
        self.update_meshes()

        self.draw_video(ImageType.VIDEO)


    def draw_in_fbo(self, tex: Texture | Fbo | SwapFbo, fbo: Fbo | SwapFbo) -> None:
        self.setView(fbo.width, fbo.height, False, True)
        fbo.begin()
        tex.draw(0, 0, fbo.width, fbo.height)
        fbo.end()

    def draw_video(self, type: ImageType) -> None:
        self.setView(self.window_width, self.window_height)
        tex: Texture = self.textures[type]
        if tex.width == 0 or tex.height == 0:
            return

        x, y, w, h = fit(tex.width, tex.height, self.window_width, self.window_height)
        self.textures[type].draw(x, 0, w, h)

        if  self.meshes[MeshType.POSE_0].isInitialized():
            self.meshes[MeshType.POSE_0].draw(x, 0, w, h)

        self.textures[ImageType.POSE].draw(0, h, 256, 256)


    def update_textures(self) -> None:
        for image_type in ImageType:
            image: np.ndarray | None = self.get_image(image_type)
            texture: Texture = self.textures[image_type]
            if image is not None:
                texture.set_from_image(image)

    def set_image(self, type: ImageType, image: np.ndarray) -> None :
        with self.input_mutex:
            self._images[type] = image
    def get_image(self, type: ImageType) -> np.ndarray | None :
        with self.input_mutex:
            ret_image: np.ndarray | None = self._images[type]
            self._images[type] = None
            return ret_image

    def update_meshes(self) -> None:
        for mesh_type in MeshType:
            vertices: np.ndarray | None = self.get_vertices(mesh_type)
            mesh: Mesh = self.meshes[mesh_type]
            if vertices is not None:
                mesh.set_vertices(vertices)
            colors: np.ndarray | None = self.get_colors(mesh_type)
            if colors is not None:
                mesh.set_colors(colors)

    def set_vertices(self, type: MeshType, vertices: np.ndarray) -> None:
        with self.input_mutex:
            self._vertices[type] = vertices

    def get_vertices(self, type: MeshType) -> np.ndarray | None:
        with self.input_mutex:
            ret_vertices: np.ndarray | None = self._vertices[type]
            self._vertices[type] = None
            return ret_vertices

    def set_colors(self, type: MeshType, colors: np.ndarray) -> None:
        with self.input_mutex:
            self._colors[type] = colors

    def get_colors(self, type: MeshType) -> np.ndarray | None:
        with self.input_mutex:
            ret_colors: np.ndarray | None = self._colors[type]
            self._colors[type] = None
            return ret_colors


    def set_camera_image(self, image: np.ndarray) -> None :
        self.set_image(ImageType.VIDEO, image)

    def set_pose_message(self, message: PoseMessage) -> None:
        self.set_image(ImageType.POSE, message.image)
        self.set_vertices(MeshType.POSE_0, message.pose_list[0].getVertices())
        self.set_colors(MeshType.POSE_0, message.pose_list[0].getColors())