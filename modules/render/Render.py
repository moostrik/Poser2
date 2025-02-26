from OpenGL.GL import * # type: ignore
import OpenGL.GLUT as glut
from threading import Lock
import cv2
import numpy as np
from enum import Enum
import os
import random

from modules.gl.RenderWindow import RenderWindow
from modules.gl.Texture import Texture
from modules.gl.Fbo import Fbo, SwapFbo
from modules.gl.shaders.Blend import Blend
from modules.gl.shaders.BlurH import BlurH
from modules.gl.shaders.BlurV import BlurV
from modules.gl.shaders.Noise import Noise
from modules.gl.shaders.NoiseBlend import NoiseBlend
from modules.gl.shaders.NoiseSimplex import NoiseSimplex
from modules.gl.shaders.NoiseSimplexBlend import NoiseSimplexBlend
from modules.gl.shaders.Omission import Omission
from modules.gl.shaders.FlashIn import FlashIn
from modules.gl.shaders.FlashOut import FlashOut
from modules.gl.shaders.Hsv import Hsv
from modules.gl.Utils import lfo, fit, fill

class ImageType(Enum):
    CAM = 0
    CAP = 1
    CAD = 2
    CN1 = 3
    CN2 = 4
    SDF = 5
    SDD = 6
    LOGO = 7
    TEXT = 8

class Render(RenderWindow):
    def __init__(self, composition_width: int, composition_height: int, window_width: int, window_height: int, name: str, fullscreen: bool = False, v_sync = False, stretch = False) -> None:
        super().__init__(window_width, window_height, name, fullscreen, v_sync)

        self.input_mutex: Lock = Lock()
        self._images: dict[ImageType, np.ndarray | None] = {}
        self.textures: dict[ImageType, Texture] = {}

        self.width: int = composition_width
        self.height: int = composition_height

        for image_type in ImageType:
            self._images[image_type] = None
            self.textures[image_type] = Texture()

        self.fbos: list[Fbo | SwapFbo] = []
        self.seq_fbo = Fbo()
        self.fbos.append(self.seq_fbo)
        self.noise_fbo = Fbo()
        self.fbos.append(self.noise_fbo)
        self.cam_depth_fbo = SwapFbo()
        self.fbos.append(self.cam_depth_fbo)
        self.camera_fbo = SwapFbo()
        self.fbos.append(self.camera_fbo)
        self.diffusion_fbo = SwapFbo()
        self.fbos.append(self.diffusion_fbo)
        self.dif_depth_fbo = SwapFbo()
        self.fbos.append(self.dif_depth_fbo)
        self.freeze_fbo = Fbo()
        self.fbos.append(self.freeze_fbo)

        self.shaders: list = []
        self.blend_shader = Blend()
        self.shaders.append(self.blend_shader)
        self.blurH_shader = BlurH()
        self.shaders.append(self.blurH_shader)
        self.blurV_shader = BlurV()
        self.shaders.append(self.blurV_shader)
        self.noise_shader = Noise()
        self.shaders.append(self.noise_shader)
        self.noise_blend_shader = NoiseBlend()
        self.shaders.append(self.noise_blend_shader)
        self.noise_simplex_shader = NoiseSimplex()
        self.shaders.append(self.noise_simplex_shader)
        self.noise_simplex_blend_shader = NoiseSimplexBlend()
        self.shaders.append(self.noise_simplex_blend_shader)
        self.flash_in_shader = FlashIn()
        self.shaders.append(self.flash_in_shader)
        self.flash_out_shader = FlashOut()
        self.shaders.append(self.flash_out_shader)
        self.omission_shader = Omission()
        self.shaders.append(self.omission_shader)
        self.hsv_shader = Hsv()
        self.shaders.append(self.hsv_shader)


        self.trail: float = 0.996
        self.dstry_blur: float = 0.1
        self.dstry_bloom: float = 0.7
        self.show_prompt: bool = False

        self.prompt: str = ''
        self.negative: str = ''

        self.freeze: bool = False
        self.prev_freeze: bool = False

        # self.sequencer = Sequencer()


        self.cap_prompt: list[str] = ['test']

    def allocate(self) -> None:
        for fbo in self.fbos: fbo.allocate(self.width, self.height, GL_RGBA32F)
        for shader in self.shaders: shader.allocate(True)

    def reload_shaders(self) -> None :
        for s in self.shaders: s.load(True)


    def draw(self) -> None: # override
        if not self.is_allocated: self.allocate()

        self.update_textures()

        self.draw_camera(ImageType.CAM)
        self.draw_diffusion(ImageType.SDF)


    def update_trails(self, tex: Texture | Fbo | SwapFbo, fbo: SwapFbo, trail: float) -> None:
        self.setView(fbo.width, fbo.height, False, True)
        fbo.swap()
        self.blend_shader.use(fbo.fbo_id,
                              tex.tex_id,
                              fbo.back_tex_id,
                              trail)

    def draw_in_fbo(self, tex: Texture | Fbo | SwapFbo, fbo: Fbo | SwapFbo) -> None:
        self.setView(fbo.width, fbo.height, False, True)
        fbo.begin()
        tex.draw(0, 0, fbo.width, fbo.height)
        fbo.end()

    def draw_camera(self, type: ImageType) -> None:
        self.setView(self.window_width, self.window_height)
        tex: Texture = self.textures[type]
        if tex.width == 0 or tex.height == 0:
            return

        x, y, w, h = fit(tex.width, tex.height, self.window_width, self.window_height / 2)
        self.textures[type].draw(x, y, w, h)

    def draw_diffusion(self, type: ImageType) -> None:
        self.update_trails(self.textures[type], self.diffusion_fbo, self.trail)
        rawFbo: SwapFbo = self.diffusion_fbo

        self.setView(self.window_width, self.window_height)
        x, y, w, h = fit(self.diffusion_fbo.width, self.diffusion_fbo.height, self.window_width, self.window_height / 2)
        self.diffusion_fbo.draw(x, y + h, w, h)

    def update_textures(self) -> None:
        for image_type in ImageType:
            image: np.ndarray | None = self.get_image(image_type)
            texture: Texture = self.textures[image_type]
            if image is not None: texture.set_from_image(image)

    def set_image(self, type: ImageType, image: np.ndarray) -> None :
        with self.input_mutex:
            self._images[type] = image
    def get_image(self, type: ImageType) -> np.ndarray | None :
        with self.input_mutex:
            ret_image: np.ndarray | None = self._images[type]
            self._images[type] = None
            return ret_image

    def set_cam_image(self, image: np.ndarray) -> None :
        self.set_image(ImageType.CAM, image)
    def set_cap_image(self, image: np.ndarray) -> None :
        self.set_image(ImageType.CAP, image)
    def set_cap_depth(self, image: np.ndarray) -> None :
        self.set_image(ImageType.CAD, image)
    def set_cn1_image(self, image: np.ndarray) -> None :
        self.set_image(ImageType.CN1, image)
    def set_cn2_image(self, image: np.ndarray) -> None :
        self.set_image(ImageType.CN2, image)
    def set_sdf_image(self, image: np.ndarray) -> None :
        self.set_image(ImageType.SDF, image)
    def set_sdf_depth(self, image: np.ndarray) -> None :
        self.set_image(ImageType.SDD, image)
