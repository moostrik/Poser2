from OpenGL.GL import * # type: ignore
import OpenGL.GLUT as glut
from threading import Lock
import cv2
import numpy as np
from enum import Enum
import pytweening as easing
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

from modules.render.Sequencer import Sequencer, SequenceType, SeqenceMessage
from modules.render.settings import RenderSettings, SequencerSettings
from modules.sd.WebUi import WebUiMessage, WebUiState

from PIL import Image, ImageDraw, ImageFont

font: ImageFont.FreeTypeFont = ImageFont.truetype("arial.ttf", 58)

def draw_text(text_list: list[str], width, height, text_color=(0, 0, 0), background_color=(255, 255, 255)) -> np.ndarray:
    image: Image.Image = Image.new("RGB", (width, height), background_color)
    draw: ImageDraw.ImageDraw = ImageDraw.Draw(image)
    lines = len(text_list)
    if lines > 0:
        step: float = height / float(lines + 1)
        cntr: int = 0
        for text in text_list:
            text_width, text_height = draw.textbbox((0, 0), text, font=font)[2:]
            x = (width - text_width) // 2
            y = step * (cntr + 0.4) + text_height * 0.5

            text_position = (x, y)
            draw.text(text_position, text, fill=text_color, font=font)
            cntr += 1

    return np.array(image)


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
    def __init__(self, settings: RenderSettings, sequencer_settings: SequencerSettings, composition_width: int, composition_height: int, window_width: int, window_height: int, name: str, v_sync = False, stretch = False) -> None:
        super().__init__(window_width, window_height, name, settings.fullscreen, v_sync)
        self.settings: RenderSettings = settings

        self.input_mutex: Lock = Lock()
        self._sequence_message: SeqenceMessage
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

        self.stretch: bool = stretch

        self.prompt: str = ''
        self.negative: str = ''

        self.prev_freeze: bool = False

        self.sequencer = Sequencer(sequencer_settings)
        self.settings.bind(RenderSettings.fullscreen, self.set_fullscreen)

        self.logo_folder_path = 'logos'
        self.logo_files = [f for f in os.listdir(self.logo_folder_path) if f.endswith('.jpg')]
        self.logo_reload: bool = True
        print (len(self.logo_files), 'logo files')

        self.cap_prompt: list[str] = ['test']

    def allocate(self) -> None:
        for fbo in self.fbos: fbo.allocate(self.width, self.height, GL_RGBA32F)
        for shader in self.shaders: shader.allocate(True)

    def reload_shaders(self) -> None :
        for s in self.shaders: s.load(True)

    def stop(self) -> None: # override
        self.settings.unbind(RenderSettings.fullscreen, self.set_fullscreen)
        self.sequencer.close()
        super().stop()

    def keyboard_callback(self, key, x, y) -> None: # override
        super().keyboard_callback(key, x, y)
        if key == b'f' or key == b'F':
            self.settings.fullscreen = not self.settings.fullscreen
        if key == b'r' or key == b'R':
            self.reload_shaders()
        if key == b'\r':
            self.sequencer.start_sequence()

    def _trail(self) -> float:
        return self.settings.trail if self.settings.trail_active else 0.0

    def load_logo(self) -> None :
        selected_file: str = random.choice(self.logo_files)
        file_path: str = os.path.join(self.logo_folder_path, selected_file)
        image: np.ndarray = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
        if image is None:
            print("Failed to load the image.")
        else:
            # Resize the image to 3840x2160
            resized_image: np.ndarray = cv2.resize(image, (3840, 2160))
            self.textures[ImageType.LOGO].set_from_image(image) # type: ignore

        # self._images[ImageType.LOGO] = imread('files/logo.jpg')
        # h, w, c =  self._images[ImageType.LOGO].shape
        # if w != self.width or h != self.height:
        #     cv2.resize(self._images[ImageType.LOGO] , self._images[ImageType.LOGO] , cv2.Size(self.width, self.height), 0, 0, cv2.INTER_CUBIC)
        # self.textures[ImageType.LOGO].set_from_image(self._images[ImageType.LOGO])

    def draw(self) -> None: # override
        if not self.is_allocated: self.allocate()

        self.update_textures()
        self.sequencer.update()

        self.draw_sequence(self.sequencer.message)
        if self.settings.show_camera:
            self.draw_texture(ImageType.CAM)

    def draw_sequence(self, message: SeqenceMessage) -> None:
        drawFbo: SwapFbo | Fbo = self.seq_fbo


        self.setView(self.width, self.height, False, True)

        if message.sequence_type == SequenceType.BLOOM :
            self.hsv_shader.use(self.camera_fbo.fbo_id, self.textures[ImageType.CAM].tex_id, 1.0, 0.0, 1.0)

            if message.initial:
                self.draw_in_fbo(self.textures[ImageType.SDD], self.dif_depth_fbo)
                self.dif_depth_fbo.swap()
                self.blurH_shader.use(self.dif_depth_fbo.fbo_id,
                                      self.dif_depth_fbo.back_tex_id,
                                      1, self.width, self.height)
                self.dif_depth_fbo.swap()
                self.blurV_shader.use(self.dif_depth_fbo.fbo_id,
                                      self.dif_depth_fbo.back_tex_id,
                                      1, self.width, self.height)

            self.omission_shader.use(self.seq_fbo.fbo_id,
                                     self.diffusion_fbo.tex_id,
                                     self.camera_fbo.tex_id,
                                     self.dif_depth_fbo.tex_id,
                                     message.value_1 * self.settings.destroy_bloom,
                                     message.value_2,
                                     self.width, self.height)

            drawFbo = self.seq_fbo

        if message.sequence_type == SequenceType.FADE :
            self.hsv_shader.use(self.camera_fbo.fbo_id, self.textures[ImageType.CAM].tex_id, 1.0, 0.0, 1.0)

            self.diffusion_fbo.swap()
            self.blurH_shader.use(self.diffusion_fbo.fbo_id,
                                  self.diffusion_fbo.back_tex_id,
                                  self.settings.destroy_blur, self.width, self.height)
            self.diffusion_fbo.swap()
            self.blurV_shader.use(self.diffusion_fbo.fbo_id,
                                  self.diffusion_fbo.back_tex_id,
                                  self.settings.destroy_blur, self.width, self.height)

            self.omission_shader.use(self.seq_fbo.fbo_id,
                                     self.diffusion_fbo.tex_id,
                                     self.camera_fbo.tex_id,
                                     self.dif_depth_fbo.tex_id,
                                     message.value_2 * self.settings.destroy_bloom,
                                     message.value_1,
                                     self.width, self.height)

            drawFbo = self.seq_fbo

        if message.sequence_type == SequenceType.CAMERA_SHOW:
            self.hsv_shader.use(self.camera_fbo.fbo_id, self.textures[ImageType.CAM].tex_id, 1.0, 0.0, 1.0)
            drawFbo = self.camera_fbo
            if self.logo_reload:
                self.load_logo()
                self.logo_reload = False
        else:
            self.logo_reload = True

        if message.sequence_type == SequenceType.CAMERA_DETECT:
            self.hsv_shader.use(self.camera_fbo.fbo_id, self.textures[ImageType.CAM].tex_id, 1.0, message.value_1, 1.0)
            drawFbo = self.camera_fbo


        if message.sequence_type == SequenceType.CAMERA_REC :
            self.draw_in_fbo(self.textures[ImageType.CAM], self.camera_fbo)
            drawFbo = self.camera_fbo


        if message.sequence_type == SequenceType.FLASH:
            self.draw_in_fbo(self.textures[ImageType.CAM], self.camera_fbo)
            self.flash_in_shader.use(self.seq_fbo.fbo_id,
                                  self.camera_fbo.tex_id,
                                  self.textures[ImageType.LOGO].tex_id,
                                  message.value_1)
            drawFbo = self.seq_fbo

        if message.sequence_type == SequenceType.CAPTURE or message.sequence_type == SequenceType.ASK:
            self.flash_in_shader.use(self.seq_fbo.fbo_id,
                                  self.camera_fbo.tex_id,
                                  self.textures[ImageType.LOGO].tex_id,
                                  message.value_1)
            drawFbo = self.seq_fbo

        if message.sequence_type == SequenceType.CONTROL_0:
            if message.initial:
                self.draw_in_fbo(self.textures[ImageType.CAP], self.camera_fbo)
                self.draw_in_fbo(self.textures[ImageType.CAD], self.cam_depth_fbo)
                self.cam_depth_fbo.swap()
                self.blurH_shader.use(self.cam_depth_fbo.fbo_id,
                                      self.cam_depth_fbo.back_tex_id,
                                      1, self.width, self.height)
                self.cam_depth_fbo.swap()
                self.blurV_shader.use(self.cam_depth_fbo.fbo_id,
                                      self.cam_depth_fbo.back_tex_id,
                                      1, self.width, self.height)

                self.draw_in_fbo(self.camera_fbo, self.diffusion_fbo)

            if message.value_1 <= 1.0:
                self.flash_out_shader.use(self.seq_fbo.fbo_id,
                                          self.diffusion_fbo.tex_id,
                                          self.textures[ImageType.LOGO].tex_id,
                                          self.cam_depth_fbo.tex_id,
                                          1.0 - message.value_1,
                                          self.width, self.height)
            drawFbo = self.seq_fbo

        if message.sequence_type == SequenceType.CONTROL_1 | message.sequence_type == SequenceType.CONTROL_2:
            if message.value_1 <= 1.0:
                self.flash_out_shader.use(self.seq_fbo.fbo_id,
                                          self.diffusion_fbo.tex_id,
                                          self.textures[ImageType.LOGO].tex_id,
                                          self.cam_depth_fbo.tex_id,
                                          1.0 - message.value_1,
                                          self.width, self.height)
            drawFbo = self.seq_fbo

        if message.sequence_type == SequenceType.DIFFUSION:
            self.update_trails(self.textures[ImageType.SDF], self.diffusion_fbo, self._trail())
            drawFbo = self.diffusion_fbo

        if message.sequence_type == SequenceType.DIFF_LOOP:
            self.update_trails(self.textures[ImageType.SDF], self.diffusion_fbo, self._trail())
            drawFbo = self.diffusion_fbo

        if message.sequence_type == SequenceType.TIMEOUT:
            self.update_trails(self.textures[ImageType.SDF], self.diffusion_fbo, self._trail())
            drawFbo = self.diffusion_fbo

        if message.sequence_type == SequenceType.RESET:
            self.update_trails(self.textures[ImageType.SDF], self.diffusion_fbo, self._trail())
            drawFbo = self.diffusion_fbo

        if message.sequence_type == SequenceType.NONE:
            self.update_trails(self.textures[ImageType.SDF], self.diffusion_fbo, self._trail())
            drawFbo = self.diffusion_fbo

        if self.settings.freeze is True:
            if self.settings.freeze is not self.prev_freeze:
                self.draw_in_fbo(drawFbo, self.freeze_fbo)
            drawFbo = self.freeze_fbo
        self.prev_freeze = self.settings.freeze


        self.setView(self.window_width, self.window_height)
        if self.stretch:
            drawFbo.draw(0, 0, self.window_width, self.window_height)
        else:
            x, y, w, h = fit(self.width, self.height, self.window_width, self.window_height)
            drawFbo.draw(x, y, w, h)

        if self.settings.show_prompt:
            if message.sequence_type == SequenceType.DIFF_LOOP or message.sequence_type == SequenceType.TIMEOUT:
                self.draw_string_field(self.window_width, self.window_height)

    def draw_string_field(self, d_w: int, d_h:int) -> None:
        w: int = int(d_w * 1.0)
        h: int = int(d_w * 0.2)
        x: int = int((d_w - w) / 2)
        y: int = int(d_h - h)
        self.textures[ImageType.TEXT].draw(x, y, w, h)


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

    def draw_texture(self, type: ImageType) -> None:
        self.setView(self.window_width, self.window_height)
        tex: Texture = self.textures[type]
        if tex.width == 0 or tex.height == 0:
            return
        if self.stretch:
            self.textures[type].draw(0,0, self.window_width, self.window_height)
        else:
            x, y, w, h = fit(tex.width, tex.height, self.window_width, self.window_height)
            self.textures[type].draw(x, y, w, h)

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

    def set_cap_prompt(self, prompt:list[str]) -> None:
        self.cap_prompt = prompt
        image: np.ndarray = draw_text(prompt, 1920, 384)
        self.set_image(ImageType.TEXT, image)

    def set_diffusion_message(self, message: WebUiMessage) -> None:
        if message.state == WebUiState.CAM_DEPTH:
            if message.image is not None:
                self.set_cap_depth(message.image)
            self.sequencer.set_sequence(SequenceType.CONTROL_0)
        if message.state == WebUiState.CN_POSE:
            if message.image is not None:
                self.set_cn1_image(message.image)
            self.sequencer.set_sequence(SequenceType.CONTROL_1)
        elif message.state == WebUiState.CN_DETAIL:
            if message.image is not None:
                self.set_cn2_image(message.image)
            self.sequencer.set_sequence(SequenceType.CONTROL_2)
        elif message.state == WebUiState.DIFFUSION:
            if message.image is not None:
                self.set_sdf_image(message.image)
            self.sequencer.set_sequence(SequenceType.DIFFUSION)
            self.sequencer.set_diffusion_progress(message.progress)
        elif message.state == WebUiState.DIFF_LOOP:
            if message.image is not None:
                self.set_sdf_image(message.image)
            self.sequencer.set_sequence(SequenceType.DIFF_LOOP)
            self.sequencer.set_diffusion_progress(message.progress)
        elif message.state == WebUiState.IDLE:
            if message.image is not None:
                self.set_sdf_depth(message.image)
            self.sequencer.set_sequence(SequenceType.TIMEOUT)

        if message.prompt != '':    self.prompt = message.prompt
        if message.negative != '':  self.negative = message.negative