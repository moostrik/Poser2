# Standard library imports
from typing import Tuple
import numpy as np
import math

# Third-party imports
from OpenGL.GL import * # type: ignore

# Local application imports
from modules.correlation.PairCorrelationStream import PairCorrelationStreamData
from modules.gl.Image import Image
from modules.gl.Fbo import Fbo, SwapFbo
from modules.gl.Text import draw_box_string, text_init

from modules.render.DataManager import DataManager
from modules.render.renders.BaseRender import BaseRender, Rect

from modules.gl.shaders.HD_Sync import HD_Sync
from modules.gl.shaders.NoiseSimplex import NoiseSimplex

from modules.utils.HotReloadMethods import HotReloadMethods

class SynchronyCam(BaseRender):

    shader = HD_Sync()
    noise_shader = NoiseSimplex()

    def __init__(self, data: DataManager, cam_id: int) -> None:
        self.data: DataManager = data
        self.cam_id: int = cam_id
        self.fbo: Fbo = Fbo()
        self.noise_fbo: Fbo = Fbo()
        self.cam_image: Image = Image()

        self.movement: float = 0.0
        text_init()
        hot_reload = HotReloadMethods(self.__class__, True, True)

    def allocate(self, width: int, height: int, internal_format: int) -> None:
        self.fbo.allocate(width, height, internal_format)
        self.noise_fbo.allocate(width, height, internal_format)
        if not SynchronyCam.shader.allocated:
            SynchronyCam.shader.allocate(True)
        if not SynchronyCam.noise_shader.allocated:
            SynchronyCam.noise_shader.allocate(True)

    def deallocate(self) -> None:
        self.fbo.deallocate()
        self.noise_fbo.deallocate()
        if SynchronyCam.shader.allocated:
            SynchronyCam.shader.deallocate()
        if SynchronyCam.noise_shader.allocated:
            SynchronyCam.noise_shader.deallocate()

    def draw(self, rect: Rect) -> None:
        self.fbo.draw(rect.x, rect.y, rect.width, rect.height)

    def update(self, cam_fbos: list[Fbo], movement: float) -> None:
        if len(cam_fbos) < 3:
            print("SynchronyCam: Not enough camera FBOs to render synchrony.")
            return

        key: int = self.cam_id
        other_keys: list[int] = [i for i in range(len(cam_fbos)) if i != key]

        syncs: PairCorrelationStreamData | None = self.data.get_correlation_streams(False, self.key())


        score_1: float = 0.0
        score_2: float = 0.0


        if syncs is not None:
            scores: dict[int, float] = syncs.get_correlation_for_key(key)

            score_1 = math.pow(scores.get(other_keys[0], 0.0), 1.5)
            score_2 = math.pow(scores.get(other_keys[1], 0.0), 1.5)

        main_fbo = cam_fbos[key]
        other_fbo_1 = cam_fbos[other_keys[0]]
        other_fbo_2 = cam_fbos[other_keys[1]]

        BaseRender.setView(self.fbo.width, self.fbo.height)

        if not SynchronyCam.shader.allocated:
            SynchronyCam.shader.allocate(True)
        if not SynchronyCam.noise_shader.allocated:
            SynchronyCam.noise_shader.allocate(True)

        # glColor4f(1.0, 1.0, 1.0, 1.0)
        # self.fbo.begin()
        # glClearColor(1.0, 0.0, 0.0, 1.0)
        # glClear(GL_COLOR_BUFFER_BIT)
        # self.fbo.end()

        SynchronyCam.noise_shader.use(self.noise_fbo.fbo_id, 40, 200, self.fbo.width, self.fbo.height)
        SynchronyCam.shader.use(self.fbo.fbo_id, main_fbo.tex_id, other_fbo_1.tex_id, other_fbo_2.tex_id, self.noise_fbo.tex_id, score_1, score_2)





