import logging
from math import ceil
from threading import Lock
import time

import numpy as np
from OpenGL.GL import (GL_COLOR_BUFFER_BIT, GL_RGBA8, GL_RGBA16F, GL_SCISSOR_TEST, glClear, glClearColor,  # type: ignore
                       glDisable, glEnable, glScissor, glViewport)

from modules.gl import Blit, Fbo, Image, RenderBase, Shader, Style, SwapFbo, Text, WindowSettings, clear_color
from modules.gl.Utils import fit
from modules.render.shaders import Blend
from modules.player.shaders import BlitRotated

logger = logging.getLogger(__name__)

NUMBER_FONT_PX: int = 320       # the screen's number in a lit cell (the laughter test), in the clip's 512 x 640
NUMBER_COLOR: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)


def lit_cells(lit: dict[str, float], now: float) -> list[str]:
    """The cells lit at `now` (id -> the time its light ends), in the order they were lit; the ones
    that ended are dropped from `lit`."""
    for id in [id for id, until in lit.items() if until <= now]:
        del lit[id]
    return list(lit)


class PlayerRender(RenderBase):
    """Grin sequences in a grid of cells, each image turned a quarter, with a trail over the whole grid.
    A cell flashes white for one frame when its next sequence starts, and is white with its screen's
    number while the laughter test lights it (`light`). Keys: f fullscreen (borderless), Esc quit (the
    window manager)."""

    COLUMNS: int = 6
    CELL_WIDTH: int = 640
    CELL_HEIGHT: int = 512

    def __init__(self, window: WindowSettings, ids: list[str], trail: float) -> None:
        super().__init__(window)
        # exclusive fullscreen next to SMILE and the other player chokes the GPU: the generation times out
        self.window_manager.exclusive_fullscreen = False
        self.trail: float = trail

        self.width: int = self.COLUMNS * self.CELL_WIDTH
        self.height: int = ceil(len(ids) / self.COLUMNS) * self.CELL_HEIGHT

        self._ids: list[str] = ids
        self._images: dict[str, Image] = {id: Image('BGR') for id in ids}
        # 1: first image of a sequence arrived (flash white), 2: flashed (draw the image directly), 0: playing
        self._initials: dict[str, int] = {id: 0 for id in ids}
        self._lit: dict[str, float] = {}       # cells the laughter test lights: id -> when the light ends
        self._numbers: dict[str, int] = {}     # and the screen number each shows (0: none)
        self._mutex: Lock = Lock()

        # cell rectangles in framebuffer pixels (origin bottom left), cell 0 at the top left
        self._rects: dict[str, tuple[int, int, int, int]] = {}
        for i, id in enumerate(ids):
            column: int = i % self.COLUMNS
            row: int = i // self.COLUMNS
            self._rects[id] = (column * self.CELL_WIDTH, self.height - (row + 1) * self.CELL_HEIGHT,
                               self.CELL_WIDTH, self.CELL_HEIGHT)

        self._image_fbo: Fbo = Fbo()
        self._trail_fbo: SwapFbo = SwapFbo()
        self._blend: Blend = Blend()
        self._rotated: BlitRotated = BlitRotated()
        # the lit cell's number: drawn upright into a portrait texture like a clip, blitted turned like one
        self._number_fbo: Fbo = Fbo()
        self._text: Text = Text()

    # RENDER THREAD
    def allocate(self) -> None:
        self._image_fbo.allocate(self.width, self.height, GL_RGBA16F)
        self._trail_fbo.allocate(self.width, self.height, GL_RGBA16F)
        self._blend.allocate()
        self._rotated.allocate()
        self._number_fbo.allocate(self.CELL_HEIGHT, self.CELL_WIDTH, GL_RGBA8)
        if not self._text.allocate(font_size=NUMBER_FONT_PX):
            logger.warning('No font for the screen numbers: a lit cell is plain white')
        Shader.enable_hot_reload()

    def deallocate(self) -> None:
        self._image_fbo.deallocate()
        self._trail_fbo.deallocate()
        self._blend.deallocate()
        self._rotated.deallocate()
        self._number_fbo.deallocate()
        self._text.deallocate()
        for image in self._images.values():
            image.deallocate()

    def update(self) -> None:
        self._notify_update()
        self._blend.reload()
        self._rotated.reload()

        flashes: list[str] = []
        starts: list[str] = []
        with self._mutex:
            for id in self._ids:
                if self._initials[id] == 2:
                    self._initials[id] = 0
                    starts.append(id)
                elif self._initials[id] == 1:
                    self._initials[id] = 2
                    flashes.append(id)
            lit: list[str] = lit_cells(self._lit, time.time())
            numbers: dict[str, int] = {id: self._numbers.get(id, 0) for id in lit}
            self._numbers = numbers
        for image in self._images.values():
            image.update()

        Style.reset_state()
        Style.set_blend_mode(Style.BlendMode.DISABLED)

        self._image_fbo.begin()
        for id in self._ids:
            glViewport(*self._rects[id])
            self._rotated.use(self._images[id])
        self._image_fbo.end()

        # the number of the lit cell, rendered before the trail pass and blitted during it (the test lights
        # one cell at a time; with more, all show the last one's number)
        if lit:
            self._draw_number(numbers[lit[-1]])

        self._trail_fbo.swap()
        self._trail_fbo.begin()
        self._blend.use(self._image_fbo, self._trail_fbo.back_texture, self.trail)
        # a new sequence skips the trail: one white frame, then its first image
        for id in starts:
            glViewport(*self._rects[id])
            self._rotated.use(self._images[id])
        # the cells the laughter test lights: white with the number, over everything, every frame
        for id in lit:
            glViewport(*self._rects[id])
            self._rotated.use(self._number_fbo)
        # the white frame of a new sequence
        glEnable(GL_SCISSOR_TEST)
        glClearColor(1.0, 1.0, 1.0, 1.0)
        for id in flashes:
            glScissor(*self._rects[id])
            glClear(GL_COLOR_BUFFER_BIT)
        glDisable(GL_SCISSOR_TEST)
        self._trail_fbo.end()

    def _draw_number(self, number: int) -> None:
        """The number texture: white, with the number centred in black (none for 0, or without a font)."""
        self._number_fbo.begin()
        glClearColor(1.0, 1.0, 1.0, 1.0)
        glClear(GL_COLOR_BUFFER_BIT)
        if number > 0 and self._text.allocated:
            text: str = str(number)
            width, height = self._text.measure_text(text)
            Style.set_blend_mode(Style.BlendMode.ALPHA)
            self._text.draw_text((self.CELL_HEIGHT - width) / 2, (self.CELL_WIDTH - height) / 2, text, NUMBER_COLOR,
                                 self.CELL_HEIGHT, self.CELL_WIDTH)
            Style.set_blend_mode(Style.BlendMode.DISABLED)
        self._number_fbo.end()

    def draw_main(self, width: int, height: int) -> None:
        clear_color(0, 0, 0, 1)
        Style.reset_state()
        x, y, w, h = fit(self.width, self.height, width, height)
        glViewport(int(x), int(y), int(w), int(h))
        Blit.use(self._trail_fbo.texture)

    def draw_secondary(self, monitor_id: int, width: int, height: int) -> None:
        pass

    def on_main_window_resize(self, width: int, height: int) -> None:
        pass

    # ANY THREAD
    def set_image(self, id: str, image: np.ndarray, initial: bool) -> None:
        self._images[id].set_image(image)
        with self._mutex:
            self._initials[id] = int(initial)

    def light(self, id: str, seconds: float, number: int = 0) -> None:
        """Light a cell white with the screen's `number` (0: none) for `seconds` (the laughter test); an
        unknown id is ignored."""
        if id not in self._rects:
            return
        with self._mutex:
            self._lit[id] = time.time() + seconds
            self._numbers[id] = number
