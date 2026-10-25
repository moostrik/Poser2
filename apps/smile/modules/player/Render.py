import logging
from math import ceil
import random
from threading import Lock
import time

import numpy as np
from OpenGL.GL import (GL_COLOR_BUFFER_BIT, GL_RGBA8, GL_RGBA16F, GL_SCISSOR_TEST, glClear, glClearColor,  # type: ignore
                       glDisable, glEnable, glScissor, glViewport)

from modules.credit import FONT, NAME, URL, qr_image, quiet_px, wrap_balanced
from modules.gl import Blit, Fbo, Image, RenderBase, Shader, Style, SwapFbo, Text, WindowSettings, clear_color
from modules.gl.Utils import fit
from modules.player.ClipPool import Clip
from modules.render.shaders import Blend
from modules.player.shaders import BlitRotated

logger = logging.getLogger(__name__)

NUMBER_FONT_PX: int = 320       # the screen's number in a lit cell (the laughter test), in the clip's 512 x 640
NUMBER_COLOR: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)

# the credit cell, in the clip's upright 512 x 640: the QR code and the name under it, as wide as the code
CREDIT_QR_PX: int = 500         # with its white border, which the white cell hides
CREDIT_GAP_PX: int = 64         # between the code and the name
CREDIT_PROBE_PX: int = 100      # the font size the name is measured at, to scale it to the code's width
CREDIT_NAME_COLOR: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)

# a clip's caption, in the clip's upright 512 x 640: the name over the top, the line along the bottom, as on the main screen
CAPTION_PX: int = 44
CAPTION_MARGIN_PX: int = 20
CAPTION_OUTLINE_PX: int = 3
CAPTION_COLOR: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
CAPTION_OUTLINE_COLOR: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
CAPTION_OUTLINE: list[tuple[int, int]] = [(x, y) for x in (-CAPTION_OUTLINE_PX, 0, CAPTION_OUTLINE_PX)
                                          for y in (-CAPTION_OUTLINE_PX, 0, CAPTION_OUTLINE_PX) if x or y]


def lit_cells(lit: dict[str, float], now: float) -> list[str]:
    """The cells lit at `now` (id -> the time its light ends), in the order they were lit; the ones
    that ended are dropped from `lit`."""
    for id in [id for id, until in lit.items() if until <= now]:
        del lit[id]
    return list(lit)


def next_credit_cell(current: str | None, left: str | None, started: list[str]) -> tuple[str | None, str | None]:
    """The credit's cell and the cell it last left, after the cells in `started` started a clip this frame.
    The credit stays for one play: it leaves its cell when the next clip starts there, and takes a cell
    whose clip starts, never the one it just left (with none, it waits for the next start)."""
    if current is not None and current in started:
        current, left = None, current
    if current is None:
        candidates: list[str] = [id for id in started if id != left]
        if candidates:
            current = random.choice(candidates)
    return current, left


class PlayerRender(RenderBase):
    """Grin sequences in a grid of cells, each image turned a quarter, with a trail over the whole grid.
    A cell flashes for one frame when its next sequence starts (the credit card with the credit on, else white), and is white with its screen's
    number while the laughter test lights it (`light`). One cell shows the artist's name and a QR code to
    the project page instead of its clip (`credit`): it takes a cell when a clip starts there, with its white
    frame, and stays for that play, so it moves on as often as the clips change. With `caption` on (SMILE's
    `credit_players` switch, over OSC), every cell shows its clip's caption from the clip's record: the
    artist's name over the top and the visitor's line along the bottom. Keys: f fullscreen (borderless), Esc quit (the
    window manager)."""

    COLUMNS: int = 6
    CELL_WIDTH: int = 640
    CELL_HEIGHT: int = 512

    def __init__(self, window: WindowSettings, ids: list[str], trail: float, credit: bool = False,
                 credit_name: str = NAME, credit_url: str = URL) -> None:
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
        # the credit: drawn upright into a portrait texture like the number, on the cell `_credit_cell`
        self.credit: bool = credit
        self.credit_name: str = credit_name
        self.credit_url: str = credit_url
        self._credit_cell: str | None = None
        self._credit_left: str | None = None
        self._credit_fbo: Fbo = Fbo()
        self._credit_qr: Image = Image('BGR')
        self._credit_quiet: int = 0         # the QR image's white border (px)
        self._credit_name_text: Text = Text()
        # the clips' captions: per cell the name and line of the clip it plays, drawn into a transparent
        # portrait texture when they change, and blitted over the cell turned like the clip
        self.caption: bool = False
        self._captions: dict[str, tuple[str, str]] = {id: ('', '') for id in ids}
        self._caption_drawn: dict[str, tuple[str, str] | None] = {id: None for id in ids}
        self._caption_fbos: dict[str, Fbo] = {id: Fbo() for id in ids}
        self._caption_text: Text = Text()

    # RENDER THREAD
    def allocate(self) -> None:
        self._image_fbo.allocate(self.width, self.height, GL_RGBA16F)
        self._trail_fbo.allocate(self.width, self.height, GL_RGBA16F)
        self._blend.allocate()
        self._rotated.allocate()
        self._number_fbo.allocate(self.CELL_HEIGHT, self.CELL_WIDTH, GL_RGBA8)
        if not self._text.allocate(font_size=NUMBER_FONT_PX):
            logger.warning('No font for the screen numbers: a lit cell is plain white')
        if self.credit:
            self._credit_fbo.allocate(self.CELL_HEIGHT, self.CELL_WIDTH, GL_RGBA8)
            qr = qr_image(self.credit_url, CREDIT_QR_PX)
            self._credit_quiet = quiet_px(qr)
            self._credit_qr.set_image(qr)
            self._allocate_credit_name(qr.shape[1] - 2 * self._credit_quiet)
        for fbo in self._caption_fbos.values():
            fbo.allocate(self.CELL_HEIGHT, self.CELL_WIDTH, GL_RGBA8)
        if not self._caption_text.allocate(FONT, CAPTION_PX):
            logger.warning('No font for the captions: the clips play without')
        Shader.enable_hot_reload()

    def deallocate(self) -> None:
        self._image_fbo.deallocate()
        self._trail_fbo.deallocate()
        self._blend.deallocate()
        self._rotated.deallocate()
        self._number_fbo.deallocate()
        self._text.deallocate()
        self._credit_fbo.deallocate()
        self._credit_qr.deallocate()
        self._credit_name_text.deallocate()
        self._caption_text.deallocate()
        for fbo in self._caption_fbos.values():
            fbo.deallocate()
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
            captions: dict[str, tuple[str, str]] = dict(self._captions) if self.caption else {}
        for image in self._images.values():
            image.update()
        if self.credit:
            self._credit_cell, self._credit_left = next_credit_cell(self._credit_cell, self._credit_left, flashes)

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
        if self.credit:
            self._draw_credit()     # also the flash of every new sequence
        for id, caption in captions.items():
            if caption != self._caption_drawn[id]:
                self._draw_caption(id, *caption)
                self._caption_drawn[id] = caption

        self._trail_fbo.swap()
        self._trail_fbo.begin()
        self._blend.use(self._image_fbo, self._trail_fbo.back_texture, self.trail)
        # a new sequence skips the trail: one white (or credit card) frame, then its first image
        for id in starts:
            glViewport(*self._rects[id])
            self._rotated.use(self._images[id])
        # the clips' captions, over each cell's clip
        if captions:
            Style.set_blend_mode(Style.BlendMode.ALPHA)
            for id, (name, line) in captions.items():
                if name or line:
                    glViewport(*self._rects[id])
                    self._rotated.use(self._caption_fbos[id])
            Style.set_blend_mode(Style.BlendMode.DISABLED)
        # the credit, over its cell's clip
        if self._credit_cell is not None:
            glViewport(*self._rects[self._credit_cell])
            self._rotated.use(self._credit_fbo)
        # the cells the laughter test lights: white with the number, over everything, every frame
        for id in lit:
            glViewport(*self._rects[id])
            self._rotated.use(self._number_fbo)
        # the flash of a new sequence: the credit card, or white without the credit
        if self.credit:
            for id in flashes:
                glViewport(*self._rects[id])
                self._rotated.use(self._credit_fbo)
        else:
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

    def _draw_caption(self, id: str, name: str, line: str) -> None:
        """The cell's caption texture: transparent, the name in even lines over the top and the line along
        the bottom, white with a black edge."""
        fbo: Fbo = self._caption_fbos[id]
        width, height = self.CELL_HEIGHT, self.CELL_WIDTH
        fbo.begin()
        glClearColor(0.0, 0.0, 0.0, 0.0)
        glClear(GL_COLOR_BUFFER_BIT)
        text: Text = self._caption_text
        if text.allocated:
            measure = lambda part: text.measure_text(part)[0]
            _, line_height = text.measure_text("Mg")
            top: list[str] = wrap_balanced(name, measure, width - 2 * CAPTION_MARGIN_PX)
            bottom: list[str] = wrap_balanced(line, measure, width - 2 * CAPTION_MARGIN_PX)
            Style.set_blend_mode(Style.BlendMode.ALPHA)
            for lines, y in ((top, CAPTION_MARGIN_PX), (bottom, height - CAPTION_MARGIN_PX - len(bottom) * line_height)):
                for i, part in enumerate(lines):
                    x: float = (width - measure(part)) / 2
                    for dx, dy in CAPTION_OUTLINE:
                        text.draw_text(x + dx, y + i * line_height + dy, part, CAPTION_OUTLINE_COLOR, width, height)
                    text.draw_text(x, y + i * line_height, part, CAPTION_COLOR, width, height)
            Style.set_blend_mode(Style.BlendMode.DISABLED)
        fbo.end()

    def _allocate_credit_name(self, width: int) -> None:
        """The name's font at the size that makes the name `width` pixels wide."""
        if not self._credit_name_text.allocate(FONT, CREDIT_PROBE_PX):
            logger.warning('No font for the credit: the credit cell has the QR code only')
            return
        probe: int = self._credit_name_text.measure_text(self.credit_name)[0]
        self._credit_name_text.deallocate()
        self._credit_name_text.allocate(FONT, max(1, round(CREDIT_PROBE_PX * width / max(1, probe))))

    def _draw_credit(self) -> None:
        """The credit texture: white, the QR code and the name under it, together in the middle."""
        self._credit_qr.update()
        width, height = self.CELL_HEIGHT, self.CELL_WIDTH
        self._credit_fbo.begin()
        glClearColor(1.0, 1.0, 1.0, 1.0)
        glClear(GL_COLOR_BUFFER_BIT)
        qr: int = self._credit_qr.width if self._credit_qr.allocated else 0
        quiet: int = self._credit_quiet
        name_width, name_height = self._credit_name_text.measure_text(self.credit_name) if self._credit_name_text.allocated else (0, 0)
        code: int = qr - 2 * quiet
        top: float = (height - code - CREDIT_GAP_PX - name_height) / 2       # of the code, from the top
        if qr:
            # the viewport's origin is the bottom left
            glViewport((width - qr) // 2, round(height - top + quiet - qr), qr, qr)
            Blit.use(self._credit_qr)
            glViewport(0, 0, width, height)
        if name_width:
            Style.set_blend_mode(Style.BlendMode.ALPHA)
            self._credit_name_text.draw_text((width - name_width) / 2, top + code + CREDIT_GAP_PX, self.credit_name,
                                             CREDIT_NAME_COLOR, width, height)
            Style.set_blend_mode(Style.BlendMode.DISABLED)
        self._credit_fbo.end()

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

    def set_caption(self, id: str, clip: Clip) -> None:
        """The clip a cell starts: its caption goes on the cell while `caption` is on (any thread)."""
        with self._mutex:
            self._captions[id] = (clip.name, clip.line)

    def light(self, id: str, seconds: float, number: int = 0) -> None:
        """Light a cell white with the screen's `number` (0: none) for `seconds` (the laughter test); an
        unknown id is ignored."""
        if id not in self._rects:
            return
        with self._mutex:
            self._lit[id] = time.time() + seconds
            self._numbers[id] = number
