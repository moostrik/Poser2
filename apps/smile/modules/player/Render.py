import logging
from math import ceil
import random
from threading import Lock
import time

import numpy as np
from OpenGL.GL import (GL_COLOR_BUFFER_BIT, GL_RGBA8, GL_RGBA16F, GL_SCISSOR_TEST, glClear, glClearColor,  # type: ignore
                       glDisable, glEnable, glScissor, glViewport)

from modules.credit import FONT, NAME, URL, qr_image, quiet_px, wrap_balanced
from modules.credit_scroll import LOOP_TIMEOUT, name_strip, scroll_done, scroll_offset
from modules.gl import Blit, Fbo, Image, RenderBase, Shader, Style, SwapFbo, Text, WindowSettings, clear_color
from modules.gl.Utils import fit
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


class Scroll:
    """A pass of the scrolling name on this player (SMILE's `/scroll`): the cells on the path with their
    place along it, moving at `speed` px/s from `started`, with `gap` px of wall between two screens. With
    `ring` (the ring's screens) it goes round without end (`/scroll_loop`) until `expires`, the time SMILE's
    next message is due by."""

    def __init__(self, name: str, speed: float, gap: float, size: int, cells: list[tuple[str, int]], started: float,
                 ring: int = 0, expires: float | None = None) -> None:
        self.name: str = name
        self.speed: float = speed
        self.gap: float = gap
        self.size: int = size
        self.cells: list[tuple[str, int]] = cells
        self.started: float = started
        self.ring: int = ring
        self.expires: float | None = expires

    def renew(self, speed: float, gap: float, size: int, started: float, expires: float) -> None:
        self.speed, self.gap, self.size, self.started, self.expires = speed, gap, size, started, expires

    @property
    def count(self) -> int:
        return max((place for _, place in self.cells), default=-1) + 1

    def travelled(self, now: float) -> float:
        return max(0.0, now - self.started) * self.speed


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
    A cell flashes white for one frame when its next sequence starts, and is white with its screen's
    number while the laughter test lights it (`light`). One cell shows the artist's name and a QR code to
    the project page instead of its clip (`credit`, SMILE's `credit_qr_players` switch over OSC): it takes a cell when a clip starts there, after its white
    frame, and stays for that play, so it moves on as often as the clips change. With `caption` on (SMILE's
    `credit_players` switch, over OSC), every cell shows a caption over its clip, set at every play
    (`set_caption`): the artist's name over the top and a random line along the bottom. From time to time
    SMILE sends a pass of the name scrolling by (`scroll`): letters nearly a screen tall move over the cells
    of one ring of the sphere, each cell showing its part of one long texture (`name_strip`). Keys: f fullscreen (borderless), Esc quit (the
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
        # the clips' captions: per cell a name and a line, drawn into a transparent portrait texture when
        # they change, and blitted over the cell turned like the clip
        self.caption: bool = False
        self._captions: dict[str, tuple[str, str]] = {id: ('', '') for id in ids}
        self._caption_drawn: dict[str, tuple[str, str] | None] = {id: None for id in ids}
        self._caption_fbos: dict[str, Fbo] = {id: Fbo() for id in ids}
        self._caption_text: Text = Text()
        # the scrolling name: the passes under way (a new one does not end the others), a strip texture per
        # name and size in use, and per cell a transparent portrait texture with the parts of the strips
        # on that screen
        self._scrolls: list[Scroll] = []
        self._scroll_strips: dict[tuple[str, int], Image] = {}
        self._scroll_fbos: dict[str, Fbo] = {id: Fbo() for id in ids}

    # RENDER THREAD
    def allocate(self) -> None:
        self._image_fbo.allocate(self.width, self.height, GL_RGBA16F)
        self._trail_fbo.allocate(self.width, self.height, GL_RGBA16F)
        self._blend.allocate()
        self._rotated.allocate()
        self._number_fbo.allocate(self.CELL_HEIGHT, self.CELL_WIDTH, GL_RGBA8)
        if not self._text.allocate(font_size=NUMBER_FONT_PX):
            logger.warning('No font for the screen numbers: a lit cell is plain white')
        # the credit card is ready whether the credit is on or off: SMILE's switch turns it on at any time
        self._credit_fbo.allocate(self.CELL_HEIGHT, self.CELL_WIDTH, GL_RGBA8)
        qr = qr_image(self.credit_url, CREDIT_QR_PX)
        self._credit_quiet = quiet_px(qr)
        self._credit_qr.set_image(qr)
        self._allocate_credit_name(qr.shape[1] - 2 * self._credit_quiet)
        for fbo in self._caption_fbos.values():
            fbo.allocate(self.CELL_HEIGHT, self.CELL_WIDTH, GL_RGBA8)
        if not self._caption_text.allocate(FONT, CAPTION_PX):
            logger.warning('No font for the captions: the clips play without')
        for fbo in self._scroll_fbos.values():
            fbo.allocate(self.CELL_HEIGHT, self.CELL_WIDTH, GL_RGBA8)
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
        for fbo in self._scroll_fbos.values():
            fbo.deallocate()
        for strip in self._scroll_strips.values():
            strip.deallocate()
        self._scroll_strips.clear()
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
            scrolls: list[Scroll] = list(self._scrolls)
        for image in self._images.values():
            image.update()
        if self.credit:
            self._credit_cell, self._credit_left = next_credit_cell(self._credit_cell, self._credit_left, flashes)
        else:
            self._credit_cell = None     # switched off: the card leaves at once, the clip under it goes on

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
            self._draw_credit()
        for id, caption in captions.items():
            if caption != self._caption_drawn[id]:
                self._draw_caption(id, *caption)
                self._caption_drawn[id] = caption
        scrolling: list[str] = self._draw_scrolls(scrolls, time.time()) if scrolls else []

        self._trail_fbo.swap()
        self._trail_fbo.begin()
        self._blend.use(self._image_fbo, self._trail_fbo.back_texture, self.trail)
        # a new sequence skips the trail: one white frame, then its first image
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
        # the scrolling name, over the cells it is on now
        if scrolling:
            Style.set_blend_mode(Style.BlendMode.ALPHA)
            for id in scrolling:
                glViewport(*self._rects[id])
                self._rotated.use(self._scroll_fbos[id])
            Style.set_blend_mode(Style.BlendMode.DISABLED)
        # the cells the laughter test lights: white with the number, over everything, every frame
        for id in lit:
            glViewport(*self._rects[id])
            self._rotated.use(self._number_fbo)
        # the flash of a new sequence: one white frame
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

    def _draw_scrolls(self, scrolls: list[Scroll], now: float) -> list[str]:
        """The cells the scrolling name is on at `now`, each cell's texture filled with its part of every
        pass that crosses it; a pass whose name has left its last screen is dropped, and so is a strip no
        pass uses any more."""
        width, height = self.CELL_HEIGHT, self.CELL_WIDTH     # the screen upright: its width is the cell's height
        parts: dict[str, list[tuple[Image, float]]] = {}     # per cell: the strips on it with their offsets
        keys: set[tuple[str, int]] = set()
        for scroll in scrolls:
            key: tuple[str, int] = (scroll.name, scroll.size)
            keys.add(key)
            strip: Image | None = self._scroll_strips.get(key)
            if strip is None:
                strip = self._scroll_strips[key] = Image('BGR')
                strip.set_image(name_strip(scroll.name, height, scroll.size))
            strip.update()
            if not strip.allocated:
                continue
            travelled: float = scroll.travelled(now)
            ended: bool = (now > scroll.expires) if scroll.expires is not None else \
                scroll_done(travelled, scroll.count, strip.width, scroll.gap, width)
            if ended:
                with self._mutex:
                    if scroll in self._scrolls:
                        self._scrolls.remove(scroll)
                continue
            # a loop: the distance round the ring, the strip drawn where it is on this lap and where the next
            # lap's copy enters at the end of the ring
            lap: float = scroll.ring * (width + scroll.gap)
            travels: list[float] = [travelled % lap, travelled % lap + lap] if lap > 0 else [travelled]
            for id, place in scroll.cells:
                for t in travels:
                    offset: float = scroll_offset(place, t, scroll.gap, width)
                    if offset < strip.width and offset + width > 0:
                        parts.setdefault(id, []).append((strip, offset))
        for key in [key for key in self._scroll_strips if key not in keys]:
            self._scroll_strips.pop(key).deallocate()
        for id, strips in parts.items():
            fbo: Fbo = self._scroll_fbos[id]
            fbo.begin()
            glClearColor(0.0, 0.0, 0.0, 0.0)
            glClear(GL_COLOR_BUFFER_BIT)
            for index, (strip, offset) in enumerate(strips):
                # the strip's x = offset at the screen's left edge: the strip drawn whole, shifted left by it;
                # the first pass straight into the cleared texture, another over it
                Style.set_blend_mode(Style.BlendMode.ALPHA if index else Style.BlendMode.DISABLED)
                glViewport(-round(offset), 0, strip.width, height)
                Blit.use(strip)
            Style.set_blend_mode(Style.BlendMode.DISABLED)
            glViewport(0, 0, width, height)
            fbo.end()
        return list(parts)

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

    def set_caption(self, id: str, name: str, line: str) -> None:
        """The caption on a cell from now on, shown while `caption` is on (any thread); an unknown id is ignored."""
        if id not in self._rects:
            return
        with self._mutex:
            self._captions[id] = (name, line)

    def scroll(self, name: str, speed: float, gap: float, size: int, cells: list[tuple[str, int]]) -> None:
        """A pass of the scrolling name from now (any thread): `cells` are the ids on the path with their
        place along it, 0 where the name enters; it moves at `speed` px/s with `gap` px between two screens
        and `size` px above and below the letters. A pass under way goes on beside it."""
        cells = [(id, place) for id, place in cells if id in self._rects]
        if not cells:
            return
        with self._mutex:
            self._scrolls.append(Scroll(name, speed, gap, size, cells, time.time()))

    def scroll_loop(self, name: str, speed: float, gap: float, size: int, ring: int, started: float,
                    cells: list[tuple[str, int]]) -> bool:
        """A word going round a ring of `ring` screens without end (any thread, SMILE sends it every second):
        `started` is when it began by the PC's clock, which SMILE and the player share, so the position comes
        from this player's own clock and a resend moves nothing, while a player that starts late joins in step.
        The loop ends when no message renews it for `LOOP_TIMEOUT`. True for a loop that is new."""
        cells = [(id, place) for id, place in cells if id in self._rects]
        if not cells or ring <= 0:
            return False
        now: float = time.time()
        with self._mutex:
            for scroll in self._scrolls:
                if scroll.ring == ring and scroll.name == name and scroll.cells == cells:
                    scroll.renew(speed, gap, size, started, now + LOOP_TIMEOUT)
                    return False
            self._scrolls.append(Scroll(name, speed, gap, size, cells, started, ring, now + LOOP_TIMEOUT))
        return True

    def light(self, id: str, seconds: float, number: int = 0) -> None:
        """Light a cell white with the screen's `number` (0: none) for `seconds` (the laughter test); an
        unknown id is ignored."""
        if id not in self._rects:
            return
        with self._mutex:
            self._lit[id] = time.time() + seconds
            self._numbers[id] = number
