"""The artist's name going round the sphere on the walls while the installation is idle: each word of it (the
first name, the surname) circles its own ring of screens, over the playing clips, like a ticker, letters most of
a screen tall, one word per lap so a screen is clear between two laps. A word enters through the side of the
first screen and, when it has to go, runs out through the side of the screen its head is on. SMILE plans the loops
(`CreditScroller`, in the render loop) and hands each wall, every second, the cells of its own on each ring with
their place round it and the time the loop began; the walls know nothing of the sphere and draw the word from
one long texture (`name_strip`) by the clock. No GL here: the geometry is tested without one."""

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

import cv2
import freetype
import numpy as np

from . import sphere
from .credit import FONT, render_text

if TYPE_CHECKING:
    from .render import RenderSettings

logger = logging.getLogger(__name__)

# the sphere and the wiring of its screens to the walls' cells are in `apps.smile.sphere`; here a screen
# is counted from 0 (the sphere's number minus 1)
ROWS: int = sphere.HEIGHTS
SCREENS: int = sphere.SCREENS
CELLS: int = sphere.CELLS
WALLS: int = sphere.WALLS
SCREEN_PX: int = 512            # a screen's width, in the wall's pixels (the clip's upright 512 x 640)
SCREEN_HEIGHT_PX: int = 640

DEFAULT_ROWS: str = "1,3"       # the rings with 12 screens, leaving the middle one to the clips
DEFAULT_GAP_PX: int = 288       # the wall between two screens, found on site 2026-10-25
SEND_EVERY: float = 1.0         # a loop is renewed every second
TIMEOUT: float = 3.0            # the walls drop a loop that was not renewed for this long (the switch off, the test on)


def row_screens(row: int) -> list[int]:
    """The screens (from 0) of ring `row` in order round the sphere: 12 for rows 1 to 3, 6 for 0 and 4."""
    return [screen - 1 for screen in sphere.row_screens(row)]


def screen_cell(screen: int) -> tuple[int, int]:
    """The wall (0 or 1) and its cell of a screen (from 0), by the wiring."""
    return sphere.cell_of_screen(screen + 1)


def parse_rows(text: str) -> list[int]:
    """The rows of the panel's text ("1,3"; also spaces), each once, in 0 to 4; junk is left out."""
    rows: list[int] = []
    for part in text.replace(',', ' ').split():
        try:
            row: int = int(part)
        except ValueError:
            continue
        if 0 <= row < ROWS and row not in rows:
            rows.append(row)
    return rows


@dataclass(frozen=True)
class RingPath:
    """The path of a word round ring `row`: all its screens in the order the word reaches them, from the rib at
    the camera, down the rib numbers (the ribs count to the camera's left, the visitor's right, and a ticker
    moves to the visitor's left: checked on site 2026-10-25), the other way round with `reverse`."""
    row: int
    reverse: bool = False

    @property
    def screens(self) -> list[int]:
        ring: list[int] = row_screens(self.row)
        step: int = 1 if self.reverse else -1
        return [ring[(step * i) % len(ring)] for i in range(len(ring))]

    @property
    def count(self) -> int:
        return len(row_screens(self.row))

    def cells(self, wall: int) -> list[tuple[int, int]]:
        """The cells of `wall` on the ring, each with its place round it (0: the rib at the camera)."""
        return [(cell, place) for place, screen in enumerate(self.screens)
                for w, cell in [screen_cell(screen)] if w == wall]


def plan_loops(rows: list[int], words: list[str], reverse: bool) -> list[tuple[str, RingPath]]:
    """Each of `rows` gets a word of the name going round it, the first word on the first row and so on (the
    words again when there are more rows)."""
    if not words:
        return []
    return [(words[index % len(words)], RingPath(row, reverse)) for index, row in enumerate(rows)]


def scroll_offset(place: int, travelled: float, gap_px: float, screen_px: int = SCREEN_PX) -> float:
    """Where the screen at `place` is in the word's texture after the word moved `travelled` px: the texture
    x at the screen's left edge. The word reads left to right and moves to the left, as a ticker, so it
    enters the first screen at its right edge and the next screen lies beyond the first one's left edge, a
    `gap_px` wide wall between them."""
    return travelled - screen_px - place * (screen_px + gap_px)


def scroll_copies(place: int, ring: int, travelled: float, gap_px: float, strip_px: int, last: int | None = None,
                  screen_px: int = SCREEN_PX) -> list[float]:
    """The offsets (`scroll_offset`) of the word on the screen at `place` of a ring of `ring` screens: this lap's
    and the tail of the lap before. `travelled` is not wrapped, so a screen is counted on from lap to lap: one
    before the first lap has nothing on it, so the word enters at the first screen's side, and neither has one
    past `last`, so the word runs out at the left side of that screen."""
    laps: int = int(travelled // (ring * (screen_px + gap_px)))
    offsets: list[float] = []
    for lap in (laps, laps - 1):
        round_place: int = place + lap * ring
        if round_place < 0 or (last is not None and round_place > last):
            continue
        offset: float = scroll_offset(round_place, travelled, gap_px, screen_px)
        if offset < strip_px and offset + screen_px > 0:
            offsets.append(offset)
    return offsets


def last_place(travelled: float, gap_px: float, screen_px: int = SCREEN_PX) -> int:
    """The screen, counted on from lap to lap, that the head of a word has reached: where it leaves."""
    return int(travelled // (screen_px + gap_px))


def scroll_gone(last: int, travelled: float, gap_px: float, strip_px: int, screen_px: int = SCREEN_PX) -> bool:
    """Whether the tail of a leaving word has passed the left side of its `last` screen."""
    return scroll_offset(last, travelled, gap_px, screen_px) >= strip_px


def name_strip(text: str, height: int, size: int, font: str = FONT, outline_ratio: float = 0.025,
               probe_px: int = 200) -> np.ndarray:
    """The word as one long transparent BGRA image `height` px tall, white letters with a black edge, their
    ink `size` px tall (edge included, at most the height) in the vertical middle (no font: an empty strip)."""
    text = text.strip()
    inner: int = max(1, min(size, height))
    outline: int = max(1, round(inner * outline_ratio))
    try:
        face = freetype.Face(font)
    except Exception:
        logger.exception('No font %s for the scrolling name', font)
        return np.zeros((height, 1, 4), np.uint8)
    probe: np.ndarray = render_text(face, text, probe_px)
    ink: int = probe.shape[0]
    if not text or ink == 0:
        return np.zeros((height, 1, 4), np.uint8)
    glyphs: np.ndarray = render_text(face, text, max(1, round(probe_px * (inner - 2 * outline) / ink)))
    # the black edge: the letters' alpha grown by `outline`, under the white letters
    kernel: np.ndarray = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * outline + 1, 2 * outline + 1))
    padded: np.ndarray = np.pad(glyphs, outline)
    edge: np.ndarray = cv2.dilate(padded, kernel)
    strip: np.ndarray = np.zeros((height, padded.shape[1], 4), np.uint8)
    top: int = max(0, (height - padded.shape[0]) // 2)
    rows: slice = slice(top, top + padded.shape[0])
    strip[rows, :, 3] = edge[:height - top]
    for channel in range(3):
        strip[rows, :, channel] = padded[:height - top]
    return strip


# wall (0 or 1), word, speed (px/s), gap (px), size (px), the ring's screens, when the loop began, the wall's cells with
# their places, and the screen it leaves at (None: it goes on)
Send = Callable[[int, str, float, float, int, int, float, list[tuple[int, int]], int | None], None]


class CreditScroller:
    """Hands the loops from the panel's Scroll settings (`RenderSettings`) to the walls (the clip player's
    `scroll_loop`). `update` runs in SMILE's render loop (with `time.time()`) and never raises there: while
    `credit_scroll` is on and nothing blocks it (a recent face, the laughter test), every `SEND_EVERY` second each
    row of `credit_scroll_rows` gets its word of `credit_name` (`plan_loops`) with the time it began and its own
    speed (`credit_scroll_speed_1` for the first word, `_2` for the others). A speed changed in the panel moves
    the start time so the word goes on from where it is. Off, or blocked, the loops get one last message with
    the screen they leave at and nothing more; a loop that was never told times out on the walls.
    `credit_scroll_last` shows what goes round."""

    def __init__(self, settings: 'RenderSettings', send: Send) -> None:
        self.settings: RenderSettings = settings
        self._send: Send = send
        self._words: dict[int, tuple[float, float]] = {}    # per word: when its loop began, at what speed (px/s)
        self._sent: float = 0.0
        self._text: str = ''
        self._warned: bool = False
        settings.credit_scroll_last = ''

    def update(self, now: float, blocked: bool) -> None:
        settings = self.settings
        if not settings.credit_scroll or blocked:
            if self._words:
                self.send_loops(now, leaving=True)
                self._words.clear()
            if self._text:
                self._text = ''
                settings.credit_scroll_last = ''
            return
        if not self._words:
            self._sent = now - SEND_EVERY
        if now - self._sent >= SEND_EVERY:
            self._sent = now
            self.send_loops(now)

    def send_loops(self, now: float, leaving: bool = False) -> list[tuple[str, RingPath]]:
        settings = self.settings
        rows: list[int] = parse_rows(str(settings.credit_scroll_rows))
        loops: list[tuple[str, RingPath]] = plan_loops(rows, str(settings.credit_name).split(), bool(settings.credit_scroll_reverse))
        if not loops:
            if not self._warned:
                self._warned = True
                logger.warning('No rows or no name for the scrolling name ("%s", "%s"; rows 0 to 4, like "%s")',
                               settings.credit_scroll_rows, settings.credit_name, DEFAULT_ROWS)
            return []
        self._warned = False
        gap: int = int(settings.credit_scroll_gap)
        speeds: tuple[float, float] = (float(settings.credit_scroll_speed_1), float(settings.credit_scroll_speed_2))
        for index, (word, path) in enumerate(loops):
            speed: float = max(1.0, speeds[min(index, 1)] * SCREEN_PX)
            start: float = self.word_start(index, speed, now)
            last: int | None = last_place((now - start) * speed, gap) if leaving else None
            for wall in range(WALLS):
                cells: list[tuple[int, int]] = path.cells(wall)
                if cells:
                    try:
                        self._send(wall, word, speed, float(gap), int(settings.credit_scroll_size), path.count, start, cells, last)
                    except Exception:
                        logger.exception('Scrolling name: wall %d failed', wall + 1)
        if leaving:
            logger.info('Scrolling name: leaving')
            return loops
        for index in [index for index in self._words if index >= len(loops)]:
            del self._words[index]
        text: str = ', '.join(f'{word} round row {path.row}' for word, path in loops)
        if text != self._text:
            self._text = text
            settings.credit_scroll_last = text
            logger.info('Scrolling name: %s', text)
        return loops

    def word_start(self, index: int, speed: float, now: float) -> float:
        """When word `index`'s loop began for its `speed` (px/s): now for a new word; a word whose speed changed
        gets a start that leaves it where it is, so it goes on from there at the new speed."""
        known: tuple[float, float] | None = self._words.get(index)
        if known is None:
            start: float = now
        else:
            start, old = known
            if old != speed:
                start = now - (now - start) * old / speed
        self._words[index] = (start, speed)
        return start
