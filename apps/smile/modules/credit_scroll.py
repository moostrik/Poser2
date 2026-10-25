"""The artist's name scrolling by on the player walls: from time to time one word of it (the first name,
the next time the surname) runs once along one ring of the sphere, over the playing clips, like a ticker,
letters nearly as tall as a screen. SMILE plans every
pass (`CreditScroller`, in the render loop) and sends each player the cells of its own that are on the
path with their place along it; the players know nothing of the sphere and draw the name from one long
texture (`name_strip`). No GL here: the geometry is shared by both apps and tested without one."""

import logging
import random
import time
from dataclasses import dataclass
from typing import Any, Callable

import cv2
import freetype
import numpy as np

from modules import sphere
from modules.credit import FONT

logger = logging.getLogger(__name__)

# the sphere and the wiring of its screens to the players' cells are in `modules.sphere`; here a screen
# is counted from 0 (the sphere's number minus 1)
ROWS: int = sphere.HEIGHTS
SCREENS: int = sphere.SCREENS
CELLS: int = sphere.CELLS
PLAYERS: int = sphere.PLAYERS
SCREEN_PX: int = 512            # a screen's width, in the player's pixels (the clip's upright 512 x 640)
SCREEN_HEIGHT_PX: int = 640

DEFAULT_ROWS: str = "1,3"       # the rings with 12 screens, leaving the middle one to the clips
ADDRESS: str = '/scroll'        # to the players: name, speed (px/s), gap (px), size (px), then cell, place, cell, place ...
LOOP_ADDRESS: str = '/scroll_loop'  # a word going round a ring without end: word, speed, gap, size, the ring's screens, the
                                    # time it began (time.time(): SMILE and the players share the PC's clock, so the
                                    # resends move nothing), then cell, place ...; sent every second, a loop no message renews ends
LOOP_SEND_EVERY: float = 1.0
LOOP_TIMEOUT: float = 3.0       # the players drop a loop that got no message for this long (SMILE gone, the switch off)


def row_screens(row: int) -> list[int]:
    """The screens (from 0) of ring `row` in order round the sphere: 12 for rows 1 to 3, 6 for 0 and 4."""
    return [screen - 1 for screen in sphere.row_screens(row)]


def screen_cell(screen: int) -> tuple[int, int]:
    """The player (0 or 1) and its cell of a screen (from 0), by the wiring."""
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
class ScrollPass:
    """One pass of the name: along `row`, entering at the `start`th screen of the ring and covering `count`
    screens from there, round the ring the other way with `reverse`. The name moves to the left of whoever
    stands in front of it, and the ribs count to the camera's left, which is that visitor's right: so the
    next screen is the rib before (checked on site 2026-10-25)."""
    row: int
    start: int
    count: int
    reverse: bool = False

    @property
    def screens(self) -> list[int]:
        """The screens in the order the name reaches them (a ring is closed: the path wraps)."""
        ring: list[int] = row_screens(self.row)
        step: int = 1 if self.reverse else -1
        return [ring[(self.start + step * i) % len(ring)] for i in range(self.count)]

    def cells(self, player: int) -> list[tuple[int, int]]:
        """The cells of `player` on the path, each with its place along it (0: where the name enters)."""
        return [(cell, place) for place, screen in enumerate(self.screens)
                for p, cell in [screen_cell(screen)] if p == player]

    def describe(self) -> str:
        """For the readout: the row and the screens as numbered on site (from 1)."""
        return f"row {self.row}: screens {', '.join(str(screen + 1) for screen in self.screens)}"


def plan_pass(rows: list[int], count_min: int, count_max: int, reverse: bool, rng: random.Random | None = None) -> ScrollPass | None:
    """A pass on one of `rows` at random, from a random screen, over `count_min` to `count_max` screens (at
    most the ring's, at least 1); None without a row."""
    if not rows:
        return None
    rng = rng or random.Random()
    row: int = rng.choice(rows)
    ring: int = len(row_screens(row))
    low: int = max(1, min(count_min, count_max, ring))
    high: int = max(low, min(max(count_min, count_max), ring))
    return ScrollPass(row, rng.randrange(ring), rng.randint(low, high), reverse)


def plan_loops(rows: list[int], words: list[str], reverse: bool) -> list[tuple[str, ScrollPass]]:
    """The continuous mode: each of `rows` gets a word of the name going round it, the first word on the
    first row and so on (the words again when there are more rows); the path is the whole ring."""
    if not words:
        return []
    return [(words[index % len(words)], ScrollPass(row, 0, len(row_screens(row)), reverse)) for index, row in enumerate(rows)]


def scroll_offset(place: int, travelled: float, gap_px: float, screen_px: int = SCREEN_PX) -> float:
    """Where the screen at `place` is in the name's texture after the name moved `travelled` px: the texture
    x at the screen's left edge. The name reads left to right and moves to the left, as a ticker, so it
    enters the first screen at its right edge and the next screen lies beyond the first one's left edge, a
    `gap_px` wide wall between them."""
    return travelled - screen_px - place * (screen_px + gap_px)


def scroll_done(travelled: float, count: int, strip_px: int, gap_px: float, screen_px: int = SCREEN_PX) -> bool:
    """Whether the end of a `strip_px` wide name has left the last of `count` screens."""
    return scroll_offset(count - 1, travelled, gap_px, screen_px) >= strip_px


def name_strip(text: str, height: int, size: int, font: str = FONT, outline_ratio: float = 0.025,
               probe_px: int = 200) -> np.ndarray:
    """The name as one long transparent BGRA image `height` px tall, white letters with a black edge, their
    ink `size` px tall (edge included, at most the height) in the vertical middle (no font: an empty strip)."""
    text = text.strip()
    inner: int = max(1, min(size, height))
    outline: int = max(1, round(inner * outline_ratio))
    try:
        face = freetype.Face(font)
    except Exception:
        logger.exception('No font %s for the scrolling name', font)
        return np.zeros((height, 1, 4), np.uint8)
    probe: np.ndarray = _render(face, text, probe_px)
    ink: int = probe.shape[0]
    if not text or ink == 0:
        return np.zeros((height, 1, 4), np.uint8)
    glyphs: np.ndarray = _render(face, text, max(1, round(probe_px * (inner - 2 * outline) / ink)))
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


def _render(face: Any, text: str, px: int) -> np.ndarray:
    """The text's ink at `px` pixels, as an alpha image cropped to the ink (0 x 0 without any)."""
    face.set_pixel_sizes(0, px)
    pen: int = 0
    pieces: list[tuple[int, int, np.ndarray]] = []     # x, top (from the baseline up), bitmap
    previous: str | None = None
    for char in text:
        if previous is not None:
            pen += face.get_kerning(previous, char).x >> 6
        face.load_char(char, freetype.FT_LOAD_RENDER)      # type: ignore[attr-defined]
        glyph = face.glyph
        bitmap = glyph.bitmap
        if bitmap.width and bitmap.rows:
            pieces.append((pen + glyph.bitmap_left, glyph.bitmap_top,
                           np.array(bitmap.buffer, np.uint8).reshape(bitmap.rows, bitmap.width)))
        pen += glyph.advance.x >> 6
        previous = char
    if not pieces:
        return np.zeros((0, 0), np.uint8)
    left: int = min(x for x, _, _ in pieces)
    right: int = max(x + b.shape[1] for x, _, b in pieces)
    top: int = max(t for _, t, _ in pieces)
    bottom: int = min(t - b.shape[0] for _, t, b in pieces)
    image: np.ndarray = np.zeros((top - bottom, right - left), np.uint8)
    for x, t, b in pieces:
        y: int = top - t
        region = image[y:y + b.shape[0], x - left:x - left + b.shape[1]]
        np.maximum(region, b, out=region)
    return image


Send = Callable[[int, str, list], None]     # player (0 or 1), address, arguments


class CreditScroller:
    """Plans the passes from the panel's Scroll settings (`RenderSettings`) and sends them to the players.
    `update` runs in SMILE's render loop and never raises there: while `credit_scroll` is on and nothing
    blocks it (the laughter test), a pass goes out every `credit_scroll_every` seconds; the button
    (`request`) sends one at once. Each pass carries one word of `credit_name`, the words by turns (a whole
    name is too long for a pass of a few screens). With `credit_scroll_loop` the words go round the rows
    without end instead (`plan_loops`), sent again every second with the time they began so a player that
    starts late joins in step, the second word `credit_scroll_offset` degrees ahead of the first; the button
    still sends a pass. `credit_scroll_last` shows the last pass."""

    def __init__(self, settings: Any, send: Send) -> None:
        self.settings: Any = settings
        self._send: Send = send
        self._next_time: float | None = None    # when the next timed pass is due, None while off
        self._requested: bool = False
        self._warned: bool = False
        self._word: int = random.randrange(2)   # which word of the name the next pass takes
        self._loop_start: float | None = None   # when the continuous mode began, None while off
        self._loop_sent: float = 0.0
        self._loop_text: str = ''
        settings.credit_scroll_last = ''

    def request(self, _: Any = None) -> None:
        """One pass now (the panel's button, any thread)."""
        self._requested = True

    def update(self, now: float, blocked: bool) -> None:
        settings = self.settings
        looping: bool = bool(settings.credit_scroll) and bool(settings.credit_scroll_loop)
        if looping and not blocked:
            if self._loop_start is None:
                self._loop_start = now
                self._loop_sent = now - LOOP_SEND_EVERY
            if now - self._loop_sent >= LOOP_SEND_EVERY:
                self._loop_sent = now
                self.send_loops(now)
        else:
            self._loop_start = None
            self._loop_text = ''
        if not settings.credit_scroll or looping:
            self._next_time = None
        elif self._next_time is None:
            self._next_time = now + float(settings.credit_scroll_every)
        due: bool = self._next_time is not None and now >= self._next_time
        if blocked:
            self._requested = False
            if due:
                self._next_time = now + float(settings.credit_scroll_every)
            return
        if not due and not self._requested:
            return
        self._requested = False
        if self._next_time is not None:
            self._next_time = now + float(settings.credit_scroll_every)
        self.send_pass(now)

    def send_pass(self, now: float) -> ScrollPass | None:
        settings = self.settings
        rows: list[int] = parse_rows(str(settings.credit_scroll_rows))
        scroll: ScrollPass | None = plan_pass(rows, int(settings.credit_scroll_screens_min), int(settings.credit_scroll_screens_max),
                                              bool(settings.credit_scroll_reverse))
        if scroll is None:
            if not self._warned:
                self._warned = True
                logger.warning('No rows for the scrolling name in "%s" (0 to 4, like "%s")', settings.credit_scroll_rows, DEFAULT_ROWS)
            return None
        self._warned = False
        word: str = self.next_word(str(settings.credit_name))
        head: list = [word, float(settings.credit_scroll_speed) * SCREEN_PX,
                      int(settings.credit_scroll_gap), int(settings.credit_scroll_size)]
        for player in range(PLAYERS):
            cells: list[tuple[int, int]] = scroll.cells(player)
            if cells:
                try:
                    self._send(player, ADDRESS, head + [value for pair in cells for value in pair])
                except Exception:
                    logger.exception('Scrolling name: player %d could not be reached', player + 1)
        settings.credit_scroll_last = f"{time.strftime('%H:%M:%S', time.localtime(now))} {word}, {scroll.describe()}"
        logger.info('Scrolling name: %s, %s', word, scroll.describe())
        return scroll

    def send_loops(self, now: float) -> list[tuple[str, ScrollPass]]:
        settings = self.settings
        rows: list[int] = parse_rows(str(settings.credit_scroll_rows))
        loops: list[tuple[str, ScrollPass]] = plan_loops(rows, str(settings.credit_name).split(), bool(settings.credit_scroll_reverse))
        if not loops:
            if not self._warned:
                self._warned = True
                logger.warning('No rows or no name for the continuous scrolling name ("%s", "%s")', settings.credit_scroll_rows, settings.credit_name)
            return []
        self._warned = False
        speed: float = float(settings.credit_scroll_speed) * SCREEN_PX
        gap: int = int(settings.credit_scroll_gap)
        start: float = self._loop_start if self._loop_start is not None else now
        for index, (word, scroll) in enumerate(loops):
            # the word of row i starts i * offset degrees further round its ring: that far ahead is that much earlier
            ahead: float = index * float(settings.credit_scroll_offset) / 360.0 * scroll.count * (SCREEN_PX + gap)
            head: list = [word, speed, gap, int(settings.credit_scroll_size), scroll.count, start - ahead / speed]
            for player in range(PLAYERS):
                cells: list[tuple[int, int]] = scroll.cells(player)
                if cells:
                    try:
                        self._send(player, LOOP_ADDRESS, head + [value for pair in cells for value in pair])
                    except Exception:
                        logger.exception('Scrolling name: player %d could not be reached', player + 1)
        text: str = 'continuous: ' + ', '.join(f'{word} row {scroll.row}' for word, scroll in loops)
        if text != self._loop_text:
            self._loop_text = text
            settings.credit_scroll_last = text
            logger.info('Scrolling name %s', text)
        return loops

    def next_word(self, name: str) -> str:
        """The word of `name` this pass takes: the words by turns, the whole name when it is one word."""
        words: list[str] = name.split()
        if not words:
            return ''
        word: str = words[self._word % len(words)]
        self._word = (self._word + 1) % len(words)
        return word
