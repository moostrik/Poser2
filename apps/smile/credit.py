"""The credit: a QR code to the project page and the artist's name on one changing cell of each wall
(`WallRender`), and the machine's joke, SMILE(ai) with a line from the lines file, shown with the result on the
main screen (`CreditLayer`) and on the walls' captions. No GL here."""

import logging
import os
import random
from typing import Any, Callable

import cv2
import freetype
import numpy as np

logger = logging.getLogger(__name__)

NAME: str = "Matthias Oostrik"
LINE: str = "loves your smile!"             # the line under SMILE(ai) when the lines file has none
LINES: str = "apps/smile/data/credit_lines.txt"        # one line each, blank lines and # comments skipped
URL: str = "https://oostrik.net/smile"
FONT: str = "apps/smile/data/Bangers-Regular.ttf"     # a comic book font: the credit is a joke (SIL Open Font License)

QUIET_MODULES: int = 4      # white modules around the code: scanners need them (the QR standard's quiet zone)

# the mark after SMILE, in fractions of the text's font size, as the page's title draws it (relay/web/style.css,
# `.ai` and `#title .ai`: the ring 1.5 em of letters at 0.5 em, its border 0.14 em of them, raised 0.9 em of them)
MARK_RING: float = 0.75         # the ring's outer diameter
MARK_STROKE: float = 0.07       # the ring's thickness
MARK_LETTERS: float = 0.5       # the font size of "ai"
MARK_LIFT: float = 0.635        # the ring's center above the baseline (the letters' baseline 0.45 up, their ink 0.37 tall)
MARK_GAP: float = 0.2           # from the E to the ring
TITLE_FONT: float = 0.9         # the text's font size, of `title_image`'s height_px; its capitals are 0.74 of that


def qr_image(url: str, size: int) -> np.ndarray:
    """The QR code of `url` with its quiet zone, black on white, as a BGR image of at most `size` pixels
    square: every module is a whole number of pixels, so the code stays sharp."""
    params = cv2.QRCodeEncoder.Params()
    params.correction_level = cv2.QRCodeEncoder_CORRECT_LEVEL_M
    modules: np.ndarray = cv2.QRCodeEncoder.create(params).encode(url)
    # OpenCV adds a narrower border of its own: crop to the code, then add the standard one
    dark = np.argwhere(modules < 128)
    (top, left), (bottom, right) = dark.min(axis=0), dark.max(axis=0) + 1
    modules = np.pad(modules[top:bottom, left:right], QUIET_MODULES, constant_values=255)
    scale: int = max(1, size // modules.shape[0])
    image: np.ndarray = np.kron(modules, np.ones((scale, scale), np.uint8))
    return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)


def quiet_px(image: np.ndarray) -> int:
    """The width of a QR image's white border (`qr_image`): the code itself starts this far in."""
    return int(np.argmax(image.min(axis=(1, 2)) < 255))


def render_text(face: Any, text: str, px: int) -> np.ndarray:
    """The text's ink at `px` pixels, as an alpha image cropped to the ink (0 x 0 without any)."""
    return render_text_baseline(face, text, px)[0]


def render_text_baseline(face: Any, text: str, px: int) -> tuple[np.ndarray, int]:
    """`render_text`'s image and the row of the baseline in it (the rows above it hold the ascent)."""
    face.set_pixel_sizes(0, px)
    pen: int = 0
    pieces: list[tuple[int, int, np.ndarray]] = []     # x, top (from the baseline up), bitmap
    previous: str | None = None
    for char in text:
        if previous is not None:
            pen += face.get_kerning(previous, char).x >> 6
        face.load_char(char, freetype.FT_LOAD_RENDER)  # pyright: ignore[reportAttributeAccessIssue]
        glyph = face.glyph
        bitmap = glyph.bitmap
        if bitmap.width and bitmap.rows:
            pieces.append((pen + glyph.bitmap_left, glyph.bitmap_top,
                           np.array(bitmap.buffer, np.uint8).reshape(bitmap.rows, bitmap.width)))
        pen += glyph.advance.x >> 6
        previous = char
    if not pieces:
        return np.zeros((0, 0), np.uint8), 0
    left: int = min(x for x, _, _ in pieces)
    right: int = max(x + b.shape[1] for x, _, b in pieces)
    top: int = max(t for _, t, _ in pieces)
    bottom: int = min(t - b.shape[0] for _, t, b in pieces)
    image: np.ndarray = np.zeros((top - bottom, right - left), np.uint8)
    for x, t, b in pieces:
        y: int = top - t
        region = image[y:y + b.shape[0], x - left:x - left + b.shape[1]]
        np.maximum(region, b, out=region)
    return image, top


def title_image(name: str, height_px: int, alpha: float = 1.0, outline_ratio: float = 0.025, scale: int = 4) -> np.ndarray:
    """The machine's title as a BGRA image: "SMILE" with the machine's mark after it, the ring with "ai" in it as
    the page's title has it (smaller than the letters and raised above them like a (R), MARK_*), and "by" `name`
    after that ("SMILE" and the mark alone without one), all on one baseline in the credit's font, white with
    one black edge round all of it, `outline_ratio` of `height_px` wide (drawn at `scale` times the size, the
    edge grown like the screen's text, then scaled down)."""
    face = freetype.Face(FONT)
    size: int = height_px * scale
    font: int = round(size * TITLE_FONT)
    title, title_base = render_text_baseline(face, 'SMILE', font)
    credit, credit_base = render_text_baseline(face, f'by {name.strip()}', font) if name.strip() else (np.zeros((0, 0), np.uint8), 0)
    letters: np.ndarray = render_text(face, 'ai', round(MARK_LETTERS * font))
    ring: int = round(MARK_RING * font)
    stroke: int = max(2, round(MARK_STROKE * font))
    lift: int = round(MARK_LIFT * font)
    gap: int = round(MARK_GAP * font)
    after: int = round(0.3 * font)
    # the rows: the baseline low enough for the raised ring, and under it the name's descenders
    baseline: int = max(title_base, lift + ring // 2 + 1)
    descent: int = max(1, title.shape[0] - title_base, credit.shape[0] - credit_base if credit.size else 0)
    width: int = title.shape[1] + gap + ring + (after + credit.shape[1] if credit.size else 0)
    ink: np.ndarray = np.zeros((baseline + descent, width), np.uint8)

    def put(piece: np.ndarray, base: int, x: int) -> None:
        h, w = piece.shape[:2]
        y: int = baseline - base
        np.maximum(ink[y:y + h, x:x + w], piece, out=ink[y:y + h, x:x + w])

    put(title, title_base, 0)
    center: tuple[int, int] = (title.shape[1] + gap + ring // 2, baseline - lift)
    cv2.circle(ink, center, (ring - stroke) // 2, 255, stroke, cv2.LINE_AA)
    h, w = letters.shape[:2]
    y0: int = center[1] - h // 2       # the letters' ink centered in the ring
    x0: int = center[0] - w // 2
    np.maximum(ink[y0:y0 + h, x0:x0 + w], letters, out=ink[y0:y0 + h, x0:x0 + w])
    if credit.size:
        put(credit, credit_base, title.shape[1] + gap + ring + after)
    outline: int = max(1, round(height_px * outline_ratio * scale))
    kernel: np.ndarray = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * outline + 1, 2 * outline + 1))
    padded: np.ndarray = np.pad(ink, outline)
    edge: np.ndarray = cv2.dilate(padded, kernel)
    mark: np.ndarray = np.zeros((padded.shape[0], padded.shape[1], 4), np.uint8)
    mark[:, :, 3] = edge
    for channel in range(3):
        mark[:, :, channel] = padded
    small: tuple[int, int] = (round(mark.shape[1] / scale), round(mark.shape[0] / scale))
    image: np.ndarray = cv2.resize(mark, small, interpolation=cv2.INTER_AREA)
    image[:, :, 3] = (image[:, :, 3].astype(np.float32) * alpha).astype(np.uint8)
    return image


class CreditLines:
    """The lines under SMILE(ai), from a text file that is read again whenever it changes, so it
    can be edited while the installation runs. `pick` draws one at random, never the last one again."""

    def __init__(self, path: str = LINES) -> None:
        self.path: str = path
        self._lines: list[str] = []
        self._mtime: float | None = None
        self._last: str | None = None

    @property
    def lines(self) -> list[str]:
        try:
            mtime: float = os.path.getmtime(self.path)
        except OSError:
            mtime = -1.0
        if mtime != self._mtime:
            self._mtime = mtime
            self._lines = self._read() if mtime >= 0 else []
            if not self._lines:
                logger.warning('No credit lines in %s: the caption says "%s"', self.path, LINE)
            else:
                logger.info('%d credit lines from %s', len(self._lines), self.path)
        return self._lines

    def _read(self) -> list[str]:
        try:
            with open(self.path, encoding='utf-8') as file:
                return [line.strip() for line in file if line.strip() and not line.lstrip().startswith('#')]
        except OSError:
            logger.exception('Could not read %s', self.path)
            return []

    def pick(self) -> str:
        """A line at random, a different one from the last pick when there is more than one."""
        lines: list[str] = self.lines or [LINE]
        candidates: list[str] = [line for line in lines if line != self._last] or lines
        self._last = random.choice(candidates)
        return self._last


def wrap_balanced(text: str, measure: Callable[[str], int], max_width: int) -> list[str]:
    """`text` in the fewest lines no wider than `max_width` (a word that is wider keeps a line of its own),
    with the longest of them as short as it can be: "Matthias Oostrik / likes your smile!", not
    "Matthias Oostrik likes / your smile!"."""
    words: list[str] = text.split()
    if not words:
        return []

    def greedy(limit: int) -> list[str]:
        lines: list[str] = [words[0]]
        for word in words[1:]:
            joined: str = f"{lines[-1]} {word}"
            if measure(joined) <= limit:
                lines[-1] = joined
            else:
                lines.append(word)
        return lines

    fewest: list[str] = greedy(max_width)
    # the narrowest limit that needs no more lines: every line width that can occur is a candidate
    candidates: set[int] = {measure(" ".join(words[i:j])) for i in range(len(words)) for j in range(i + 1, len(words) + 1)}
    for limit in sorted(width for width in candidates if width <= max_width):
        lines: list[str] = greedy(limit)
        if len(lines) <= len(fewest):
            return lines
    return fewest
