"""The artist's credit: a QR code to the project page and the text around it, shown with the result on
the main screen (`CreditLayer`) and on one changing cell of each player wall (`PlayerRender`). No GL here."""

from typing import Callable

import cv2
import numpy as np

NAME: str = "Matthias Oostrik"
LINE: str = f"{NAME} likes your smile!"
URL: str = "https://oostrik.net/smile"
FONT: str = "data/Bangers-Regular.ttf"     # a comic book font: the credit is a joke (SIL Open Font License)

QUIET_MODULES: int = 4      # white modules around the code: scanners need them (the QR standard's quiet zone)


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
