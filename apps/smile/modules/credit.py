"""The artist's credit: a QR code to the project page and the text around it, shown with the result on
the main screen (`CreditLayer`) and on one changing cell of each player wall (`PlayerRender`). No GL here."""

from typing import Callable

import cv2
import numpy as np

NAME: str = "Matthias Oostrik"
LINE: str = f"{NAME} likes your smile!"
URL: str = "https://oostrik.net/smile"

QUIET_MODULES: int = 4      # white modules around the code: scanners need them (the QR standard's quiet zone)


def qr_image(url: str, size: int) -> np.ndarray:
    """The QR code of `url` with its quiet zone, black on white, as a BGR image of at most `size` pixels
    square: every module is a whole number of pixels, so the code stays sharp."""
    params = cv2.QRCodeEncoder.Params()
    params.correction_level = cv2.QRCodeEncoder_CORRECT_LEVEL_M
    modules: np.ndarray = cv2.QRCodeEncoder.create(params).encode(url)
    modules = np.pad(modules, QUIET_MODULES, constant_values=255)
    scale: int = max(1, size // modules.shape[0])
    image: np.ndarray = np.kron(modules, np.ones((scale, scale), np.uint8))
    return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)


def shown_url(url: str) -> str:
    """The url as people read it: without the scheme."""
    return url.split("://", 1)[-1].rstrip("/")


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
