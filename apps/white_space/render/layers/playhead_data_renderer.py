"""PlayheadDataRenderer — per-player text overlay for the playhead↔pose relationship.

Draws two numeric readouts in a pose data cell, read straight off the board's LERP frame:
  PH:<deg>            signed PlayheadPhase (pose azimuth relative to the playhead)
  ST:<0-1> [####----] PlayheadStability with an ASCII fill bar

App-local (playhead is a White Space concept); modelled on modules' ``MTimeRenderer`` —
a pure-text ``LayerBase`` with no shader/FBO of its own.
"""

import math

from OpenGL.GL import *  # type: ignore  # noqa: F401,F403

from modules.board import HasFrames
from modules.gl import Text
from modules.pose.frame import Frame
from modules.render.layers.LayerBase import LayerBase

from apps.white_space.pose import PlayheadPhase, PlayheadStability
from apps.white_space.settings import Stage

_BAR_CELLS: int = 10


class PlayheadDataRenderer(LayerBase):
    def __init__(self, track_id: int, board: HasFrames, stage: int = int(Stage.LERP)) -> None:
        self._board: HasFrames = board
        self._track_id: int = track_id
        self._stage: int = stage
        self._lines: list[str] | None = None

        self._text_renderer: Text = Text()
        self._width: int = 0
        self._height: int = 0

    def allocate(self, width: int | None = None, height: int | None = None, internal_format: int | None = None) -> None:
        if width and height:
            self._width = width
            self._height = height
        self._text_renderer.allocate("data/RobotoMono-Regular.ttf", font_size=34)

    def deallocate(self) -> None:
        self._text_renderer.deallocate()

    def draw(self) -> None:
        if not self._lines or self._width == 0:
            return
        # Stacked below the MT: readout drawn by MTimeRenderer (top-left, y≈20).
        y = 70
        for line in self._lines:
            self._text_renderer.draw_box_text(
                20, y, line,
                color=(1.0, 1.0, 1.0, 1.0),
                bg_color=(0.0, 0.0, 0.0, 0.8),
                screen_width=self._width,
                screen_height=self._height,
            )
            y += 45

    def update(self) -> None:
        pose: Frame | None = self._board.get_frame(self._stage, self._track_id)
        if pose is None:
            self._lines = None
            return

        phase: float = pose[PlayheadPhase].value
        phase_str: str = "PH:--" if math.isnan(phase) else f"PH:{math.degrees(phase):+.0f}"

        stability = pose[PlayheadStability]
        if stability.score <= 0.0 or math.isnan(stability.value):
            stab_str: str = "ST:--"
        else:
            filled = max(0, min(_BAR_CELLS, round(stability.value * _BAR_CELLS)))
            bar = "[" + "#" * filled + "-" * (_BAR_CELLS - filled) + "]"
            stab_str = f"ST:{stability.value:.2f} {bar}"

        self._lines = [phase_str, stab_str]
