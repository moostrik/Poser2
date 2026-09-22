"""PlayheadMarker — the content playhead drawn as a marker in the projection.

A band of ``width`` degrees at the normalized azimuth of ``frame.playhead`` (the continuous
content playhead, radians [-π, π); NaN → nothing drawn), added to each channel at its level and
dimmed to ``at_mask`` of it inside a person's mask, so the marker never blinds at a person. The
pose instrument draws it over its masks (``pose_instrument.py``); the ``projection_playhead``
debug layer draws it alone. Distinct from the beam-mode ``BeamPlayhead`` and the motor/content
``Playhead`` (the NCO in ``light/playhead.py``).

``draw`` is a static method so ``HotReloadMethods`` can patch it while the app runs.
"""

import math

import numpy as np

from modules.settings import BaseSettings, Field, Widget

from .._utilities import normalize_azimuth

KNOB = Widget.knob


class PlayheadMarkerSettings(BaseSettings):
    """The marker: its width, its level per channel (0 is off), and its level inside a mask."""
    width:   Field[float] = Field(3.6, min=0.1, max=36.0, step=0.1,  widget=KNOB, label="Width",   description="Marker width (deg)", row_label="Playhead", newline=True)
    white:   Field[float] = Field(1.0, min=0.0, max=1.0,  step=0.01, widget=KNOB, label="White",   description="Marker white level")
    blue:    Field[float] = Field(0.0, min=0.0, max=1.0,  step=0.01, widget=KNOB, label="Blue",    description="Marker blue level")
    at_mask: Field[float] = Field(0.3, min=0.0, max=1.0,  step=0.01, widget=KNOB, label="At Mask", description="Marker levels inside a mask (fraction)")


class PlayheadMarker:
    """See the module docstring."""

    @staticmethod
    def draw(white: np.ndarray, blue: np.ndarray, resolution: int, playhead: float,
             settings: PlayheadMarkerSettings, masked: np.ndarray) -> None:
        """Add the marker to both channels; ``masked`` is the pixels under a mask this tick."""
        if math.isnan(playhead):
            return
        centre = int(normalize_azimuth(playhead) * resolution)          # [0, R)
        w = max(1, round(settings.width / 360.0 * resolution))           # deg → pixel count
        start = centre - w // 2
        idx = np.arange(start, start + w) % resolution
        dim = np.where(masked[idx], settings.at_mask, 1.0)
        for channel, level in ((white, settings.white), (blue, settings.blue)):
            if level > 0.0:
                channel[idx] += (level * dim).astype(channel.dtype)
