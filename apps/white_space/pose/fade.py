"""Fade — a White Space-native pose feature: how present a pose is, in [0, 1].

``1.0`` = fully present, ``0.0`` = gone. Every live pose is ``1.0``; a ghost starts at ``1.0`` when it
is left behind and fades to ``0.0`` over ``Ghoster``'s ``fade_sweeps`` playhead sweeps, after which it
is removed. The sound and light layers scale a voice's volume / flash brightness by it.

The fade is a White Space concept (ghosts + the rotating playhead live with the app), so the feature
and its extractor live here rather than in ``modules/pose``. The open Frame ECS still lets the feature
ride on ``Frame`` (via ``replace``) without modules depending on app code.
"""

from __future__ import annotations

from modules.pose.features import NormalizedSingleValue
from modules.pose.frame import Frame, replace
from modules.pose.nodes import FilterNode


class Fade(NormalizedSingleValue):
    """Presence of a pose in [0, 1]: ``1.0`` = full (live pose / fresh ghost), ``0.0`` = faded out.

    Absent (NaN, score 0.0) only when never stamped; consumers treat absent as fully present.
    """


class FadeExtractor(FilterNode):
    """Stamps ``Fade = 1.0`` on every live pose. Ghosts are ``reidentify``-ed from these frames, so
    they inherit ``1.0`` at commit; ``Ghoster`` then fades them down over its ``fade_sweeps`` sweeps."""

    def process(self, pose: Frame) -> Frame:
        return replace(pose, {Fade: Fade.from_value(1.0)})
