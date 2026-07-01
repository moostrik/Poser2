"""GhostedFeature — a White Space-native pose feature: whether a live person currently stands
on a spot that a ghost claims (i.e. inside a ghost's claimed azimuth band).

Stamped per live frame by ``Ghoster`` (see ``ghoster.py``): ``1.0`` when the person's azimuth
falls within an existing ghost's band, ``0.0`` otherwise. It flows to consumers via the normal
Frame path — the light's ``HauntedFlash`` layer reads it off the board frames to draw the
quarter-turn-early blue flash, and ``Ghoster`` uses the same per-tick computation to mute the
person's live voice over OSC (the ghost is the only voice on a claimed spot).

Like ``PlayheadOffset`` / ``PlayheadStability`` this is a White Space concept, so the feature
lives with the app rather than in ``modules/pose``; the open Frame ECS still lets it ride on
``Frame`` (via ``replace``) without modules depending on app code.
"""

from __future__ import annotations

from modules.pose.features import NormalizedSingleValue


class GhostedFeature(NormalizedSingleValue):
    """Per-person flag in ``[0, 1]``: ``1.0`` = standing on a spot a ghost claims.

    Always stamped on live frames by ``Ghoster`` (``0.0`` when not ghosted, score ``1.0``),
    so consumers can read it directly without a presence check.
    """
