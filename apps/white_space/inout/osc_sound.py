import numpy as np
from pythonosc.osc_bundle_builder import OscBundleBuilder, IMMEDIATELY
from pythonosc.osc_message_builder import OscMessageBuilder

from modules.inout import OscSound as BaseOscSound, OscSoundSettings
from modules.pose.frame import Frame as PoseFrame, FrameDict
from modules.pose.features import Azimuth, Distance
from modules.session import SequencerState
from ..light import Frame
from ..pose import PlayheadOffset, PlayheadStability

import logging
logger = logging.getLogger(__name__)


class OscSound(BaseOscSound):
    """OscSound extended with the rotation playhead (/global/playhead) and the
    panoramic-only per-pose azimuth, distance, and playhead-offset messages.

    Also owns the id-slot count: it sends ``max_players`` live slots plus ``virtual_players``
    ghost slots (ids Ghoster injects beyond the tracked players). It overrides the base's
    ``_build_bundle`` / ``_send_blackout`` to iterate that extended range while keeping the
    per-track (similarity/gate/leader) array width at ``max_players``."""

    def __init__(self, settings: OscSoundSettings) -> None:
        super().__init__(settings)
        self._composition: Frame | None = None
        # Extend the base's inactive-reset throttle to cover the virtual (ghost) slots.
        self._inactive_counts = {id: 0 for id in range(self._slot_count)}

    @property
    def _slot_count(self) -> int:
        """Live players (``max_players``) + the virtual (ghost) id slots Ghoster injects.
        The per-track array width stays ``max_players``; only the slot count grows."""
        return self._config.max_players + self._config.virtual_players  # type: ignore[attr-defined]

    def set_composition(self, output: Frame) -> None:
        """Store the latest Frame; thread-safe."""
        with self._input_lock:
            self._composition = output

    def _build_bundle(self, frames: FrameDict, seq_state: SequencerState | None) -> OscBundleBuilder:
        """Live + ghost slots; per-track arrays stay ``max_players`` wide (see class doc)."""
        bundle = OscBundleBuilder(IMMEDIATELY)  # type: ignore
        self._add_global_messages(bundle, seq_state)

        track = self._config.max_players   # per-track (similarity/gate/leader) array width
        self._config.active_players = sum(1 for id in range(self._slot_count) if id in frames)

        for id in range(self._slot_count):
            if id in frames:
                self._inactive_counts[id] = 0
                self._add_active_frame_messages(bundle, frames[id], frames, track)
            elif self._inactive_counts[id] < 2:
                # Only send a slot's inactive reset for the first 2 ticks after it empties.
                self._inactive_counts[id] += 1
                self._add_inactive_frame_messages(bundle, id, track)
        return bundle

    def _send_blackout(self) -> None:
        """All-zero bundle across every live + ghost slot."""
        bundle = OscBundleBuilder(IMMEDIATELY)  # type: ignore
        self._add_global_messages(bundle, None)  # None → zeroed globals
        for id in range(self._slot_count):
            self._add_inactive_frame_messages(bundle, id, self._config.max_players)
        self._config.active_players = 0
        self._send_bundle(bundle)

    def _add_global_messages(self, bundle_builder: OscBundleBuilder, seq_state: SequencerState | None) -> None:
        super()._add_global_messages(bundle_builder, seq_state)

        with self._input_lock:
            composition = self._composition

        # seq_state is None during idle/blackout → zero the playhead.
        idle: bool = seq_state is None
        playhead: float = 0.0 if (idle or composition is None) else float(composition.playhead)
        playhead_msg = OscMessageBuilder(address="/global/playhead")
        playhead_msg.add_arg(playhead, OscMessageBuilder.ARG_TYPE_FLOAT)
        bundle_builder.add_content(playhead_msg.build())  # type: ignore

    def _add_active_frame_messages(self, bundle_builder: OscBundleBuilder, frame: PoseFrame, frames: FrameDict, num_players: int) -> None:
        super()._add_active_frame_messages(bundle_builder, frame, frames, num_players)
        id: int = frame.track_id

        azimuth: float = frame[Azimuth].value if Azimuth in frame else np.nan
        azimuth_msg = OscMessageBuilder(address=f"/pose/{id}/azimuth")
        azimuth_msg.add_arg(azimuth, OscMessageBuilder.ARG_TYPE_FLOAT)
        bundle_builder.add_content(azimuth_msg.build())  # type: ignore

        distance: float = frame[Distance].value if Distance in frame else np.nan
        distance_msg = OscMessageBuilder(address=f"/pose/{id}/distance")
        distance_msg.add_arg(distance, OscMessageBuilder.ARG_TYPE_FLOAT)
        bundle_builder.add_content(distance_msg.build())  # type: ignore

        playhead_offset: float = frame[PlayheadOffset].value if PlayheadOffset in frame else np.nan
        offset_msg = OscMessageBuilder(address=f"/pose/{id}/playhead_offset")
        offset_msg.add_arg(playhead_offset, OscMessageBuilder.ARG_TYPE_FLOAT)
        bundle_builder.add_content(offset_msg.build())  # type: ignore

        stability: float = frame[PlayheadStability].value if PlayheadStability in frame else np.nan
        stability_msg = OscMessageBuilder(address=f"/pose/{id}/playhead_stability")
        stability_msg.add_arg(stability, OscMessageBuilder.ARG_TYPE_FLOAT)
        bundle_builder.add_content(stability_msg.build())  # type: ignore

    def _add_inactive_frame_messages(self, bundle_builder: OscBundleBuilder, id: int, num_players: int) -> None:
        super()._add_inactive_frame_messages(bundle_builder, id, num_players)
        for address in (f"/pose/{id}/azimuth", f"/pose/{id}/distance", f"/pose/{id}/playhead_offset", f"/pose/{id}/playhead_stability"):
            msg = OscMessageBuilder(address=address)
            msg.add_arg(0.0, OscMessageBuilder.ARG_TYPE_FLOAT)
            bundle_builder.add_content(msg.build())  # type: ignore
