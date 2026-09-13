import math

import numpy as np
from pythonosc.osc_bundle_builder import OscBundleBuilder, IMMEDIATELY
from pythonosc.osc_message_builder import OscMessageBuilder

from modules.inout import OscSound as BaseOscSound, OscSoundSettings
from modules.pose.frame import Frame as PoseFrame, FrameDict
from modules.pose.features import Azimuth
from modules.session import SequencerState
from ..light import Frame
from ..pose import GhostElement, GhostFeature, PlayheadOffset

import logging
logger = logging.getLogger(__name__)


class OscSoundSender(BaseOscSound):
    """The sound sender — modules' OscSound extended with the rotation playhead
    (/global/playhead), the motor mode (/global/motor), the two operator settings
    (/global/volume, /global/speaker/offset), and the panoramic-only per-pose azimuth and
    playhead-offset messages.

    ``/global/state`` carries the show state (``StateId``, 0–10) from the state machine, and
    −1 in the shutdown blackout. The motor mode goes out on ``/global/motor``.

    ``/global/speaker/offset`` is the one number Max needs of its own: where speaker 0 stands
    as an azimuth. With the speakers placed by the fixed layout (speaker 0 on the connection
    side, counter-clockwise from there) it is 0 and Max needs no constant at all — see
    ``docs/CALIBRATION.md``. Radians on the wire, like every other azimuth here.

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
        with self._input_lock:
            composition = self._composition

        super()._add_global_messages(bundle_builder, seq_state)

        # seq_state is None during idle/blackout → zero the playhead and motor.
        idle: bool = seq_state is None
        playhead: float = 0.0 if (idle or composition is None) else float(composition.playhead)
        playhead_msg = OscMessageBuilder(address="/global/playhead")
        playhead_msg.add_arg(playhead, OscMessageBuilder.ARG_TYPE_FLOAT)
        bundle_builder.add_content(playhead_msg.build())  # type: ignore

        motor_mode: int = 0 if (idle or composition is None) else int(composition.motor_command.mode)
        motor_msg = OscMessageBuilder(address="/global/motor")
        motor_msg.add_arg(motor_mode, OscMessageBuilder.ARG_TYPE_INT)
        bundle_builder.add_content(motor_msg.build())  # type: ignore

        # Operator settings, not show state, so they are never zeroed on idle: the fader and
        # the speaker placement must read true whenever they are turned, show or no show.
        volume_msg = OscMessageBuilder(address="/global/volume")
        volume_msg.add_arg(float(self._config.volume), OscMessageBuilder.ARG_TYPE_FLOAT)  # type: ignore[attr-defined]
        bundle_builder.add_content(volume_msg.build())  # type: ignore

        # Degrees in the panel, radians on the wire — the contract every azimuth here keeps.
        speaker_msg = OscMessageBuilder(address="/global/speaker/offset")
        speaker_msg.add_arg(math.radians(self._config.speaker_offset), OscMessageBuilder.ARG_TYPE_FLOAT)  # type: ignore[attr-defined]
        bundle_builder.add_content(speaker_msg.build())  # type: ignore

    def _add_active_frame_messages(self, bundle_builder: OscBundleBuilder, frame: PoseFrame, frames: FrameDict, num_players: int) -> None:
        super()._add_active_frame_messages(bundle_builder, frame, frames, num_players)
        id: int = frame.track_id

        azimuth: float = frame[Azimuth].value if Azimuth in frame else np.nan
        azimuth_msg = OscMessageBuilder(address=f"/pose/{id}/azimuth")
        azimuth_msg.add_arg(azimuth, OscMessageBuilder.ARG_TYPE_FLOAT)
        bundle_builder.add_content(azimuth_msg.build())  # type: ignore

        playhead_offset: float = frame[PlayheadOffset].value if PlayheadOffset in frame else np.nan
        offset_msg = OscMessageBuilder(address=f"/pose/{id}/playhead/offset")
        offset_msg.add_arg(playhead_offset, OscMessageBuilder.ARG_TYPE_FLOAT)
        bundle_builder.add_content(offset_msg.build())  # type: ignore

        # 1.0 for a live pose; a ghost's presence fades 1→0 over its lifetime (absent → fully present).
        fade: float = frame[GhostFeature].get(GhostElement.Fade) if GhostFeature in frame else 1.0
        fade_msg = OscMessageBuilder(address=f"/pose/{id}/playhead/fade")
        fade_msg.add_arg(fade, OscMessageBuilder.ARG_TYPE_FLOAT)
        bundle_builder.add_content(fade_msg.build())  # type: ignore

    def _add_inactive_frame_messages(self, bundle_builder: OscBundleBuilder, id: int, num_players: int) -> None:
        super()._add_inactive_frame_messages(bundle_builder, id, num_players)
        for address in (f"/pose/{id}/azimuth", f"/pose/{id}/playhead/offset", f"/pose/{id}/playhead/fade"):
            msg = OscMessageBuilder(address=address)
            msg.add_arg(0.0, OscMessageBuilder.ARG_TYPE_FLOAT)
            bundle_builder.add_content(msg.build())  # type: ignore
