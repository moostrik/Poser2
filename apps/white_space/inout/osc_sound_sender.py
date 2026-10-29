import math

import numpy as np
from pythonosc.osc_bundle_builder import OscBundleBuilder, IMMEDIATELY
from pythonosc.osc_message_builder import OscMessageBuilder

from modules.inout import OscSound as BaseOscSound, OscSoundSettings
from modules.pose import Frame, FrameDict, Azimuth, Distance, Angles, ArmTravel, TravelElement, ShoulderDifference
from modules.session import SequencerState
from ..light import LightFrame
from ..pose import PlayheadOffset

import logging
logger = logging.getLogger(__name__)


class OscSoundSender(BaseOscSound):
    """The sound sender — modules' OscSound extended with the rotation playhead
    (/global/playhead), the motor mode (/global/motor), group sync's envelope (/global/sync,
    the pipeline's gate through the instrument's strobe envelope: 0..1, 0 while the instrument is not drawn), the live player
    count (/global/players: occupied live slots, the dummy excluded; presence
    data, never zeroed on idle), the two operator settings (/global/volume,
    /global/speaker/offset), and the panoramic-only per-pose azimuth, distance and
    playhead-offset messages.

    ``/global/state`` carries the show state (``StateId``, 0–11) from the state machine, and
    −1 in the shutdown blackout. The motor mode goes out on ``/global/motor``.

    ``/global/speaker/offset`` is the one number Max needs of its own: where speaker 0 stands
    as an azimuth. With the speakers placed by the fixed layout (speaker 0 on the connection
    side, counter-clockwise from there) it is 0 and Max needs no constant at all — see
    ``docs/CALIBRATION.md``. Radians on the wire, like every other azimuth here.

    The four arm entries of ``/pose/{id}/angle/rad`` carry the joints' travel times π
    (``ArmTravel``: the calibrated angle through the pipeline's dead zones), so Max's absolute
    over π reads the value the light reads; the other entries are the angles (``docs/SOUND.md``).
    ``/pose/{id}/arm`` carries the whole arms — left and right, each the union of its shoulder's
    and its elbow's travel, and their mean — and ``/pose/{id}/shoulder/difference`` the
    shoulders' signed travel difference: the pipeline's values, the same ones the light plays.

    Also owns the id-slot count: it sends ``max_players`` live slots plus the dummy's. It
    overrides the base's ``_build_bundle`` / ``_send_blackout`` to iterate that extended range
    while keeping the per-track (similarity/gate/leader) array width at ``max_players``."""

    def __init__(self, settings: OscSoundSettings) -> None:
        super().__init__(settings)
        self._light_frame: LightFrame | None = None
        # Extend the base's inactive-reset throttle to cover the dummy's slot.
        self._inactive_counts = {id: 0 for id in range(self._slot_count)}

    @property
    def _slot_count(self) -> int:
        """Live players (``max_players``) + the dummy's slot. The per-track array width stays
        ``max_players``; only the slot count grows."""
        return self.settings.max_players + 1

    def set_light_frame(self, frame: LightFrame) -> None:
        """Store the latest LightFrame; thread-safe."""
        with self._input_lock:
            self._light_frame = frame

    def _build_bundle(self, frames: FrameDict, seq_state: SequencerState | None) -> OscBundleBuilder:
        """Live slots and the dummy's; per-track arrays stay ``max_players`` wide (see class doc)."""
        bundle = OscBundleBuilder(IMMEDIATELY)  # type: ignore
        self._add_global_messages(bundle, frames, seq_state)

        track = self.settings.max_players   # per-track (similarity/gate/leader) array width
        self.settings.active_players = sum(1 for id in range(self._slot_count) if id in frames)

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
        """All-zero bundle across every slot."""
        bundle = OscBundleBuilder(IMMEDIATELY)  # type: ignore
        self._add_global_messages(bundle, {}, None)  # no frames, None → zeroed globals
        for id in range(self._slot_count):
            self._add_inactive_frame_messages(bundle, id, self.settings.max_players)
        self.settings.active_players = 0
        self._send_bundle(bundle)

    def _add_global_messages(self, bundle_builder: OscBundleBuilder, frames: FrameDict, seq_state: SequencerState | None) -> None:
        with self._input_lock:
            light_frame = self._light_frame

        super()._add_global_messages(bundle_builder, frames, seq_state)

        # seq_state is None during idle/blackout → zero the playhead and motor.
        idle: bool = seq_state is None
        playhead: float = 0.0 if (idle or light_frame is None) else float(light_frame.playhead)
        playhead_msg = OscMessageBuilder(address="/global/playhead")
        playhead_msg.add_arg(playhead, OscMessageBuilder.ARG_TYPE_FLOAT)
        bundle_builder.add_content(playhead_msg.build())  # type: ignore

        motor_mode: int = 0 if (idle or light_frame is None) else int(light_frame.motor_command.mode)
        motor_msg = OscMessageBuilder(address="/global/motor")
        motor_msg.add_arg(motor_mode, OscMessageBuilder.ARG_TYPE_INT)
        bundle_builder.add_content(motor_msg.build())  # type: ignore

        # Group sync's envelope (0..1): the gate through the strobe's envelope, 0 while the pose instrument is
        # not drawn — the compositor only carries it out of a rendered frame.
        sync: float = 0.0 if (idle or light_frame is None) else float(light_frame.sync)
        sync_msg = OscMessageBuilder(address="/global/sync")
        sync_msg.add_arg(sync, OscMessageBuilder.ARG_TYPE_FLOAT)
        bundle_builder.add_content(sync_msg.build())  # type: ignore

        # Presence, not show state, so never zeroed on idle: the count of live players,
        # the dummy excluded — the same frames the active flags are built from.
        players: int = sum(1 for id in range(self.settings.max_players) if id in frames)
        players_msg = OscMessageBuilder(address="/global/players")
        players_msg.add_arg(players, OscMessageBuilder.ARG_TYPE_INT)
        bundle_builder.add_content(players_msg.build())  # type: ignore

        # Operator settings, not show state, so they are never zeroed on idle: the fader and
        # the speaker placement must read true whenever they are turned, show or no show.
        volume_msg = OscMessageBuilder(address="/global/volume")
        volume_msg.add_arg(float(self.settings.volume), OscMessageBuilder.ARG_TYPE_FLOAT)  # type: ignore[attr-defined]
        bundle_builder.add_content(volume_msg.build())  # type: ignore

        # Degrees in the panel, radians on the wire — the contract every azimuth here keeps.
        speaker_msg = OscMessageBuilder(address="/global/speaker/offset")
        speaker_msg.add_arg(math.radians(self.settings.speaker_offset), OscMessageBuilder.ARG_TYPE_FLOAT)  # type: ignore[attr-defined]
        bundle_builder.add_content(speaker_msg.build())  # type: ignore

    def _angle_values(self, frame: Frame) -> list[float]:
        """The angles, the four arm entries replaced by the joints' travel × π (see the class doc)."""
        values = frame[Angles].values.astype(np.float64)
        if ArmTravel in frame:
            values[:TravelElement.left_arm] = frame[ArmTravel].values[:TravelElement.left_arm] * math.pi
        return values.tolist()

    def _add_active_frame_messages(self, bundle_builder: OscBundleBuilder, frame: Frame, frames: FrameDict, max_players: int) -> None:
        super()._add_active_frame_messages(bundle_builder, frame, frames, max_players)
        id: int = frame.track_id

        # range [0, 1] - the whole arms: left, right (each the union of its shoulder's and its
        # elbow's travel) and their mean
        arm_values = frame[ArmTravel].values[TravelElement.left_arm:] if ArmTravel in frame else (0.0, 0.0, 0.0)
        arm_msg = OscMessageBuilder(address=f"/pose/{id}/arm")
        for value in arm_values:
            arm_msg.add_arg(float(value), OscMessageBuilder.ARG_TYPE_FLOAT)
        bundle_builder.add_content(arm_msg.build())  # type: ignore

        # range [-1, 1] - the shoulders' signed travel difference, left less right: the higher arm
        difference: float = frame[ShoulderDifference].value if ShoulderDifference in frame else 0.0
        difference_msg = OscMessageBuilder(address=f"/pose/{id}/shoulder/difference")
        difference_msg.add_arg(float(difference), OscMessageBuilder.ARG_TYPE_FLOAT)
        bundle_builder.add_content(difference_msg.build())  # type: ignore

        azimuth: float = frame[Azimuth].value if Azimuth in frame else np.nan
        azimuth_msg = OscMessageBuilder(address=f"/pose/{id}/azimuth")
        azimuth_msg.add_arg(azimuth, OscMessageBuilder.ARG_TYPE_FLOAT)
        bundle_builder.add_content(azimuth_msg.build())  # type: ignore

        # range [0, 1] - 0 at the zone's near edge, 1 at its far edge
        distance: float = frame[Distance].value if Distance in frame else np.nan
        distance_msg = OscMessageBuilder(address=f"/pose/{id}/distance")
        distance_msg.add_arg(distance, OscMessageBuilder.ARG_TYPE_FLOAT)
        bundle_builder.add_content(distance_msg.build())  # type: ignore

        playhead_offset: float = frame[PlayheadOffset].value if PlayheadOffset in frame else np.nan
        offset_msg = OscMessageBuilder(address=f"/pose/{id}/playhead/offset")
        offset_msg.add_arg(playhead_offset, OscMessageBuilder.ARG_TYPE_FLOAT)
        bundle_builder.add_content(offset_msg.build())  # type: ignore

    def _add_inactive_frame_messages(self, bundle_builder: OscBundleBuilder, id: int, max_players: int) -> None:
        super()._add_inactive_frame_messages(bundle_builder, id, max_players)
        arm_msg = OscMessageBuilder(address=f"/pose/{id}/arm")
        for _ in range(3):
            arm_msg.add_arg(0.0, OscMessageBuilder.ARG_TYPE_FLOAT)
        bundle_builder.add_content(arm_msg.build())  # type: ignore
        for address in (f"/pose/{id}/shoulder/difference", f"/pose/{id}/azimuth", f"/pose/{id}/distance",
                        f"/pose/{id}/playhead/offset"):
            msg = OscMessageBuilder(address=address)
            msg.add_arg(0.0, OscMessageBuilder.ARG_TYPE_FLOAT)
            bundle_builder.add_content(msg.build())  # type: ignore
