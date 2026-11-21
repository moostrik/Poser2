import math
from threading import Lock
from typing import TYPE_CHECKING

import numpy as np

from modules.inout import OscMessage
from modules.pose import Frame, FrameDict, Azimuth, Distance, Angles, ArmTravel, TravelElement, ShoulderDifference
from modules.session import SequencerState
from ..light import LightFrame
from ..pose import PlayheadOffset

if TYPE_CHECKING:
    from ..settings import _OscSoundSettings     # the app's settings import this package


class SoundExtras:
    """White Space's own messages in the sound's bundle (``OscSoundExtras``): the rotation playhead
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

    The sound sends ``max_players`` live slots plus the dummy's (``slot_count``); the per-player
    rows (similarity, gate, leader) stay ``max_players`` wide."""

    def __init__(self, settings: '_OscSoundSettings') -> None:
        self.settings: '_OscSoundSettings' = settings
        self._lock: Lock = Lock()
        self._light_frame: LightFrame | None = None

    @property
    def slot_count(self) -> int:
        """Live players (``max_players``) + the dummy's slot."""
        return self.settings.max_players + 1

    def set_light_frame(self, frame: LightFrame) -> None:
        """Store the latest LightFrame; thread-safe."""
        with self._lock:
            self._light_frame = frame

    def global_messages(self, frames: FrameDict, seq_state: SequencerState | None) -> list[OscMessage]:
        with self._lock:
            light_frame = self._light_frame

        # seq_state is None during idle/blackout → zero the playhead, the motor and the sync.
        idle: bool = seq_state is None
        return [
            ("/global/playhead", (0.0 if idle or light_frame is None else float(light_frame.playhead),)),
            ("/global/motor", (0 if idle or light_frame is None else int(light_frame.motor_command.mode),)),
            # Group sync's envelope (0..1): the gate through the strobe's envelope, 0 while the pose instrument is
            # not drawn — the compositor only carries it out of a rendered frame.
            ("/global/sync", (0.0 if idle or light_frame is None else float(light_frame.sync),)),
            # Presence, not show state, so never zeroed on idle: the count of live players,
            # the dummy excluded — the same frames the active flags are built from.
            ("/global/players", (sum(1 for track_id in range(self.settings.max_players) if track_id in frames),)),
            # Operator settings, not show state, so they are never zeroed on idle: the fader and
            # the speaker placement must read true whenever they are turned, show or no show.
            ("/global/volume", (float(self.settings.volume),)),
            # Degrees in the panel, radians on the wire — the contract every azimuth here keeps.
            ("/global/speaker/offset", (math.radians(self.settings.speaker_offset),)),
        ]

    def angle_values(self, frame: Frame) -> np.ndarray:
        """The angles, the four arm entries replaced by the joints' travel × π (see the class doc)."""
        values = frame[Angles].values.astype(np.float64)
        if ArmTravel in frame:
            values[:TravelElement.left_arm] = frame[ArmTravel].values[:TravelElement.left_arm] * math.pi
        return values

    def player_messages(self, frame: Frame) -> list[OscMessage]:
        track_id: int = frame.track_id
        arms: np.ndarray = frame[ArmTravel].values[TravelElement.left_arm:] if ArmTravel in frame else np.zeros(3, dtype=np.float32)
        return [
            # range [0, 1] - the whole arms: left, right (each the union of its shoulder's and its
            # elbow's travel) and their mean
            (f"/pose/{track_id}/arm", (arms,)),
            # range [-1, 1] - the shoulders' signed travel difference, left less right: the higher arm
            (f"/pose/{track_id}/shoulder/difference", (float(frame[ShoulderDifference].value) if ShoulderDifference in frame else 0.0,)),
            (f"/pose/{track_id}/azimuth", (float(frame[Azimuth].value) if Azimuth in frame else np.nan,)),
            # range [0, 1] - 0 at the zone's near edge, 1 at its far edge
            (f"/pose/{track_id}/distance", (float(frame[Distance].value) if Distance in frame else np.nan,)),
            (f"/pose/{track_id}/playhead/offset", (float(frame[PlayheadOffset].value) if PlayheadOffset in frame else np.nan,)),
        ]

    def inactive_messages(self, track_id: int) -> list[OscMessage]:
        return [
            (f"/pose/{track_id}/arm", (np.zeros(3, dtype=np.float32),)),
            (f"/pose/{track_id}/shoulder/difference", (0.0,)),
            (f"/pose/{track_id}/azimuth", (0.0,)),
            (f"/pose/{track_id}/distance", (0.0,)),
            (f"/pose/{track_id}/playhead/offset", (0.0,)),
        ]
