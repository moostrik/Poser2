import numpy as np
from pythonosc.osc_bundle_builder import OscBundleBuilder
from pythonosc.osc_message_builder import OscMessageBuilder

from modules.inout import OscSound as BaseOscSound, OscSoundSettings
from modules.pose.frame import Frame as PoseFrame, FrameDict
from modules.pose.features import Azimuth, Distance
from modules.session import SequencerState
from .light import Frame
from .pose import PlayheadPhase

import logging
logger = logging.getLogger(__name__)


class OscSound(BaseOscSound):
    """OscSound extended with the rotation playhead (/global/playhead) and the
    panoramic-only per-pose azimuth, distance, and playhead-phase messages.

    Everything flows through the base's single-bundle send path by overriding the three
    leaf builders; there is no separate _send_data/_send_blackout."""

    def __init__(self, settings: OscSoundSettings) -> None:
        super().__init__(settings)
        self._composition: Frame | None = None

    def set_composition(self, output: Frame) -> None:
        """Store the latest Frame; thread-safe."""
        with self._input_lock:
            self._composition = output

    def _add_global_messages(self, bundle_builder: OscBundleBuilder, seq_state: SequencerState | None) -> None:
        super()._add_global_messages(bundle_builder, seq_state)

        with self._input_lock:
            composition = self._composition

        # seq_state is None during idle/blackout → zero the playhead and emit no hits.
        idle: bool = seq_state is None
        playhead: float = 0.0 if (idle or composition is None) else float(composition.motor.playhead)
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

        playhead_phase: float = frame[PlayheadPhase].value if PlayheadPhase in frame else np.nan
        phase_msg = OscMessageBuilder(address=f"/pose/{id}/playhead_phase")
        phase_msg.add_arg(playhead_phase, OscMessageBuilder.ARG_TYPE_FLOAT)
        bundle_builder.add_content(phase_msg.build())  # type: ignore

    def _add_inactive_frame_messages(self, bundle_builder: OscBundleBuilder, id: int, num_players: int) -> None:
        super()._add_inactive_frame_messages(bundle_builder, id, num_players)
        for address in (f"/pose/{id}/azimuth", f"/pose/{id}/distance", f"/pose/{id}/playhead_phase"):
            msg = OscMessageBuilder(address=address)
            msg.add_arg(0.0, OscMessageBuilder.ARG_TYPE_FLOAT)
            bundle_builder.add_content(msg.build())  # type: ignore
