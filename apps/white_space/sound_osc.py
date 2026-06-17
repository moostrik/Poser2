from pythonosc.osc_message_builder import OscMessageBuilder

from modules.inout import OscSound, OscSoundSettings
from .light import Frame

import logging
logger = logging.getLogger(__name__)


class WhiteSpaceSoundOsc(OscSound):
    """OscSound extended with a rotation playhead sent as /global/playhead."""

    def __init__(self, settings: OscSoundSettings) -> None:
        super().__init__(settings)
        self._composition: Frame | None = None

    def set_composition(self, output: Frame) -> None:
        """Store the latest Frame; thread-safe."""
        with self._input_lock:
            self._composition = output

    def _send_data(self) -> None:
        with self._input_lock:
            composition = self._composition

        playhead = composition.motor.playhead if composition is not None else 0.0
        msg = OscMessageBuilder(address="/global/playhead")
        msg.add_arg(float(playhead), OscMessageBuilder.ARG_TYPE_FLOAT)
        with self._client_lock:
            self._client.send(msg.build())  # type: ignore[arg-type]

        hits = composition.hits if composition is not None else ()
        for hit in hits:
            hit_msg = OscMessageBuilder(address="/playhead/hit")
            hit_msg.add_arg(hit.track_id, OscMessageBuilder.ARG_TYPE_INT)
            hit_msg.add_arg(float(hit.position), OscMessageBuilder.ARG_TYPE_FLOAT)
            with self._client_lock:
                self._client.send(hit_msg.build())  # type: ignore[arg-type]

        super()._send_data()
