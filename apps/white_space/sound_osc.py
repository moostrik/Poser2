from pythonosc.osc_message_builder import OscMessageBuilder

from modules.inout import OscSound, OscSoundSettings
from .composition import CompositionOutput

import logging
logger = logging.getLogger(__name__)


class WhiteSpaceSoundOsc(OscSound):
    """OscSound extended with a rotation playhead sent as /global/playhead."""

    def __init__(self, settings: OscSoundSettings) -> None:
        super().__init__(settings)
        self._composition: CompositionOutput | None = None

    def set_composition(self, output: CompositionOutput) -> None:
        """Store the latest CompositionOutput; thread-safe."""
        with self._input_lock:
            self._composition = output

    def _send_data(self) -> None:
        with self._input_lock:
            composition = self._composition

        playhead = composition.playhead if composition is not None else 0.0
        msg = OscMessageBuilder(address="/global/playhead")
        msg.add_arg(float(playhead), OscMessageBuilder.ARG_TYPE_FLOAT)
        with self._client_lock:
            self._client.send(msg.build())  # type: ignore[arg-type]

        super()._send_data()
