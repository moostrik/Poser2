"""The sound sender's global messages.

The show state (playhead, motor) is zeroed when there is no sequencer state — that is the
blackout. The two operator settings are not show state and are never zeroed: a fader that
reads 0 while someone is turning it, or a speaker calibration that snaps to 0 between shows,
would both be wrong. `speaker_offset` is degrees in the panel and radians on the wire, the
contract every azimuth here keeps.
"""

import math
import unittest

from pythonosc.osc_bundle_builder import OscBundleBuilder, IMMEDIATELY

from apps.white_space.inout.osc_sound_sender import OscSoundSender
from apps.white_space.settings import _OscSoundSettings


def _globals(sender: OscSoundSender, seq_state=None) -> dict[str, float]:
    """Address → first argument, for the bundle's global messages."""
    builder = OscBundleBuilder(IMMEDIATELY)
    sender._add_global_messages(builder, seq_state)
    bundle = builder.build()
    return {bundle.content(i).address: bundle.content(i).params[0]
            for i in range(bundle.num_contents)}


class SoundGlobalsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.settings = _OscSoundSettings()
        self.sender = OscSoundSender(self.settings)

    def test_volume_and_speaker_offset_are_sent(self) -> None:
        messages = _globals(self.sender)
        self.assertIn("/global/volume", messages)
        self.assertIn("/global/speaker/offset", messages)

    def test_volume_carries_the_setting(self) -> None:
        self.settings.volume = 0.4
        self.assertAlmostEqual(_globals(self.sender)["/global/volume"], 0.4, places=6)

    def test_speaker_offset_is_degrees_in_and_radians_out(self) -> None:
        for degrees in (0.0, 90.0, 198.0, 359.0):
            self.settings.speaker_offset = degrees
            self.assertAlmostEqual(_globals(self.sender)["/global/speaker/offset"],
                                   math.radians(degrees), places=6)

    def test_operator_settings_survive_the_blackout(self) -> None:
        """`seq_state is None` zeroes the show state; the two settings must stand."""
        self.settings.volume = 0.8
        self.settings.speaker_offset = 45.0
        messages = _globals(self.sender, seq_state=None)
        self.assertEqual(messages["/global/playhead"], 0.0)          # show state: zeroed
        self.assertEqual(messages["/global/motor"], 0)               # show state: zeroed
        self.assertAlmostEqual(messages["/global/volume"], 0.8, places=6)
        self.assertAlmostEqual(messages["/global/speaker/offset"], math.radians(45.0), places=6)

    def test_defaults_are_full_volume_and_no_offset(self) -> None:
        """Speakers placed by the fixed layout need no constant at all (see CALIBRATION.md)."""
        messages = _globals(self.sender)
        self.assertAlmostEqual(messages["/global/volume"], 1.0, places=6)
        self.assertEqual(messages["/global/speaker/offset"], 0.0)


if __name__ == "__main__":
    unittest.main()
