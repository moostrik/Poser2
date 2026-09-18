"""The sound sender's per-pose messages of its own: what White Space adds to the modules' bundle.

The distance is the panoramic tracker's, 0 at the zone's near edge and 1 at its far edge, and goes
out as it is on the frame. A slot that empties resets it to 0 with the rest.
"""

import math
import unittest

from pythonosc.osc_bundle_builder import OscBundleBuilder, IMMEDIATELY

from apps.white_space.inout.osc_sound_sender import OscSoundSender
from apps.white_space.settings import _OscSoundSettings
from modules.pose.features import Azimuth, Distance
from modules.pose.frame import Frame

ID = 2


def _messages(build) -> dict[str, tuple]:
    """Address → arguments, for the messages ``build`` adds to a bundle."""
    builder = OscBundleBuilder(IMMEDIATELY)
    build(builder)
    bundle = builder.build()
    return {bundle.content(i).address: tuple(bundle.content(i).params)
            for i in range(bundle.num_contents)}


class SoundPosesTest(unittest.TestCase):
    def setUp(self) -> None:
        self.settings = _OscSoundSettings()
        self.sender = OscSoundSender(self.settings)

    def _active(self, frame: Frame) -> dict[str, tuple]:
        return _messages(lambda b: self.sender._add_active_frame_messages(b, frame, {ID: frame}, self.settings.max_players))

    def test_the_distance_is_sent_as_it_is_on_the_frame(self) -> None:
        frame = Frame(ID, 0, features={Azimuth: Azimuth.from_value(1.0), Distance: Distance.from_value(0.25)})
        messages = self._active(frame)
        self.assertAlmostEqual(messages[f"/pose/{ID}/distance"][0], 0.25, places=6)
        self.assertAlmostEqual(messages[f"/pose/{ID}/azimuth"][0], 1.0, places=6)

    def test_without_a_reading_the_distance_is_nan(self) -> None:
        messages = self._active(Frame(ID, 0))
        self.assertTrue(math.isnan(messages[f"/pose/{ID}/distance"][0]))     # as the azimuth is without one

    def test_an_emptied_slot_resets_the_distance(self) -> None:
        messages = _messages(lambda b: self.sender._add_inactive_frame_messages(b, ID, self.settings.max_players))
        self.assertEqual(messages[f"/pose/{ID}/distance"], (0.0,))
        self.assertEqual(messages[f"/pose/{ID}/active"], (0,))


if __name__ == "__main__":
    unittest.main()
