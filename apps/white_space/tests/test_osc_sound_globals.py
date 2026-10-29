"""The sound sender's global messages.

The show state (playhead, motor) is zeroed when there is no sequencer state — that is the
blackout. The two operator settings are not show state and are never zeroed: a fader that
reads 0 while someone is turning it, or a speaker calibration that snaps to 0 between shows,
would both be wrong. `speaker_offset` is degrees in the panel and radians on the wire, the
contract every azimuth here keeps. The player count is presence data, built from the tick's
frames: live slots only, true show or no show, 0 only when the frames are gone (the blackout).
"""

import math
import unittest

from pythonosc.osc_bundle_builder import OscBundleBuilder, IMMEDIATELY

from modules.pose import Frame as PoseFrame, FrameDict
from modules.session import SequencerState

from apps.white_space.inout.osc_sound_sender import OscSoundSender
from apps.white_space.light import Tick
from apps.white_space.light.frame import Frame
from apps.white_space.settings import _OscSoundSettings


def _globals(sender: OscSoundSender, seq_state=None, frames: FrameDict = {}) -> dict[str, float]:
    """Address → first argument, for the bundle's global messages."""
    builder = OscBundleBuilder(IMMEDIATELY)
    sender._add_global_messages(builder, frames, seq_state)
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

    def test_sync_is_the_compositions_and_zeroed_on_idle(self) -> None:
        """/global/sync carries the frame's full-sync envelope while a show runs; it is show
        state, so idle (no sequencer state) zeroes it like the playhead and the motor."""
        composition = Frame(360, Tick(0.0, 1 / 32, index=0))
        composition.sync = 0.7
        self.sender.set_composition(composition)
        seq = SequencerState(stage=4, stage_progress=0.5, progress=0.5, elapsed=10.0, active=True)
        self.assertAlmostEqual(_globals(self.sender, seq)["/global/sync"], 0.7, places=6)
        self.assertEqual(_globals(self.sender, None)["/global/sync"], 0.0)

    def test_players_counts_only_live_slots(self) -> None:
        """/global/players counts the occupied live slots; the dummy doesn't count."""
        dummy = self.settings.max_players
        frames = {id: PoseFrame(id, 0) for id in (0, 2, dummy, dummy + 1)}
        seq = SequencerState(stage=4, stage_progress=0.5, progress=0.5, elapsed=10.0, active=True)
        self.assertEqual(_globals(self.sender, seq, frames)["/global/players"], 2)

    def test_players_is_presence_and_survives_idle(self) -> None:
        """The count is presence data, not show state: no sequencer state leaves it true."""
        frames = {0: PoseFrame(0, 0)}
        self.assertEqual(_globals(self.sender, None, frames)["/global/players"], 1)

    def test_players_is_zero_without_frames(self) -> None:
        """The blackout builds its bundle from no frames, so the count reads 0."""
        self.assertEqual(_globals(self.sender, None, {})["/global/players"], 0)

    def test_defaults_are_full_volume_and_no_offset(self) -> None:
        """Speakers placed by the fixed layout need no constant at all (see CALIBRATION.md)."""
        messages = _globals(self.sender)
        self.assertAlmostEqual(messages["/global/volume"], 1.0, places=6)
        self.assertEqual(messages["/global/speaker/offset"], 0.0)


if __name__ == "__main__":
    unittest.main()
