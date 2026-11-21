"""The sound sender's per-pose messages of its own: what White Space adds to the modules' bundle.

The distance is the panoramic tracker's, 0 at the zone's near edge and 1 at its far edge, and goes
out as it is on the frame. A slot that empties resets it to 0 with the rest. The four arm entries
of ``angle/rad`` are the arm travel times π; the other entries are the angles.
"""

import math
import unittest

import numpy as np

from apps.white_space.inout import SoundExtras
from apps.white_space.settings import _OscSoundSettings
from modules.inout import OscSound, OscMessage
from modules.pose import Azimuth, Distance, Angles, AngleLandmark, ArmTravel, TravelElement, ShoulderDifference, Frame

ID = 2


def _messages(messages: list[OscMessage]) -> dict[str, tuple]:
    """Address → arguments, a row's values each one argument as on the wire."""
    return {address: tuple(value for arg in args for value in (arg.tolist() if isinstance(arg, np.ndarray) else (arg,)))
            for address, args in messages}


class SoundPosesTest(unittest.TestCase):
    def setUp(self) -> None:
        self.settings = _OscSoundSettings()
        extras = SoundExtras(self.settings)
        self.sender = OscSound(self.settings, extras.slot_count, extras)

    def _active(self, frame: Frame) -> dict[str, tuple]:
        return _messages(self.sender._active_messages(frame))

    def _inactive(self) -> dict[str, tuple]:
        return _messages(self.sender._inactive_messages(ID))

    def test_the_distance_is_sent_as_it_is_on_the_frame(self) -> None:
        frame = Frame(ID, 0, 0.0, features={Azimuth: Azimuth.from_value(1.0), Distance: Distance.from_value(0.25)})
        messages = self._active(frame)
        self.assertAlmostEqual(messages[f"/pose/{ID}/distance"][0], 0.25, places=6)
        self.assertAlmostEqual(messages[f"/pose/{ID}/azimuth"][0], 1.0, places=6)

    def test_without_a_reading_the_distance_is_nan(self) -> None:
        messages = self._active(Frame(ID, 0, 0.0))
        self.assertTrue(math.isnan(messages[f"/pose/{ID}/distance"][0]))     # as the azimuth is without one

    def test_the_arm_entries_of_the_angles_are_the_joints_travel_times_pi(self) -> None:
        angles = np.full(len(AngleLandmark), 0.3, dtype=np.float32)
        travel = np.array([0.0, 1.0, -0.5, np.nan, 0.6, np.nan, np.nan], dtype=np.float32)
        scores = np.array([1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float32)
        frame = Frame(ID, 0, 0.0, features={
            Angles: Angles(angles, np.ones(len(AngleLandmark), dtype=np.float32)),
            ArmTravel: ArmTravel(travel, scores),
        })
        sent = self._active(frame)[f"/pose/{ID}/angle/rad"]
        self.assertEqual(len(sent), len(AngleLandmark))
        self.assertAlmostEqual(sent[TravelElement.left_shoulder], 0.0, places=6)
        self.assertAlmostEqual(sent[TravelElement.right_shoulder], math.pi, places=6)
        self.assertAlmostEqual(sent[TravelElement.left_elbow], -math.pi / 2, places=6)
        self.assertTrue(math.isnan(sent[TravelElement.right_elbow]))
        for i in range(TravelElement.left_arm, len(AngleLandmark)):
            self.assertAlmostEqual(sent[i], 0.3, places=6)                # the angles as they are: the arms are no angles

    def test_the_arms_and_the_difference_are_sent_as_on_the_frame(self) -> None:
        travel = np.array([0.0, 0.5, 0.0, 1.0, 0.25, 0.5, 0.375], dtype=np.float32)
        frame = Frame(ID, 0, 0.0, features={
            ArmTravel: ArmTravel(travel, np.ones(len(TravelElement), dtype=np.float32)),
            ShoulderDifference: ShoulderDifference.from_value(-0.5),
        })
        messages = self._active(frame)
        self.assertEqual(messages[f"/pose/{ID}/arm"], (0.25, 0.5, 0.375))        # left arm, right arm, their mean
        self.assertAlmostEqual(messages[f"/pose/{ID}/shoulder/difference"][0], -0.5, places=6)

    def test_without_the_features_the_arms_and_the_difference_are_zero(self) -> None:
        messages = self._active(Frame(ID, 0, 0.0))
        self.assertEqual(messages[f"/pose/{ID}/arm"], (0.0, 0.0, 0.0))
        self.assertEqual(messages[f"/pose/{ID}/shoulder/difference"], (0.0,))

    def test_an_emptied_slot_resets_the_arms_and_the_difference(self) -> None:
        messages = self._inactive()
        self.assertEqual(messages[f"/pose/{ID}/arm"], (0.0, 0.0, 0.0))
        self.assertEqual(messages[f"/pose/{ID}/shoulder/difference"], (0.0,))

    def test_without_a_travel_the_angles_go_out_as_they_are(self) -> None:
        angles = np.full(len(AngleLandmark), 0.3, dtype=np.float32)
        frame = Frame(ID, 0, 0.0, features={Angles: Angles(angles, np.ones(len(AngleLandmark), dtype=np.float32))})
        sent = self._active(frame)[f"/pose/{ID}/angle/rad"]
        self.assertAlmostEqual(sent[0], 0.3, places=6)

    def test_an_emptied_slot_resets_the_distance(self) -> None:
        messages = self._inactive()
        self.assertEqual(messages[f"/pose/{ID}/distance"], (0.0,))
        self.assertEqual(messages[f"/pose/{ID}/active"], (0,))

    def test_an_emptied_slot_resets_the_angles_with_the_length_a_player_sends(self) -> None:
        angles = np.full(len(AngleLandmark), 0.3, dtype=np.float32)
        frame = Frame(ID, 0, 0.0, features={Angles: Angles(angles, np.ones(len(AngleLandmark), dtype=np.float32))})
        live = self._active(frame)
        reset = self._inactive()
        for address in (f"/pose/{ID}/angle/rad", f"/pose/{ID}/angle/vel"):
            with self.subTest(address=address):
                self.assertEqual(len(reset[address]), len(live[address]))
                self.assertTrue(all(value == 0.0 for value in reset[address]))


if __name__ == "__main__":
    unittest.main()
