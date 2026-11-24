"""A rig camera's settings: its readings' one value that is set and saved, `roll_offset`."""

import unittest

from modules.camera import RigCameraReadings, RigCameraSettings


class TestRollOffsetIsABox(unittest.TestCase):
    """A box, not a slider. `Widget.resolve` sends a bounded float to `slider`, so this is one
    stray `min=` away from silently becoming one — worth a test rather than a comment."""

    def test_it_resolves_to_a_number_box(self) -> None:
        from modules.settings import Widget
        field = RigCameraReadings.roll_offset
        self.assertEqual(Widget.resolve(field), Widget.number)
        self.assertIsNone(field.min)
        self.assertIsNone(field.max)

    def test_it_is_recorded_in_the_settings_file(self) -> None:
        """It is a measurement, so unlike the rest of `readings` it has to survive a save."""
        camera = RigCameraSettings()
        camera.readings.roll_offset = -2.2458989103710865
        self.assertIn('roll_offset', camera.to_dict()['readings'])
        self.assertEqual(camera.to_dict()['readings']['roll_offset'], -2.2458989103710865)

    def test_the_rest_of_readings_stays_out_of_the_settings_file(self) -> None:
        stored = RigCameraSettings().to_dict()['readings']
        for read_only in ('video_fps', 'tilt_measured', 'roll_measured', 'fov_factory'):
            self.assertNotIn(read_only, stored)


if __name__ == '__main__':
    unittest.main()
