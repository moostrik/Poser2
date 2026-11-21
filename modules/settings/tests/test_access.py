"""A field's access: READ written by code, INIT locked by initialize()."""

import unittest

from .fixtures import RenderMode, CameraSettings, InnerSettings, OuterSettings


class TestReadAccess(unittest.TestCase):
    """Tests for Setting.READ fields (GUI hint, not enforced by descriptor)."""

    def test_read_allows_set(self):
        s = CameraSettings()
        s.fps = 30.0
        self.assertEqual(s.fps, 30.0)

    def test_read_set_works(self):
        s = CameraSettings()
        s.fps = 29.97
        self.assertEqual(s.fps, 29.97)

    def test_read_callback_fires_on_set(self):
        s = CameraSettings()
        results = []
        s.bind(CameraSettings.fps, lambda v: results.append(v))
        s.fps = 60.0
        self.assertEqual(results, [60.0])


class TestInitAccess(unittest.TestCase):
    """Tests for Setting.INIT fields."""

    def test_init_set_during_construction(self):
        s = CameraSettings(resolution=720)
        self.assertEqual(s.resolution, 720)

    def test_init_raises_after_initialize(self):
        s = CameraSettings()
        s.initialize()
        with self.assertRaises(AttributeError):
            s.resolution = 480

    def test_init_writable_before_initialize(self):
        s = CameraSettings()
        s.resolution = 480
        self.assertEqual(s.resolution, 480)

    def test_init_skipped_by_update_from_dict_after_initialize(self):
        s = CameraSettings(resolution=720)
        # Before initialize: INIT fields are applied
        s.update_from_dict({"resolution": 480})
        self.assertEqual(s.resolution, 480)
        # Lock INIT fields
        s.initialize()
        # After initialize: INIT fields are skipped
        s.update_from_dict({"resolution": 1080})
        self.assertEqual(s.resolution, 480)

    def test_initialize_recurses_to_children(self):
        s = OuterSettings()
        self.assertFalse(s._initialized)
        self.assertFalse(s.inner._initialized)
        s.initialize()
        self.assertTrue(s._initialized)
        self.assertTrue(s.inner._initialized)


if __name__ == "__main__":
    unittest.main()
