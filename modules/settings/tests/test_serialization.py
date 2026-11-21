"""to_dict and update_from_dict: what is written, what is skipped, what a bad value does."""

import unittest

from modules.settings import Field, Widget, BaseSettings
from modules.utils import Color

from .fixtures import RenderMode, CameraSettings, SettingsWithActions, InnerSettings, OuterSettings


class TestSerialization(unittest.TestCase):
    """Tests for to_dict / update_from_dict."""

    def test_to_dict_excludes_read_access(self):
        s = CameraSettings()
        d = s.to_dict()
        self.assertNotIn("fps", d)

    def test_to_dict_excludes_a_field_not_saved(self):
        class Marked(BaseSettings):
            kept = Field(1)
            elsewhere = Field(2, saved=False)
        self.assertEqual(Marked().to_dict(), {"kept": 1})

    def test_to_dict_includes_writable(self):
        s = CameraSettings()
        d = s.to_dict()
        self.assertIn("exposure", d)
        # INIT fields are included in serialization (editable in JSON)
        self.assertIn("resolution", d)

    def test_to_dict_values(self):
        s = CameraSettings(exposure=2000)
        d = s.to_dict()
        self.assertEqual(d["exposure"], 2000)

    def test_to_dict_custom_type_uses_to_dict(self):
        s = CameraSettings()
        d = s.to_dict()
        self.assertEqual(d["overlay_color"], {"r": 1, "g": 0, "b": 0, "a": 1.0})

    def test_to_dict_enum_uses_name(self):
        s = CameraSettings()
        d = s.to_dict()
        self.assertEqual(d["mode"], "SOLID")

    def test_round_trip(self):
        s1 = CameraSettings(exposure=4000)
        s1.overlay_color = Color(0.1, 0.2, 0.3)
        s1.mode = RenderMode.TEXTURED
        data = s1.to_dict()

        s2 = CameraSettings()
        s2.update_from_dict(data)
        self.assertEqual(s2.exposure, 4000)
        # init_only fields are serialized but not restored after init
        self.assertEqual(s2.resolution, 1080)  # stays at default
        self.assertEqual(s2.overlay_color, Color(0.1, 0.2, 0.3))
        self.assertEqual(s2.mode, RenderMode.TEXTURED)

    def test_update_from_dict_fires_callbacks(self):
        s = CameraSettings()
        results = []
        s.bind(CameraSettings.exposure, lambda v: results.append(v))
        s.update_from_dict({"exposure": 7777})
        self.assertEqual(results, [7777])

    def test_update_from_dict_ignores_unknown_keys(self):
        s = CameraSettings()
        with self.assertLogs(level="WARNING") as cm:
            s.update_from_dict({"unknown_key": 42})
        self.assertTrue(any("unknown key 'unknown_key'" in m for m in cm.output))

    def test_update_from_dict_skips_bad_value_continues(self):
        """M1: one bad value should not prevent the rest from loading."""
        s = CameraSettings()
        s.update_from_dict({
            "exposure": "not_a_number",   # bad — should be skipped
            "gain": 8.0,                  # good — should still apply
        })
        self.assertEqual(s.exposure, 1000)   # unchanged (bad value skipped)
        self.assertEqual(s.gain, 8.0)        # applied despite earlier bad value

    def test_update_from_dict_bad_value_logs_warning(self):
        s = CameraSettings()
        with self.assertLogs(level="WARNING") as cm:
            s.update_from_dict({"exposure": "not_a_number"})
        self.assertTrue(any("exposure" in m for m in cm.output))


class TestToDictExclusions(unittest.TestCase):
    """to_dict must exclude readonly AND Widget.button fields (#5)."""

    def test_button_excluded(self):
        s = SettingsWithActions()
        d = s.to_dict()
        self.assertNotIn("reset", d)
        self.assertNotIn("hidden_action", d)

    def test_normal_field_included(self):
        s = SettingsWithActions()
        d = s.to_dict()
        self.assertIn("exposure", d)


class TestUpdateFromDictEdgeCases(unittest.TestCase):
    """Non-dict child value, equality with children."""

    def test_non_dict_child_logs_warning(self):
        s = OuterSettings()
        with self.assertLogs(level="WARNING") as cm:
            s.update_from_dict({"inner": 42})
        self.assertTrue(any("expected dict for child 'inner'" in m for m in cm.output))
        # inner should be unchanged
        self.assertEqual(s.inner.speed, 1.0)


if __name__ == "__main__":
    unittest.main()
