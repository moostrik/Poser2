"""Tests for the reactive settings system."""

import json
import os
import tempfile
import threading
import unittest
from enum import Enum

from modules.settings.field import Field
from modules.settings.widget import Widget
from modules.settings.settings import Settings
from modules.settings.group import Group
from modules.settings import presets
from modules.utils import Color, Point2f, Rect


# ---------------------------------------------------------------------------
# Helper types
# ---------------------------------------------------------------------------

class RenderMode(Enum):
    WIREFRAME = "wireframe"
    SOLID = "solid"
    TEXTURED = "textured"


# ---------------------------------------------------------------------------
# Test settings classes
# ---------------------------------------------------------------------------

class CameraSettings(Settings):
    exposure = Field(1000, min=100, max=10000, step=100, description="Exposure time (µs)")
    gain = Field(1.0, min=0.0, max=16.0, step=0.1)
    overlay_color = Field(Color(1, 0, 0))
    resolution = Field(1080, access=Field.INIT)
    fps = Field(0.0, access=Field.READ, visible=False)
    mode = Field(RenderMode.SOLID)


class MinimalSettings(Settings):
    value = Field(0)


class SettingsWithActions(Settings):
    exposure = Field(1000)
    reset = Field(False, widget=Widget.button, description="Reset all values")
    hidden_action = Field(False, widget=Widget.button, visible=False)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSettingDescriptor(unittest.TestCase):
    """Tests for the Field descriptor itself."""

    def test_default_value(self):
        s = CameraSettings()
        self.assertEqual(s.exposure, 1000)
        self.assertEqual(s.gain, 1.0)
        self.assertEqual(s.overlay_color, Color(1, 0, 0))
        self.assertEqual(s.fps, 0.0)

    def test_set_and_get(self):
        s = CameraSettings()
        s.exposure = 2000
        self.assertEqual(s.exposure, 2000)

    def test_kwargs_override(self):
        s = CameraSettings(exposure=500, resolution=720)
        self.assertEqual(s.exposure, 500)
        self.assertEqual(s.resolution, 720)

    def test_kwargs_unknown_raises(self):
        with self.assertRaises(TypeError):
            CameraSettings(nonexistent=42)

    def test_class_access_returns_descriptor(self):
        self.assertIsInstance(CameraSettings.exposure, Field)


class TestTypeCoercion(unittest.TestCase):
    """Tests for type enforcement and coercion."""

    def test_direct_instance(self):
        s = CameraSettings()
        s.overlay_color = Color(0, 1, 0)
        self.assertEqual(s.overlay_color, Color(0, 1, 0))

    def test_tuple_coercion(self):
        s = CameraSettings()
        s.overlay_color = (0.5, 0.5, 0.5) # type: ignore
        self.assertEqual(s.overlay_color, Color(0.5, 0.5, 0.5))

    def test_list_coercion(self):
        s = CameraSettings()
        s.overlay_color = [0.2, 0.3, 0.4] # type: ignore
        self.assertEqual(s.overlay_color, Color(0.2, 0.3, 0.4))

    def test_dict_coercion_from_dict(self):
        s = CameraSettings()
        s.overlay_color = {"r": 0.1, "g": 0.2, "b": 0.3} # type: ignore
        self.assertEqual(s.overlay_color, Color(0.1, 0.2, 0.3))

    def test_wrong_type_raises(self):
        s = CameraSettings()
        with self.assertRaises(TypeError):
            s.exposure = "not a number"     # type: ignore

    def test_wrong_tuple_raises(self):
        s = MinimalSettings()
        with self.assertRaises(TypeError):
            s.value = (1, 2, 3)  # type: ignore

    def test_enum_direct(self):
        s = CameraSettings()
        s.mode = RenderMode.TEXTURED
        self.assertEqual(s.mode, RenderMode.TEXTURED)


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


class TestCallbacks(unittest.TestCase):
    """Tests for the callback system."""

    def test_callback_fires_on_change(self):
        s = CameraSettings()
        results = []
        s.bind(CameraSettings.exposure, lambda v: results.append(v))
        s.exposure = 2000
        self.assertEqual(results, [2000])

    def test_callback_not_fired_on_same_value(self):
        s = CameraSettings()
        results = []
        s.bind(CameraSettings.exposure, lambda v: results.append(v))
        s.exposure = 1000  # same as default
        self.assertEqual(results, [])

    def test_multiple_callbacks(self):
        s = CameraSettings()
        a, b = [], []
        s.bind(CameraSettings.exposure, lambda v: a.append(v))
        s.bind(CameraSettings.exposure, lambda v: b.append(v))
        s.exposure = 3000
        self.assertEqual(a, [3000])
        self.assertEqual(b, [3000])

    def test_remove_callback(self):
        s = CameraSettings()
        results = []
        cb = lambda v: results.append(v)
        s.bind(CameraSettings.exposure, cb)
        s.unbind(CameraSettings.exposure, cb)
        s.exposure = 5000
        self.assertEqual(results, [])

    def test_bind_unknown_field_raises(self):
        s = CameraSettings()
        with self.assertRaises(KeyError):
            s.bind(MinimalSettings.value, lambda v: None)


class TestAttributeGuard(unittest.TestCase):
    """Tests that setting non-existent attributes raises."""

    def test_set_nonexistent_raises(self):
        s = CameraSettings()
        with self.assertRaises(AttributeError):
            s.typo = 42

    def test_get_nonexistent_raises(self):
        s = CameraSettings()
        with self.assertRaises(AttributeError):
            _ = s.typo


class TestInstanceIsolation(unittest.TestCase):
    """Tests that instances don't share state or callbacks."""

    def test_values_independent(self):
        a = CameraSettings()
        b = CameraSettings()
        a.exposure = 5000
        self.assertEqual(a.exposure, 5000)
        self.assertEqual(b.exposure, 1000)  # unchanged

    def test_callbacks_independent(self):
        a = CameraSettings()
        b = CameraSettings()
        results_a, results_b = [], []
        a.bind(CameraSettings.exposure, lambda v: results_a.append(v))
        b.bind(CameraSettings.exposure, lambda v: results_b.append(v))
        a.exposure = 9000
        self.assertEqual(results_a, [9000])
        self.assertEqual(results_b, [])


class TestSerialization(unittest.TestCase):
    """Tests for to_dict / update_from_dict."""

    def test_to_dict_excludes_read_access(self):
        s = CameraSettings()
        d = s.to_dict()
        self.assertNotIn("fps", d)

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
        with self.assertLogs("modules.settings.settings", level="WARNING") as cm:
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
        with self.assertLogs("modules.settings.settings", level="WARNING") as cm:
            s.update_from_dict({"exposure": "not_a_number"})
        self.assertTrue(any("exposure" in m for m in cm.output))


class TestRepr(unittest.TestCase):
    """Tests for __repr__."""

    def test_repr_excludes_invisible(self):
        s = CameraSettings()
        r = repr(s)
        self.assertNotIn("fps", r)

    def test_repr_includes_visible(self):
        s = CameraSettings()
        r = repr(s)
        self.assertIn("exposure=1000", r)

    def test_repr_starts_with_class_name(self):
        s = CameraSettings()
        self.assertTrue(repr(s).startswith("CameraSettings("))


class TestThreadSafety(unittest.TestCase):
    """Tests for concurrent access from multiple threads."""

    def test_concurrent_writes(self):
        s = MinimalSettings()
        errors = []

        def writer(n):
            try:
                for i in range(500):
                    s.value = n * 1000 + i
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer, args=(t,)) for t in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(errors, [])
        # Value should be one of the written values (deterministic only in that it's an int)
        self.assertIsInstance(s.value, int)

    def test_concurrent_callbacks(self):
        s = MinimalSettings()
        counter = {"n": 0}
        lock = threading.Lock()

        def increment(v):
            with lock:
                counter["n"] += 1

        s.bind(MinimalSettings.value, increment)

        def writer(start):
            for i in range(100):
                s.value = start + i

        threads = [threading.Thread(target=writer, args=(t * 1000,)) for t in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # At least some callbacks should have fired (exact count depends on ordering)
        self.assertGreater(counter["n"], 0)


class TestPresetSaveLoad(unittest.TestCase):
    """Tests for preset save/load using a root Settings."""

    def _make_root(self):
        class Root(Settings):
            camera = Group(CameraSettings)
        return Root()

    def test_save_and_load(self):
        root = self._make_root()
        root.camera.exposure = 3000
        root.camera.mode = RenderMode.WIREFRAME

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "settings.json")
            presets.save(root, path)

            # Verify JSON file content
            with open(path, "r") as f:
                raw = json.load(f)
            self.assertIn("camera", raw)
            self.assertEqual(raw["camera"]["exposure"], 3000)

            # Load into fresh root
            root2 = self._make_root()
            presets.load(root2, path)

            self.assertEqual(root2.camera.exposure, 3000)
            # init_only fields stay at default
            self.assertEqual(root2.camera.resolution, 1080)
            self.assertEqual(root2.camera.mode, RenderMode.WIREFRAME)

    def test_load_fires_callbacks(self):
        root = self._make_root()
        results = []
        root.camera.bind(CameraSettings.exposure, lambda v: results.append(v))

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "settings.json")
            # Save with different value
            root.camera.exposure = 5000
            presets.save(root, path)
            results.clear()

            # Reset and load
            root.camera.exposure = 1000
            results.clear()
            presets.load(root, path)
            self.assertEqual(results, [5000])

    def test_children_accessible(self):
        root = self._make_root()
        self.assertIn("camera", root.children)
        self.assertIsInstance(root.camera, CameraSettings)

    def test_save_is_atomic(self):
        """M5: save writes to a temp file then atomically replaces."""
        root = self._make_root()
        root.camera.exposure = 4242

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "settings.json")
            # Save twice — second write should not corrupt the first
            presets.save(root, path)
            root.camera.exposure = 9999
            presets.save(root, path)

            with open(path, "r") as f:
                raw = json.load(f)
            self.assertEqual(raw["camera"]["exposure"], 9999)

            # No leftover .tmp files
            tmp_files = [f for f in os.listdir(tmpdir) if f.endswith(".tmp")]
            self.assertEqual(tmp_files, [])


class TestSettingMetadata(unittest.TestCase):
    """Tests that metadata attributes are accessible."""

    def test_min_max_step(self):
        field = CameraSettings.gain  # class-level access returns descriptor
        self.assertEqual(field.min, 0.0)
        self.assertEqual(field.max, 16.0)
        self.assertEqual(field.step, 0.1)

    def test_description(self):
        field = CameraSettings.exposure
        self.assertEqual(field.description, "Exposure time (µs)")

    def test_metadata_via_instance_fields(self):
        s = CameraSettings()
        field = s._fields["exposure"]
        self.assertEqual(field.min, 100)
        self.assertEqual(field.max, 10000)


class TestMinMaxMetadata(unittest.TestCase):
    """Tests that min/max/step are stored as GUI metadata, not enforced."""

    def test_value_above_max_allowed(self):
        s = CameraSettings()
        s.exposure = 99999
        self.assertEqual(s.exposure, 99999)

    def test_value_below_min_allowed(self):
        s = CameraSettings()
        s.exposure = 1
        self.assertEqual(s.exposure, 1)

    def test_min_max_accessible_as_metadata(self):
        s = CameraSettings()
        field = s.fields["exposure"]
        self.assertEqual(field.min, 100)
        self.assertEqual(field.max, 10000)

    def test_step_accessible_as_metadata(self):
        s = CameraSettings()
        field = s.fields["exposure"]
        self.assertEqual(field.step, 100)


class TestBoolIntSafety(unittest.TestCase):
    """Tests that bool is rejected when int is expected."""

    def test_bool_rejected_for_int(self):
        s = MinimalSettings()
        with self.assertRaises(TypeError):
            s.value = True

    def test_bool_rejected_for_int_false(self):
        s = MinimalSettings()
        with self.assertRaises(TypeError):
            s.value = False


class TestDeleteGuard(unittest.TestCase):
    """Tests that deleting settings raises."""

    def test_delete_setting_raises(self):
        s = CameraSettings()
        with self.assertRaises(AttributeError):
            del s.exposure

    def test_delete_private_ok(self):
        s = CameraSettings()
        s._temp = 42
        del s._temp


class TestFieldsProperty(unittest.TestCase):
    """Tests for the public fields property."""

    def test_fields_returns_dict(self):
        s = CameraSettings()
        f = s.fields
        self.assertIsInstance(f, dict)
        self.assertIn("exposure", f)

    def test_fields_is_a_copy(self):
        s = CameraSettings()
        f = s.fields
        f["fake"] = None # type: ignore
        self.assertNotIn("fake", s.fields)


class TestPresetExtras(unittest.TestCase):
    """Tests for preset edge cases (missing/corrupt files)."""

    def _make_root(self):
        class Root(Settings):
            cam: CameraSettings
        return Root()

    def test_load_missing_file_no_error(self):
        root = self._make_root()
        presets.load(root, "nonexistent_path_12345.json")  # should not raise

    def test_load_corrupt_file_no_error(self):
        root = self._make_root()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("{corrupt json!!!")
            path = f.name
        try:
            presets.load(root, path)  # should not raise
        finally:
            os.unlink(path)


class TestAction(unittest.TestCase):
    """Tests for Widget.button Settings (replacing the old Action descriptor)."""

    def test_action_fires_callbacks(self):
        s = SettingsWithActions()
        results = []
        s.bind(SettingsWithActions.reset, lambda _: results.append("fired"))
        SettingsWithActions.reset.fire(s)
        self.assertEqual(results, ["fired"])

    def test_action_multiple_callbacks(self):
        s = SettingsWithActions()
        a, b = [], []
        s.bind(SettingsWithActions.reset, lambda _: a.append(1))
        s.bind(SettingsWithActions.reset, lambda _: b.append(2))
        SettingsWithActions.reset.fire(s)
        self.assertEqual(a, [1])
        self.assertEqual(b, [2])

    def test_action_no_callbacks_ok(self):
        s = SettingsWithActions()
        SettingsWithActions.reset.fire(s)  # should not raise

    def test_action_is_assignable(self):
        """Widget.button fields are normal Settings — assignment is allowed."""
        s = SettingsWithActions()
        s.reset = True  # should not raise
        self.assertEqual(s.reset, True)

    def test_bind_action_unknown_raises(self):
        s = SettingsWithActions()
        with self.assertRaises(KeyError):
            s.bind(MinimalSettings.value, lambda _: None)

    def test_unbind_action(self):
        s = SettingsWithActions()
        results = []
        cb = lambda _: results.append(1)
        s.bind(SettingsWithActions.reset, cb)
        s.unbind(SettingsWithActions.reset, cb)
        SettingsWithActions.reset.fire(s)
        self.assertEqual(results, [])

    def test_actions_property(self):
        s = SettingsWithActions()
        a = s.actions
        self.assertIsInstance(a, dict)
        self.assertIn("reset", a)
        self.assertIn("hidden_action", a)

    def test_actions_property_is_copy(self):
        s = SettingsWithActions()
        a = s.actions
        a["fake"] = None # type: ignore
        self.assertNotIn("fake", s.actions)

    def test_action_repr(self):
        r = repr(SettingsWithActions.reset)
        self.assertIn("Reset all values", r)

    def test_action_class_access_returns_descriptor(self):
        self.assertIsInstance(SettingsWithActions.reset, Field)

    def test_action_broken_callback_doesnt_crash(self):
        s = SettingsWithActions()
        results = []
        s.bind(SettingsWithActions.reset, lambda _: 1 / 0)  # will raise
        s.bind(SettingsWithActions.reset, lambda _: results.append("ok"))
        SettingsWithActions.reset.fire(s)  # should not raise
        self.assertEqual(results, ["ok"])


# ===== Child descriptor tests =============================================

class InnerSettings(Settings):
    speed = Field(1.0, min=0.0, max=10.0)
    scale = Field(0.5)


class OuterSettings(Settings):
    fps = Field(60.0, min=1.0, max=240.0)
    inner = Group(InnerSettings)


class DoubleChildSettings(Settings):
    name = Field("default")
    alpha = Group(InnerSettings)
    beta = Group(InnerSettings)


class MiddleSettings(Settings):
    weight = Field(0.5, min=0.0, max=1.0)
    inner = Group(InnerSettings)


class DeepSettings(Settings):
    label = Field("root")
    middle = Group(MiddleSettings)


class TestChild(unittest.TestCase):
    """Tests for the Child descriptor."""

    # -- Basic access -------------------------------------------------------

    def test_child_access(self):
        cfg = OuterSettings()
        self.assertIsInstance(cfg.inner, InnerSettings)
        self.assertEqual(cfg.inner.speed, 1.0)
        self.assertEqual(cfg.inner.scale, 0.5)

    def test_child_field_mutation(self):
        cfg = OuterSettings()
        cfg.inner.speed = 5.0
        self.assertEqual(cfg.inner.speed, 5.0)

    def test_child_not_replaceable(self):
        cfg = OuterSettings()
        with self.assertRaises(AttributeError):
            cfg.inner = InnerSettings()

    def test_each_instance_gets_own_child(self):
        a = OuterSettings()
        b = OuterSettings()
        a.inner.speed = 9.0
        self.assertEqual(b.inner.speed, 1.0)

    # -- Children property --------------------------------------------------

    def test_children_property(self):
        cfg = OuterSettings()
        children = cfg.children
        self.assertIn("inner", children)
        self.assertIsInstance(children["inner"], InnerSettings)

    def test_children_property_returns_copy(self):
        cfg = OuterSettings()
        children = cfg.children
        children["fake"] = None
        self.assertNotIn("fake", cfg.children)

    # -- Multiple children --------------------------------------------------

    def test_multiple_children_independent(self):
        cfg = DoubleChildSettings()
        cfg.alpha.speed = 3.0
        cfg.beta.speed = 7.0
        self.assertEqual(cfg.alpha.speed, 3.0)
        self.assertEqual(cfg.beta.speed, 7.0)

    # -- Serialization: to_dict ---------------------------------------------

    def test_to_dict_includes_children(self):
        cfg = OuterSettings()
        cfg.inner.speed = 2.5
        d = cfg.to_dict()
        self.assertIn("fps", d)
        self.assertIn("inner", d)
        self.assertEqual(d["inner"]["speed"], 2.5)
        self.assertEqual(d["inner"]["scale"], 0.5)

    def test_to_dict_nested_structure(self):
        cfg = DoubleChildSettings()
        cfg.alpha.speed = 1.0
        cfg.beta.speed = 2.0
        d = cfg.to_dict()
        self.assertEqual(d["alpha"]["speed"], 1.0)
        self.assertEqual(d["beta"]["speed"], 2.0)
        self.assertEqual(d["name"], "default")

    # -- Serialization: update_from_dict ------------------------------------

    def test_update_from_dict_restores_children(self):
        cfg = OuterSettings()
        cfg.update_from_dict({"fps": 120.0, "inner": {"speed": 8.0, "scale": 0.25}})
        self.assertEqual(cfg.fps, 120.0)
        self.assertEqual(cfg.inner.speed, 8.0)
        self.assertEqual(cfg.inner.scale, 0.25)

    def test_update_from_dict_partial_child(self):
        cfg = OuterSettings()
        cfg.update_from_dict({"inner": {"speed": 4.0}})
        self.assertEqual(cfg.inner.speed, 4.0)
        self.assertEqual(cfg.inner.scale, 0.5)  # Unchanged

    def test_update_from_dict_ignores_unknown_child_keys(self):
        cfg = OuterSettings()
        cfg.update_from_dict({"inner": {"speed": 3.0, "nonexistent": 99}})
        self.assertEqual(cfg.inner.speed, 3.0)

    # -- Round-trip ---------------------------------------------------------

    def test_json_round_trip(self):
        cfg1 = OuterSettings()
        cfg1.fps = 144.0
        cfg1.inner.speed = 3.5
        cfg1.inner.scale = 0.75

        serialized = json.dumps(cfg1.to_dict())
        data = json.loads(serialized)

        cfg2 = OuterSettings()
        cfg2.update_from_dict(data)
        self.assertEqual(cfg2.fps, 144.0)
        self.assertEqual(cfg2.inner.speed, 3.5)
        self.assertEqual(cfg2.inner.scale, 0.75)

    # -- Callbacks on child fields ------------------------------------------

    def test_child_field_callbacks_still_fire(self):
        cfg = OuterSettings()
        received = []
        cfg.inner.bind(InnerSettings.speed, lambda v: received.append(v))
        cfg.inner.speed = 7.0
        self.assertEqual(received, [7.0])

    # -- Repr ---------------------------------------------------------------

    def test_repr_shows_children(self):
        cfg = OuterSettings()
        r = repr(cfg)
        self.assertIn("InnerSettings(...)", r)
        self.assertIn("fps=", r)

    # -- Class-level attribute access (Group descriptor on the class) --------

    def test_class_level_group_descriptor_exists(self):
        desc = OuterSettings.__dict__.get("inner")
        self.assertIsInstance(desc, Group)

    # -- Preset save/load with children -------------------------------------

    def test_save_load_with_children(self):
        class Root(Settings):
            outer = Group(OuterSettings)
        root = Root()
        root.outer.fps = 30.0
        root.outer.inner.speed = 2.0

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            path = f.name
        try:
            presets.save(root, path)

            root2 = Root()
            presets.load(root2, path)

            self.assertEqual(root2.outer.fps, 30.0)
            self.assertEqual(root2.outer.inner.speed, 2.0)
        finally:
            os.unlink(path)

    # -- Deep nesting (3 levels) --------------------------------------------

    def test_deep_access(self):
        cfg = DeepSettings()
        self.assertEqual(cfg.label, "root")
        self.assertIsInstance(cfg.middle, MiddleSettings)
        self.assertEqual(cfg.middle.weight, 0.5)
        self.assertIsInstance(cfg.middle.inner, InnerSettings)
        self.assertEqual(cfg.middle.inner.speed, 1.0)
        self.assertEqual(cfg.middle.inner.scale, 0.5)

    def test_deep_mutation(self):
        cfg = DeepSettings()
        cfg.middle.inner.speed = 9.9
        self.assertEqual(cfg.middle.inner.speed, 9.9)
        cfg.middle.weight = 0.1
        self.assertEqual(cfg.middle.weight, 0.1)

    def test_deep_to_dict(self):
        cfg = DeepSettings()
        cfg.middle.inner.speed = 3.0
        cfg.middle.weight = 0.8
        d = cfg.to_dict()
        self.assertEqual(d["label"], "root")
        self.assertIn("middle", d)
        self.assertEqual(d["middle"]["weight"], 0.8)
        self.assertIn("inner", d["middle"])
        self.assertEqual(d["middle"]["inner"]["speed"], 3.0)
        self.assertEqual(d["middle"]["inner"]["scale"], 0.5)

    def test_deep_update_from_dict(self):
        cfg = DeepSettings()
        cfg.update_from_dict({
            "label": "updated",
            "middle": {
                "weight": 0.2,
                "inner": {"speed": 7.0, "scale": 0.1}
            }
        })
        self.assertEqual(cfg.label, "updated")
        self.assertEqual(cfg.middle.weight, 0.2)
        self.assertEqual(cfg.middle.inner.speed, 7.0)
        self.assertEqual(cfg.middle.inner.scale, 0.1)

    def test_deep_update_from_dict_partial(self):
        cfg = DeepSettings()
        cfg.update_from_dict({"middle": {"inner": {"speed": 5.5}}})
        self.assertEqual(cfg.middle.inner.speed, 5.5)
        self.assertEqual(cfg.middle.inner.scale, 0.5)  # unchanged
        self.assertEqual(cfg.middle.weight, 0.5)        # unchanged
        self.assertEqual(cfg.label, "root")              # unchanged

    def test_deep_json_round_trip(self):
        cfg1 = DeepSettings()
        cfg1.label = "trip"
        cfg1.middle.weight = 0.3
        cfg1.middle.inner.speed = 4.4
        cfg1.middle.inner.scale = 0.9

        serialized = json.dumps(cfg1.to_dict())
        data = json.loads(serialized)

        cfg2 = DeepSettings()
        cfg2.update_from_dict(data)
        self.assertEqual(cfg2.label, "trip")
        self.assertEqual(cfg2.middle.weight, 0.3)
        self.assertEqual(cfg2.middle.inner.speed, 4.4)
        self.assertEqual(cfg2.middle.inner.scale, 0.9)

    def test_deep_callback(self):
        cfg = DeepSettings()
        received = []
        cfg.middle.inner.bind(InnerSettings.speed, lambda v: received.append(v))
        cfg.middle.inner.speed = 6.6
        self.assertEqual(received, [6.6])

    def test_deep_instances_independent(self):
        a = DeepSettings()
        b = DeepSettings()
        a.middle.inner.speed = 100.0
        self.assertEqual(b.middle.inner.speed, 1.0)

    def test_deep_children_property(self):
        cfg = DeepSettings()
        self.assertIn("middle", cfg.children)
        self.assertIn("inner", cfg.middle.children)


# ── List Setting ───────────────────────────────────────────────────────────

class ListSettings(Settings):
    tags = Field(["default"])
    ids = Field([1, 2, 3])


class TestListSetting(unittest.TestCase):
    """Tests for list[T] support in Field."""

    def test_default_value(self):
        s = ListSettings()
        self.assertEqual(s.tags, ["default"])
        self.assertEqual(s.ids, [1, 2, 3])

    def test_defaults_are_independent(self):
        a = ListSettings()
        b = ListSettings()
        a.tags.append("x")
        # b should be unaffected — Setting stores a separate list per instance
        # (coerce runs list(value) which copies)
        # Actually, defaults share the same list object until first set.
        # We verify via set:
        a.tags = ["a"]
        self.assertEqual(b.tags, ["default"])

    def test_set_list(self):
        s = ListSettings()
        s.ids = [10, 20]
        self.assertEqual(s.ids, [10, 20])

    def test_coercion_int_elements(self):
        s = ListSettings()
        s.ids = [1.0, 2.0, 3.0]  # type: ignore
        self.assertEqual(s.ids, [1, 2, 3])
        self.assertIsInstance(s.ids[0], int)

    def test_coercion_str_elements(self):
        s = ListSettings()
        s.tags = ["hello", "world"]
        self.assertEqual(s.tags, ["hello", "world"])

    def test_reject_non_list(self):
        s = ListSettings()
        with self.assertRaises(TypeError):
            s.ids = 42  # type: ignore

    def test_callback_fires(self):
        s = ListSettings()
        received = []
        s.bind(ListSettings.ids, lambda v: received.append(v))
        s.ids = [7, 8, 9]
        self.assertEqual(received, [[7, 8, 9]])

    def test_callback_skip_equal(self):
        s = ListSettings()
        s.ids = [1, 2, 3]  # same as default
        received = []
        s.bind(ListSettings.ids, lambda v: received.append(v))
        s.ids = [1, 2, 3]  # equal — should NOT fire
        self.assertEqual(received, [])

    def test_to_dict(self):
        s = ListSettings()
        s.ids = [4, 5]
        d = s.to_dict()
        self.assertEqual(d["ids"], [4, 5])
        # Verify it's a copy, not the same list
        d["ids"].append(99)
        self.assertEqual(s.ids, [4, 5])

    def test_update_from_dict(self):
        s = ListSettings()
        s.update_from_dict({"ids": [10, 20, 30], "tags": ["a", "b"]})
        self.assertEqual(s.ids, [10, 20, 30])
        self.assertEqual(s.tags, ["a", "b"])

    def test_json_round_trip(self):
        import json
        s1 = ListSettings()
        s1.ids = [100, 200]
        s1.tags = ["foo"]
        data = json.loads(json.dumps(s1.to_dict()))
        s2 = ListSettings()
        s2.update_from_dict(data)
        self.assertEqual(s2.ids, [100, 200])
        self.assertEqual(s2.tags, ["foo"])

    def test_repr(self):
        desc = repr(ListSettings.__dict__["ids"])
        self.assertIn("list[int]", desc)

    def test_empty_list_assignment(self):
        s = ListSettings()
        s.ids = []
        self.assertEqual(s.ids, [])

    def test_empty_list_default_rejected(self):
        with self.assertRaises(ValueError):
            Field([])

    def test_reject_bool_in_int_list(self):
        """M2: bool should not pass as int in list elements."""
        s = ListSettings()
        with self.assertRaises(TypeError):
            s.ids = [True, 2, 3]

    def test_reject_bool_in_float_list(self):
        class FloatListSettings(Settings):
            vals = Field([1.0, 2.0])
        s = FloatListSettings()
        with self.assertRaises(TypeError):
            s.vals = [False, 2.0]


# ---------------------------------------------------------------------------
# Color utility class tests
# ---------------------------------------------------------------------------

class TestColorClass(unittest.TestCase):
    """Tests for the Color utility class from modules.utils.Color."""

    # -- Construction --------------------------------------------------------

    def test_rgba_defaults(self):
        c = Color(0.5, 0.2, 0.1)
        self.assertAlmostEqual(c.r, 0.5)
        self.assertAlmostEqual(c.g, 0.2)
        self.assertAlmostEqual(c.b, 0.1)
        self.assertAlmostEqual(c.a, 1.0)

    def test_rgba_explicit_alpha(self):
        c = Color(0.5, 0.2, 0.1, 0.8)
        self.assertAlmostEqual(c.a, 0.8)

    # -- from_int / to_int ---------------------------------------------------

    def test_from_int(self):
        c = Color.from_int(128, 64, 32)
        self.assertAlmostEqual(c.r, 128 / 255, places=3)
        self.assertAlmostEqual(c.g, 64 / 255, places=3)
        self.assertAlmostEqual(c.b, 32 / 255, places=3)
        self.assertAlmostEqual(c.a, 1.0)

    def test_from_int_with_alpha(self):
        c = Color.from_int(255, 0, 0, 128)
        self.assertAlmostEqual(c.a, 128 / 255, places=3)

    def test_to_int_round_trip(self):
        c = Color.from_int(200, 100, 50, 255)
        ri, gi, bi, ai = c.to_int()
        self.assertEqual((ri, gi, bi, ai), (200, 100, 50, 255))

    # -- from_hex / to_hex ---------------------------------------------------

    def test_from_hex_6(self):
        c = Color.from_hex("#FF8040")
        self.assertAlmostEqual(c.r, 1.0, places=2)
        self.assertAlmostEqual(c.g, 128 / 255, places=2)
        self.assertAlmostEqual(c.b, 64 / 255, places=2)
        self.assertAlmostEqual(c.a, 1.0)

    def test_from_hex_8(self):
        c = Color.from_hex("#FF804080")
        self.assertAlmostEqual(c.a, 128 / 255, places=2)

    def test_from_hex_3(self):
        c = Color.from_hex("#F84")
        self.assertAlmostEqual(c.r, 0xFF / 255, places=2)
        self.assertAlmostEqual(c.g, 0x88 / 255, places=2)
        self.assertAlmostEqual(c.b, 0x44 / 255, places=2)

    def test_from_hex_4(self):
        c = Color.from_hex("#F84A")
        self.assertAlmostEqual(c.r, 0xFF / 255, places=2)
        self.assertAlmostEqual(c.a, 0xAA / 255, places=2)

    def test_from_hex_no_hash(self):
        c = Color.from_hex("FF0000")
        self.assertAlmostEqual(c.r, 1.0)

    def test_to_hex(self):
        c = Color(1.0, 0.0, 0.0)
        self.assertEqual(c.to_hex(), "#FF0000")

    def test_to_hex_with_alpha(self):
        c = Color(1.0, 0.0, 0.0, 0.5)
        h = c.to_hex(include_alpha=True)
        self.assertTrue(h.startswith("#FF0000"))
        self.assertEqual(len(h), 9)

    def test_hex_round_trip(self):
        original = Color(0.8, 0.4, 0.2)
        rebuilt = Color.from_hex(original.to_hex())
        # Hex is 8-bit so we lose some precision
        for ch in ("r", "g", "b"):
            self.assertAlmostEqual(getattr(original, ch), getattr(rebuilt, ch), places=2)

    def test_from_hex_invalid(self):
        with self.assertRaises(ValueError):
            Color.from_hex("#ZZZZZZ")

    def test_from_hex_bad_length(self):
        with self.assertRaises(ValueError):
            Color.from_hex("#12345")

    # -- from_hsv / to_hsv ---------------------------------------------------

    def test_hsv_round_trip(self):
        c = Color(0.8, 0.3, 0.1)
        h, s, v, a = c.to_hsv()
        rebuilt = Color.from_hsv(h, s, v, a)
        self.assertAlmostEqual(c.r, rebuilt.r, places=5)
        self.assertAlmostEqual(c.g, rebuilt.g, places=5)
        self.assertAlmostEqual(c.b, rebuilt.b, places=5)

    def test_from_hsv_red(self):
        c = Color.from_hsv(0.0, 1.0, 1.0)
        self.assertAlmostEqual(c.r, 1.0)
        self.assertAlmostEqual(c.g, 0.0)
        self.assertAlmostEqual(c.b, 0.0)

    # -- from_hsl / to_hsl ---------------------------------------------------

    def test_hsl_round_trip(self):
        c = Color(0.6, 0.2, 0.9)
        h, s, l, a = c.to_hsl()
        rebuilt = Color.from_hsl(h, s, l, a)
        self.assertAlmostEqual(c.r, rebuilt.r, places=5)
        self.assertAlmostEqual(c.g, rebuilt.g, places=5)
        self.assertAlmostEqual(c.b, rebuilt.b, places=5)

    # -- Serialization -------------------------------------------------------

    def test_to_dict(self):
        c = Color(0.1, 0.2, 0.3, 0.4)
        self.assertEqual(c.to_dict(), {"r": 0.1, "g": 0.2, "b": 0.3, "a": 0.4})

    def test_from_dict(self):
        c = Color.from_dict({"r": 0.5, "g": 0.6, "b": 0.7, "a": 0.8})
        self.assertEqual(c, Color(0.5, 0.6, 0.7, 0.8))

    def test_from_dict_no_alpha(self):
        c = Color.from_dict({"r": 0.1, "g": 0.2, "b": 0.3})
        self.assertEqual(c, Color(0.1, 0.2, 0.3))
        self.assertAlmostEqual(c.a, 1.0)

    def test_to_tuple(self):
        c = Color(0.1, 0.2, 0.3, 0.4)
        self.assertEqual(c.to_tuple(), (0.1, 0.2, 0.3, 0.4))

    def test_from_tuple_3(self):
        c = Color.from_tuple((0.1, 0.2, 0.3))
        self.assertEqual(c, Color(0.1, 0.2, 0.3))

    def test_from_tuple_4(self):
        c = Color.from_tuple((0.1, 0.2, 0.3, 0.4))
        self.assertEqual(c, Color(0.1, 0.2, 0.3, 0.4))

    # -- Protocol methods ----------------------------------------------------

    def test_iter(self):
        c = Color(0.1, 0.2, 0.3, 0.4)
        self.assertEqual(list(c), [0.1, 0.2, 0.3, 0.4])

    def test_getitem(self):
        c = Color(0.1, 0.2, 0.3, 0.4)
        self.assertAlmostEqual(c[0], 0.1)
        self.assertAlmostEqual(c[1], 0.2)
        self.assertAlmostEqual(c[2], 0.3)
        self.assertAlmostEqual(c[3], 0.4)

    def test_getitem_out_of_range(self):
        c = Color(0.1, 0.2, 0.3)
        with self.assertRaises(IndexError):
            _ = c[4]

    def test_len(self):
        self.assertEqual(len(Color(0, 0, 0)), 4)

    def test_equality(self):
        self.assertEqual(Color(0.1, 0.2, 0.3), Color(0.1, 0.2, 0.3))
        self.assertNotEqual(Color(0.1, 0.2, 0.3), Color(0.1, 0.2, 0.4))
        self.assertNotEqual(Color(0.1, 0.2, 0.3, 0.5), Color(0.1, 0.2, 0.3, 0.6))

    def test_eq_not_implemented_for_other_types(self):
        c = Color(0, 0, 0)
        self.assertIs(c.__eq__("not a color"), NotImplemented)

    def test_repr_no_alpha(self):
        self.assertEqual(repr(Color(0.1, 0.2, 0.3)), "Color(0.1, 0.2, 0.3)")

    def test_repr_with_alpha(self):
        self.assertIn("0.5", repr(Color(0.1, 0.2, 0.3, 0.5)))

    def test_copy(self):
        c = Color(0.1, 0.2, 0.3, 0.4)
        c2 = c.copy()
        self.assertEqual(c, c2)
        self.assertIsNot(c, c2)

    # -- Operations ----------------------------------------------------------

    def test_lerp(self):
        a = Color(0.0, 0.0, 0.0, 0.0)
        b = Color(1.0, 1.0, 1.0, 1.0)
        mid = a.lerp(b, 0.5)
        for ch in (mid.r, mid.g, mid.b, mid.a):
            self.assertAlmostEqual(ch, 0.5)

    def test_clamped(self):
        c = Color(-0.1, 1.5, 0.5, 2.0)
        cl = c.clamped()
        self.assertAlmostEqual(cl.r, 0.0)
        self.assertAlmostEqual(cl.g, 1.0)
        self.assertAlmostEqual(cl.b, 0.5)
        self.assertAlmostEqual(cl.a, 1.0)

    def test_with_alpha(self):
        c = Color(0.1, 0.2, 0.3)
        c2 = c.with_alpha(0.5)
        self.assertAlmostEqual(c2.a, 0.5)
        self.assertAlmostEqual(c2.r, 0.1)

    def test_luminance(self):
        white = Color(1.0, 1.0, 1.0)
        self.assertAlmostEqual(white.luminance, 1.0, places=3)
        black = Color(0.0, 0.0, 0.0)
        self.assertAlmostEqual(black.luminance, 0.0)

    def test_rgb_property(self):
        c = Color(0.1, 0.2, 0.3, 0.4)
        self.assertEqual(c.rgb, (0.1, 0.2, 0.3))


# ---------------------------------------------------------------------------
# Point2f / Rect serialization tests
# ---------------------------------------------------------------------------

class TestPoint2fSerialization(unittest.TestCase):
    """Tests for Point2f to_dict / from_dict and Setting integration."""

    def test_to_dict(self):
        p = Point2f(1.5, 2.5)
        self.assertEqual(p.to_dict(), {"x": 1.5, "y": 2.5})

    def test_from_dict(self):
        p = Point2f.from_dict({"x": 3.0, "y": 4.0})
        self.assertEqual(p, Point2f(3.0, 4.0))

    def test_round_trip(self):
        p = Point2f(1.23, 4.56)
        self.assertEqual(Point2f.from_dict(p.to_dict()), p)


class TestRectSerialization(unittest.TestCase):
    """Tests for Rect to_dict / from_dict and Setting integration."""

    def test_to_dict(self):
        r = Rect(1.0, 2.0, 3.0, 4.0)
        self.assertEqual(r.to_dict(), {"x": 1.0, "y": 2.0, "width": 3.0, "height": 4.0})

    def test_from_dict(self):
        r = Rect.from_dict({"x": 10.0, "y": 20.0, "width": 30.0, "height": 40.0})
        self.assertEqual(r, Rect(10.0, 20.0, 30.0, 40.0))

    def test_round_trip(self):
        r = Rect(1.5, 2.5, 100.0, 200.0)
        self.assertEqual(Rect.from_dict(r.to_dict()), r)


# ---------------------------------------------------------------------------
# Setting integration with Color, Point2f, Rect
# ---------------------------------------------------------------------------

class _ColorSettings(Settings):
    color = Field(Color(1.0, 0.0, 0.0))


class _PointSettings(Settings):
    pos = Field(Point2f(0.0, 0.0))


class _RectSettings(Settings):
    region = Field(Rect(0.0, 0.0, 1.0, 1.0))


class TestColorSettingIntegration(unittest.TestCase):
    """Tests for Color as a Setting type."""

    def test_default(self):
        s = _ColorSettings()
        self.assertEqual(s.color, Color(1.0, 0.0, 0.0))

    def test_direct_set(self):
        s = _ColorSettings()
        s.color = Color(0.0, 1.0, 0.0, 0.5)
        self.assertEqual(s.color, Color(0.0, 1.0, 0.0, 0.5))

    def test_tuple_coercion_3(self):
        s = _ColorSettings()
        s.color = (0.5, 0.5, 0.5)  # type: ignore
        self.assertEqual(s.color, Color(0.5, 0.5, 0.5))

    def test_tuple_coercion_4(self):
        s = _ColorSettings()
        s.color = (0.1, 0.2, 0.3, 0.4)  # type: ignore
        self.assertEqual(s.color, Color(0.1, 0.2, 0.3, 0.4))

    def test_list_coercion(self):
        s = _ColorSettings()
        s.color = [0.2, 0.3, 0.4]  # type: ignore
        self.assertEqual(s.color, Color(0.2, 0.3, 0.4))

    def test_dict_coercion(self):
        s = _ColorSettings()
        s.color = {"r": 0.1, "g": 0.2, "b": 0.3, "a": 0.9}  # type: ignore
        self.assertEqual(s.color, Color(0.1, 0.2, 0.3, 0.9))

    def test_dict_coercion_no_alpha(self):
        s = _ColorSettings()
        s.color = {"r": 0.5, "g": 0.6, "b": 0.7}  # type: ignore
        self.assertEqual(s.color, Color(0.5, 0.6, 0.7))

    def test_to_dict(self):
        s = _ColorSettings()
        d = s.to_dict()
        self.assertEqual(d["color"], {"r": 1.0, "g": 0.0, "b": 0.0, "a": 1.0})

    def test_json_round_trip(self):
        s1 = _ColorSettings()
        s1.color = Color(0.1, 0.2, 0.3, 0.4)
        data = json.loads(json.dumps(s1.to_dict()))
        s2 = _ColorSettings()
        s2.update_from_dict(data)
        self.assertEqual(s2.color, Color(0.1, 0.2, 0.3, 0.4))

    def test_callback(self):
        s = _ColorSettings()
        results = []
        s.bind(_ColorSettings.color, lambda v: results.append(v))
        s.color = Color(0, 0, 1)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0], Color(0, 0, 1))


class TestPoint2fSettingIntegration(unittest.TestCase):
    """Tests for Point2f as a Setting type."""

    def test_default(self):
        s = _PointSettings()
        self.assertEqual(s.pos, Point2f(0.0, 0.0))

    def test_direct_set(self):
        s = _PointSettings()
        s.pos = Point2f(5.0, 10.0)
        self.assertEqual(s.pos, Point2f(5.0, 10.0))

    def test_tuple_coercion(self):
        s = _PointSettings()
        s.pos = (3.0, 4.0)  # type: ignore
        self.assertEqual(s.pos, Point2f(3.0, 4.0))

    def test_list_coercion(self):
        s = _PointSettings()
        s.pos = [1.0, 2.0]  # type: ignore
        self.assertEqual(s.pos, Point2f(1.0, 2.0))

    def test_dict_coercion(self):
        s = _PointSettings()
        s.pos = {"x": 7.0, "y": 8.0}  # type: ignore
        self.assertEqual(s.pos, Point2f(7.0, 8.0))

    def test_to_dict(self):
        s = _PointSettings()
        s.pos = Point2f(1.5, 2.5)
        self.assertEqual(s.to_dict()["pos"], {"x": 1.5, "y": 2.5})

    def test_json_round_trip(self):
        s1 = _PointSettings()
        s1.pos = Point2f(99.0, 88.0)
        data = json.loads(json.dumps(s1.to_dict()))
        s2 = _PointSettings()
        s2.update_from_dict(data)
        self.assertEqual(s2.pos, Point2f(99.0, 88.0))


class TestRectSettingIntegration(unittest.TestCase):
    """Tests for Rect as a Setting type."""

    def test_default(self):
        s = _RectSettings()
        self.assertEqual(s.region, Rect(0.0, 0.0, 1.0, 1.0))

    def test_direct_set(self):
        s = _RectSettings()
        s.region = Rect(10, 20, 100, 200)
        self.assertEqual(s.region, Rect(10, 20, 100, 200))

    def test_tuple_coercion(self):
        s = _RectSettings()
        s.region = (5, 10, 50, 100)  # type: ignore
        self.assertEqual(s.region, Rect(5, 10, 50, 100))

    def test_list_coercion(self):
        s = _RectSettings()
        s.region = [1, 2, 3, 4]  # type: ignore
        self.assertEqual(s.region, Rect(1, 2, 3, 4))

    def test_dict_coercion(self):
        s = _RectSettings()
        s.region = {"x": 0, "y": 0, "width": 640, "height": 480}  # type: ignore
        self.assertEqual(s.region, Rect(0, 0, 640, 480))

    def test_to_dict(self):
        s = _RectSettings()
        self.assertEqual(
            s.to_dict()["region"],
            {"x": 0.0, "y": 0.0, "width": 1.0, "height": 1.0}
        )

    def test_json_round_trip(self):
        s1 = _RectSettings()
        s1.region = Rect(10, 20, 320, 240)
        data = json.loads(json.dumps(s1.to_dict()))
        s2 = _RectSettings()
        s2.update_from_dict(data)
        self.assertEqual(s2.region, Rect(10, 20, 320, 240))


# ── Hardening tests ────────────────────────────────────────────────────────

class TestBindAllWithButtons(unittest.TestCase):
    """bind_all must work when Widget.button fields are present (#1)."""

    def test_bind_all_fires_for_buttons(self):
        s = SettingsWithActions()
        results = []
        s.bind_all(lambda v: results.append(v))
        SettingsWithActions.reset.fire(s)
        self.assertIn(True, results)

    def test_bind_all_fires_for_normal_fields(self):
        s = SettingsWithActions()
        results = []
        s.bind_all(lambda v: results.append(v))
        s.exposure = 5000
        self.assertIn(5000, results)


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


class TestSettingsEquality(unittest.TestCase):
    """__eq__ on Settings (#9)."""

    def test_equal_defaults(self):
        self.assertEqual(CameraSettings(), CameraSettings())

    def test_different_values(self):
        a = CameraSettings()
        b = CameraSettings()
        a.exposure = 9999
        self.assertNotEqual(a, b)

    def test_different_types(self):
        self.assertNotEqual(CameraSettings(), MinimalSettings())

    def test_unhashable(self):
        with self.assertRaises(TypeError):
            hash(CameraSettings())


class TestDeepCopyDefaults(unittest.TestCase):
    """List defaults must be deep-copied between instances (#10)."""

    def test_list_not_shared(self):
        class TagSettings(Settings):
            tags = Field(["a", "b"])

        s1 = TagSettings()
        s2 = TagSettings()
        s1.tags = s1.tags + ["c"]
        self.assertEqual(s2.tags, ["a", "b"])

    def test_nested_list_not_shared(self):
        class NestedList(Settings):
            items = Field(["x"])

        a = NestedList()
        b = NestedList()
        # Directly mutate the internal default (simulates shallow-copy bug)
        self.assertIsNot(a._values["items"], b._values["items"])


# ── Widget class tests ─────────────────────────────────────────────────────

class TestWidgetClass(unittest.TestCase):
    """Tests for the Widget Enum."""

    def test_eq_same(self):
        self.assertEqual(Widget.button, Widget.button)

    def test_eq_different(self):
        self.assertNotEqual(Widget.button, Widget.toggle)

    def test_eq_non_widget(self):
        self.assertNotEqual(Widget.button, "button")

    def test_hash_consistent(self):
        self.assertEqual(hash(Widget.button), hash(Widget.button))
        self.assertNotEqual(hash(Widget.button), hash(Widget.toggle))

    def test_hash_usable_as_dict_key(self):
        d = {Widget.button: "b", Widget.toggle: "t"}
        self.assertEqual(d[Widget.button], "b")

    def test_repr(self):
        self.assertEqual(repr(Widget.button), "Widget.button")
        self.assertEqual(repr(Widget.default), "Widget.default")

    def test_name_property(self):
        self.assertEqual(Widget.slider.name, "slider")

    def test_types_property(self):
        self.assertEqual(Widget.switch.types, (bool,))
        self.assertEqual(Widget.slider.types, (int, float))
        self.assertIsNone(Widget.default.types)

    def test_is_enum(self):
        self.assertIsInstance(Widget.button, Widget)
        self.assertIsInstance(Widget.default, Widget)

    def test_identity(self):
        self.assertIs(Widget.button, Widget.button)
        self.assertIsNot(Widget.button, Widget.toggle)

    def test_same_types_still_distinct(self):
        """Members with identical compatible types must remain separate."""
        self.assertIsNot(Widget.switch, Widget.toggle)
        self.assertIsNot(Widget.switch, Widget.button)
        self.assertIsNot(Widget.slider, Widget.number)
        self.assertIsNot(Widget.select, Widget.radio)
        self.assertIsNot(Widget.checklist, Widget.order)
        self.assertIsNot(Widget.default, Widget.color)

    def test_iteration(self):
        members = list(Widget)
        self.assertIn(Widget.button, members)
        self.assertEqual(len(members), 17)

    def test_match_statement(self):
        """Widget members work in match/case."""
        w = Widget.slider
        match w:
            case Widget.slider:
                result = "slider"
            case _:
                result = "other"
        self.assertEqual(result, "slider")


class TestWidgetAccepts(unittest.TestCase):
    """Widget.accepts() validates type compatibility."""

    def test_default_accepts_anything(self):
        self.assertTrue(Widget.default.accepts(bool))
        self.assertTrue(Widget.default.accepts(int))
        self.assertTrue(Widget.default.accepts(str))

    def test_switch_accepts_bool(self):
        self.assertTrue(Widget.switch.accepts(bool))

    def test_switch_rejects_int(self):
        self.assertFalse(Widget.switch.accepts(int))

    def test_slider_accepts_int(self):
        self.assertTrue(Widget.slider.accepts(int))

    def test_slider_accepts_float(self):
        self.assertTrue(Widget.slider.accepts(float))

    def test_slider_rejects_str(self):
        self.assertFalse(Widget.slider.accepts(str))

    def test_select_accepts_enum(self):
        self.assertTrue(Widget.select.accepts(RenderMode))

    def test_select_rejects_str(self):
        self.assertFalse(Widget.select.accepts(str))

    def test_checklist_accepts_list(self):
        import typing
        self.assertTrue(Widget.checklist.accepts(list[RenderMode]))

    def test_checklist_rejects_enum(self):
        self.assertFalse(Widget.checklist.accepts(RenderMode))

    def test_ip_accepts_str(self):
        self.assertTrue(Widget.ip_field.accepts(str))

    def test_ip_rejects_int(self):
        self.assertFalse(Widget.ip_field.accepts(int))

    def test_color_accepts_color(self):
        self.assertTrue(Widget.color.accepts(Color))

    def test_color_rejects_str(self):
        self.assertFalse(Widget.color.accepts(str))

    def test_color_alpha_accepts_color(self):
        self.assertTrue(Widget.color_alpha.accepts(Color))


class TestWidgetResolve(unittest.TestCase):
    """Widget.resolve() maps Widget.default to a concrete widget."""

    def test_bool_resolves_to_switch(self):
        field = Field(True)
        field.__set_name__(None, "test")
        self.assertEqual(Widget.resolve(field), Widget.switch)

    def test_int_with_range_resolves_to_slider(self):
        field = Field(50, min=0, max=100)
        field.__set_name__(None, "test")
        self.assertEqual(Widget.resolve(field), Widget.slider)

    def test_int_without_range_resolves_to_number(self):
        field = Field(50)
        field.__set_name__(None, "test")
        self.assertEqual(Widget.resolve(field), Widget.number)

    def test_float_with_range_resolves_to_slider(self):
        field = Field(0.5, min=0.0, max=1.0)
        field.__set_name__(None, "test")
        self.assertEqual(Widget.resolve(field), Widget.slider)

    def test_enum_resolves_to_select(self):
        field = Field(RenderMode.SOLID)
        field.__set_name__(None, "test")
        self.assertEqual(Widget.resolve(field), Widget.select)

    def test_str_resolves_to_input(self):
        field = Field("hello")
        field.__set_name__(None, "test")
        self.assertEqual(Widget.resolve(field), Widget.input)

    def test_color_resolves_to_color(self):
        field = Field(Color(1, 0, 0))
        field.__set_name__(None, "test")
        self.assertEqual(Widget.resolve(field), Widget.color)

    def test_list_resolves_to_checklist(self):
        field = Field([RenderMode.SOLID])
        field.__set_name__(None, "test")
        self.assertEqual(Widget.resolve(field), Widget.checklist)

    def test_explicit_widget_returned_unchanged(self):
        field = Field(True, widget=Widget.toggle)
        field.__set_name__(None, "test")
        self.assertEqual(Widget.resolve(field), Widget.toggle)


class TestWidgetValidation(unittest.TestCase):
    """Setting.__init__ rejects incompatible widget+type combinations."""

    def test_incompatible_widget_raises(self):
        with self.assertRaises(TypeError):
            Field(1.0, widget=Widget.ip_field)

    def test_incompatible_bool_on_slider(self):
        with self.assertRaises(TypeError):
            Field(True, widget=Widget.slider)

    def test_compatible_passes(self):
        # Should not raise
        Field("0.0.0.0", widget=Widget.ip_field)
        Field(True, widget=Widget.toggle)
        Field(True, widget=Widget.button)
        Field(50, widget=Widget.slider, min=0, max=100)
        Field(RenderMode.SOLID, widget=Widget.radio)
        Field(Color(1, 0, 0), widget=Widget.color_alpha)

    def test_default_never_raises(self):
        # Widget.default is always compatible
        Field(1.0, widget=Widget.default)
        Field("text", widget=Widget.default)
        Field(True, widget=Widget.default)


# ── Enum list support ──────────────────────────────────────────────────────

class EnumListSettings(Settings):
    modes = Field([RenderMode.SOLID])


class TestEnumListRoundTrip(unittest.TestCase):
    """Enum list serialization and coercion (was completely untested)."""

    def test_default_value(self):
        s = EnumListSettings()
        self.assertEqual(s.modes, [RenderMode.SOLID])

    def test_set_enum_list(self):
        s = EnumListSettings()
        s.modes = [RenderMode.WIREFRAME, RenderMode.TEXTURED]
        self.assertEqual(s.modes, [RenderMode.WIREFRAME, RenderMode.TEXTURED])

    def test_to_dict_serializes_names(self):
        s = EnumListSettings()
        s.modes = [RenderMode.WIREFRAME, RenderMode.SOLID]
        d = s.to_dict()
        self.assertEqual(d["modes"], ["WIREFRAME", "SOLID"])

    def test_update_from_dict_restores_enums(self):
        s = EnumListSettings()
        s.update_from_dict({"modes": ["TEXTURED", "WIREFRAME"]})
        self.assertEqual(s.modes, [RenderMode.TEXTURED, RenderMode.WIREFRAME])

    def test_json_round_trip(self):
        s1 = EnumListSettings()
        s1.modes = [RenderMode.TEXTURED]
        data = json.loads(json.dumps(s1.to_dict()))
        s2 = EnumListSettings()
        s2.update_from_dict(data)
        self.assertEqual(s2.modes, [RenderMode.TEXTURED])

    def test_bad_enum_name_in_list_raises(self):
        s = EnumListSettings()
        with self.assertRaises(TypeError):
            s.modes = ["NONEXISTENT"] # type: ignore


# ── Enum coercion edge cases ──────────────────────────────────────────────

class TestEnumCoercion(unittest.TestCase):
    """Enum construction from name vs value, plus error paths."""

    def test_enum_from_name(self):
        s = CameraSettings()
        s.mode = "WIREFRAME"  # type: ignore
        self.assertEqual(s.mode, RenderMode.WIREFRAME)

    def test_enum_from_value(self):
        """For non-string-valued enums, coercion by value works."""
        class Priority(Enum):
            LOW = 1
            HIGH = 2
        class PrioSettings(Settings):
            level = Field(Priority.LOW)
        s = PrioSettings()
        s.level = 2  # type: ignore
        self.assertEqual(s.level, Priority.HIGH)

    def test_bad_enum_name_raises(self):
        s = CameraSettings()
        with self.assertRaises(TypeError):
            s.mode = "NONEXISTENT" # type: ignore

    def test_bad_enum_value_raises(self):
        s = CameraSettings()
        with self.assertRaises(TypeError):
            s.mode = 999  # type: ignore


# ── Callback edge cases ───────────────────────────────────────────────────

class TestCallbackEdgeCases(unittest.TestCase):
    """Bind idempotency, unbind no-op, broken callback in _apply."""

    def test_bind_duplicate_prevention(self):
        s = CameraSettings()
        calls = []
        cb = lambda v: calls.append(v)
        s.bind(CameraSettings.exposure, cb)
        s.bind(CameraSettings.exposure, cb)  # duplicate — should be ignored
        s.exposure = 5000
        self.assertEqual(calls, [5000])  # exactly one call

    def test_unbind_silent_noop(self):
        s = CameraSettings()
        cb = lambda v: None
        # Never bound — should not raise
        s.unbind(CameraSettings.exposure, cb)

    def test_broken_callback_doesnt_crash_apply(self):
        """A callback that raises during field set should not crash."""
        s = CameraSettings()
        results = []
        s.bind(CameraSettings.exposure, lambda v: (_ for _ in ()).throw(RuntimeError("boom")))
        s.bind(CameraSettings.exposure, lambda v: results.append(v))
        s.exposure = 9999  # should not raise
        # The second callback should still have fired
        self.assertEqual(results, [9999])

    def test_unbind_all(self):
        s = CameraSettings()
        calls = []
        cb = lambda v: calls.append(v)
        s.bind_all(cb)
        s.exposure = 2000
        self.assertTrue(len(calls) > 0)
        calls.clear()
        s.unbind_all(cb)
        s.exposure = 3000
        self.assertEqual(calls, [])


# ── update_from_dict edge cases ───────────────────────────────────────────

class TestUpdateFromDictEdgeCases(unittest.TestCase):
    """Non-dict child value, equality with children."""

    def test_non_dict_child_logs_warning(self):
        s = OuterSettings()
        with self.assertLogs("modules.settings.settings", level="WARNING") as cm:
            s.update_from_dict({"inner": 42})
        self.assertTrue(any("expected dict for child 'inner'" in m for m in cm.output))
        # inner should be unchanged
        self.assertEqual(s.inner.speed, 1.0)


# ── Equality with children ────────────────────────────────────────────────

class TestEqualityWithChildren(unittest.TestCase):
    """__eq__ should compare child state too."""

    def test_equal_children(self):
        a = OuterSettings()
        b = OuterSettings()
        self.assertEqual(a, b)

    def test_different_child_values(self):
        a = OuterSettings()
        b = OuterSettings()
        a.inner.speed = 9.9
        self.assertNotEqual(a, b)


# ── @property setter routing ──────────────────────────────────────────────

class SettingsWithProperty(Settings):
    raw_val = Field(0)

    @property
    def doubled(self):
        return self.raw_val * 2

    @doubled.setter
    def doubled(self, value):
        self.raw_val = value // 2


class TestPropertySetter(unittest.TestCase):
    """__setattr__ should route to @property setters."""

    def test_property_getter(self):
        s = SettingsWithProperty()
        self.assertEqual(s.doubled, 0)

    def test_property_setter(self):
        s = SettingsWithProperty()
        s.doubled = 20
        self.assertEqual(s.raw_val, 10)
        self.assertEqual(s.doubled, 20)


# ── Field inheritance via MRO ─────────────────────────────────────────────

class ExtendedCamera(CameraSettings):
    zoom = Field(1.0, min=1.0, max=10.0)


class TestFieldInheritance(unittest.TestCase):
    """Fields should be inherited from parent Settings classes."""

    def test_parent_fields_accessible(self):
        s = ExtendedCamera()
        self.assertEqual(s.exposure, 1000)
        self.assertEqual(s.zoom, 1.0)

    def test_parent_fields_in_fields_dict(self):
        s = ExtendedCamera()
        self.assertIn("exposure", s.fields)
        self.assertIn("zoom", s.fields)

    def test_serialization_includes_both(self):
        s = ExtendedCamera()
        s.exposure = 2000
        s.zoom = 5.0
        d = s.to_dict()
        self.assertEqual(d["exposure"], 2000)
        self.assertEqual(d["zoom"], 5.0)


# ── None value rejection ──────────────────────────────────────────────────

class TestNoneRejection(unittest.TestCase):
    """None should be rejected for any typed field."""

    def test_none_int(self):
        s = CameraSettings()
        with self.assertRaises(TypeError):
            s.exposure = None # type: ignore

    def test_none_float(self):
        s = CameraSettings()
        with self.assertRaises(TypeError):
            s.gain = None # type: ignore

    def test_none_custom_type(self):
        s = CameraSettings()
        with self.assertRaises(TypeError):
            s.overlay_color = None # type: ignore


# ── Field repr branches ──────────────────────────────────────────────────

class TestFieldReprBranches(unittest.TestCase):
    """Cover all repr branches."""

    def test_repr_with_init_access(self):
        f = Field(42, access=Field.INIT)
        r = repr(f)
        self.assertIn("access=Field.INIT", r)

    def test_repr_with_invisible(self):
        f = Field(0, visible=False)
        r = repr(f)
        self.assertIn("visible=False", r)

    def test_repr_basic(self):
        f = Field(1.0)
        r = repr(f)
        self.assertIn("float", r)
        self.assertIn("1.0", r)
        self.assertTrue(r.startswith("Field("))


# ── load() return value ──────────────────────────────────────────────────

class TestLoadReturnValue(unittest.TestCase):
    """load() should return True on success, False on failure."""

    def _make_root(self):
        class Root(Settings):
            camera = Group(CameraSettings)
        return Root()

    def test_load_success_returns_true(self):
        root = self._make_root()
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "test.json")
            presets.save(root, p)
            self.assertTrue(presets.load(root, p))

    def test_load_missing_returns_false(self):
        root = self._make_root()
        self.assertFalse(presets.load(root, "/nonexistent/path.json"))

    def test_load_corrupt_returns_false(self):
        root = self._make_root()
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "bad.json")
            with open(p, "w") as f:
                f.write("{corrupted json!!")
            self.assertFalse(presets.load(root, p))


# ── presets utility functions ─────────────────────────────────────────────

class TestPresetsUtilities(unittest.TestCase):
    """Tests for path(), scan(), get_startup(), set_startup()."""

    def test_path_returns_expected(self):
        p = presets.path("mypreset")
        self.assertEqual(p.name, "mypreset.json")

    def test_set_startup_get_startup_round_trip(self):
        with tempfile.TemporaryDirectory() as d:
            from pathlib import Path
            orig_dir = presets.SETTINGS_DIR
            presets.SETTINGS_DIR = Path(d)
            try:
                presets.set_startup("custom")
                self.assertEqual(presets.get_startup(), "custom")
            finally:
                presets.SETTINGS_DIR = orig_dir

    def test_get_startup_default_fallback(self):
        with tempfile.TemporaryDirectory() as d:
            from pathlib import Path
            orig_dir = presets.SETTINGS_DIR
            presets.SETTINGS_DIR = Path(d)
            try:
                # No _startup_preset.txt → should return "default"
                self.assertEqual(presets.get_startup(), "default")
            finally:
                presets.SETTINGS_DIR = orig_dir

    def test_scan_returns_sorted_names(self):
        with tempfile.TemporaryDirectory() as d:
            from pathlib import Path
            orig_dir = presets.SETTINGS_DIR
            presets.SETTINGS_DIR = Path(d)
            try:
                (Path(d) / "beta.json").write_text("{}")
                (Path(d) / "alpha.json").write_text("{}")
                result = presets.scan()
                self.assertEqual(result, ["alpha", "beta"])
            finally:
                presets.SETTINGS_DIR = orig_dir

    def test_set_startup_rejects_path_traversal(self):
        with self.assertRaises(ValueError):
            presets.set_startup("../etc/passwd")

    def test_set_startup_rejects_empty(self):
        with self.assertRaises(ValueError):
            presets.set_startup("")

    def test_set_startup_rejects_dotfile(self):
        with self.assertRaises(ValueError):
            presets.set_startup(".hidden")

    def test_pinned_attribute(self):
        f = Field(42, pinned=True)
        self.assertTrue(f.pinned)
        f2 = Field(42)
        self.assertFalse(f2.pinned)


# ── Bug-fix regression tests ──────────────────────────────────────────────

class TestToJsonValueListElements(unittest.TestCase):
    """Bug 1: to_json_value should serialize list elements with to_dict()."""

    def test_list_of_colors_serialized(self):
        class ColorListSettings(Settings):
            colors = Field([Color(1, 0, 0)])

        s = ColorListSettings()
        s.colors = [Color(1, 0, 0), Color(0, 1, 0)]
        d = s.to_dict()
        self.assertEqual(d["colors"], [
            {"r": 1, "g": 0, "b": 0, "a": 1.0},
            {"r": 0, "g": 1, "b": 0, "a": 1.0},
        ])

    def test_list_of_enums_still_works(self):
        s = EnumListSettings()
        s.modes = [RenderMode.WIREFRAME, RenderMode.SOLID]
        d = s.to_dict()
        self.assertEqual(d["modes"], ["WIREFRAME", "SOLID"])

    def test_list_of_ints_still_works(self):
        s = ListSettings()
        s.ids = [4, 5, 6]
        d = s.to_dict()
        self.assertEqual(d["ids"], [4, 5, 6])


class TestLossyFloatToIntRejection(unittest.TestCase):
    """Bug 2: non-integer floats must be rejected in int list coercion."""

    def test_integer_floats_accepted(self):
        """1.0, 2.0, 3.0 are lossless → still accepted."""
        s = ListSettings()
        s.ids = [1.0, 2.0, 3.0]  # type: ignore
        self.assertEqual(s.ids, [1, 2, 3])

    def test_non_integer_float_rejected(self):
        """1.5 is lossy → must raise TypeError."""
        s = ListSettings()
        with self.assertRaises(TypeError):
            s.ids = [1, 2.5, 3]  # type: ignore

    def test_negative_non_integer_rejected(self):
        s = ListSettings()
        with self.assertRaises(TypeError):
            s.ids = [-0.7]  # type: ignore


class TestListCopyOnRead(unittest.TestCase):
    """Bug 4: __get__ returns a shallow copy so in-place mutation can't corrupt state."""

    def test_append_does_not_mutate_internal(self):
        s = ListSettings()
        got = s.ids
        got.append(999)
        self.assertEqual(s.ids, [1, 2, 3])  # internal unchanged

    def test_clear_does_not_mutate_internal(self):
        s = ListSettings()
        got = s.ids
        got.clear()
        self.assertEqual(s.ids, [1, 2, 3])

    def test_consecutive_reads_are_equal(self):
        s = ListSettings()
        self.assertEqual(s.ids, s.ids)

    def test_read_is_not_same_object(self):
        s = ListSettings()
        self.assertIsNot(s.ids, s.ids)


# ── Group descriptor tests ────────────────────────────────────────────────

from modules.settings.group import Group


class CoreSettings(Settings):
    """Standalone settings — no knowledge of any parent."""
    fps      = Field(30.0, access=Field.INIT)
    color    = Field(True, access=Field.INIT)
    exposure = Field(1000)


class PlayerSettings(Settings):
    folder = Field("")
    playing = Field(False, access=Field.READ)


class ParentWithChild(Settings):
    fps    = Field(30.0, access=Field.INIT)
    color  = Field(True, access=Field.INIT)
    player = Group(PlayerSettings)
    core_0 = Group(CoreSettings, share=[fps, color])
    core_1 = Group(CoreSettings, share=[fps, color])
    core_2 = Group(CoreSettings, share=[fps, color])

    @property
    def cores(self) -> list[CoreSettings]:
        return [self.core_0, self.core_1, self.core_2]


class TestGroupSingle(unittest.TestCase):
    """Tests for Group(Type) — single child instance."""

    def test_access(self):
        s = ParentWithChild()
        self.assertIsInstance(s.player, PlayerSettings)
        self.assertEqual(s.player.folder, "")

    def test_mutate(self):
        s = ParentWithChild()
        s.player.folder = "test"
        self.assertEqual(s.player.folder, "test")

    def test_not_replaceable(self):
        s = ParentWithChild()
        with self.assertRaises(AttributeError):
            s.player = PlayerSettings()

    def test_parent_wired(self):
        s = ParentWithChild()
        self.assertIs(s.player.parent, s)

    def test_root_parent_is_none(self):
        s = ParentWithChild()
        self.assertIsNone(s.parent)

    def test_in_children_dict(self):
        s = ParentWithChild()
        self.assertIn("player", s.children)

    def test_instances_independent(self):
        a = ParentWithChild()
        b = ParentWithChild()
        a.player.folder = "a"
        self.assertEqual(b.player.folder, "")

    def test_class_access_returns_descriptor(self):
        self.assertIsInstance(ParentWithChild.__dict__["player"], Group)


class TestGroupMulti(unittest.TestCase):
    """Tests for named Group children accessed via property."""

    def test_count(self):
        s = ParentWithChild()
        self.assertEqual(len(s.cores), 3)

    def test_index_access(self):
        s = ParentWithChild()
        self.assertIsInstance(s.cores[0], CoreSettings)
        self.assertEqual(s.cores[0].exposure, 1000)

    def test_each_instance_independent(self):
        s = ParentWithChild()
        s.core_0.exposure = 500
        self.assertEqual(s.core_1.exposure, 1000)

    def test_returns_list(self):
        s = ParentWithChild()
        self.assertIsInstance(s.cores, list)

    def test_parent_wired(self):
        s = ParentWithChild()
        for core in s.cores:
            self.assertIs(core.parent, s)

    def test_not_replaceable(self):
        s = ParentWithChild()
        with self.assertRaises(AttributeError):
            s.core_0 = CoreSettings()


class TestGroupShare(unittest.TestCase):
    """Tests for share=[fps, color] — parent pushes values to children."""

    def test_shared_values_propagated(self):
        s = ParentWithChild()
        for core in s.cores:
            self.assertEqual(core.fps, 30.0)
            self.assertEqual(core.color, True)

    def test_shared_from_kwargs(self):
        s = ParentWithChild(fps=60.0, color=False)
        for core in s.cores:
            self.assertEqual(core.fps, 60.0)
            self.assertEqual(core.color, False)

    def test_shared_excluded_from_child_serialization(self):
        s = ParentWithChild()
        d = s.to_dict()
        for i in range(3):
            core_dict = d[f"core_{i}"]
            self.assertNotIn("fps", core_dict)
            self.assertNotIn("color", core_dict)
            self.assertIn("exposure", core_dict)

    def test_shared_present_on_parent_serialization(self):
        s = ParentWithChild()
        d = s.to_dict()
        self.assertIn("fps", d)
        self.assertIn("color", d)

    def test_non_shared_fields_not_affected(self):
        s = ParentWithChild()
        self.assertEqual(s.core_0.exposure, 1000)

    def test_share_re_propagated_on_update_from_dict(self):
        s = ParentWithChild()
        s.update_from_dict({"fps": 90.0})
        for core in s.cores:
            self.assertEqual(core.fps, 90.0)

    def test_round_trip(self):
        s1 = ParentWithChild(fps=45.0, color=False)
        s1.core_0.exposure = 2000
        s1.core_2.exposure = 500
        data = s1.to_dict()

        s2 = ParentWithChild()
        s2.update_from_dict(data)
        self.assertEqual(s2.fps, 45.0)
        self.assertEqual(s2.core_0.fps, 45.0)
        self.assertEqual(s2.core_0.exposure, 2000)
        self.assertEqual(s2.core_2.exposure, 500)


class TestGroupSerialization(unittest.TestCase):
    """Tests for to_dict / update_from_dict with Group descriptors."""

    def test_single_child_serializes_as_dict(self):
        s = ParentWithChild()
        d = s.to_dict()
        self.assertIsInstance(d["player"], dict)

    def test_named_children_serialize_as_dicts(self):
        s = ParentWithChild()
        d = s.to_dict()
        for i in range(3):
            self.assertIsInstance(d[f"core_{i}"], dict)

    def test_initialize_recurses_into_children(self):
        s = ParentWithChild()
        s.initialize()
        self.assertTrue(s._initialized)
        self.assertTrue(s.player._initialized)
        for core in s.cores:
            self.assertTrue(core._initialized)

    def test_initialize_locks_shared_fields(self):
        s = ParentWithChild()
        s.initialize()
        with self.assertRaises(AttributeError):
            s.core_0.fps = 999.0


class TestGroupRepr(unittest.TestCase):
    """Tests for __repr__ with Group descriptors."""

    def test_repr_single(self):
        s = ParentWithChild()
        r = repr(s)
        self.assertIn("player=PlayerSettings(...)", r)

    def test_repr_multi(self):
        s = ParentWithChild()
        r = repr(s)
        self.assertIn("core_0=CoreSettings(...)", r)


class TestGroupEquality(unittest.TestCase):
    """Tests for __eq__ with Group descriptors."""

    def test_equal(self):
        a = ParentWithChild()
        b = ParentWithChild()
        self.assertEqual(a, b)

    def test_not_equal_field(self):
        a = ParentWithChild()
        b = ParentWithChild()
        b.core_0.exposure = 999
        self.assertNotEqual(a, b)

    def test_not_equal_single_child(self):
        a = ParentWithChild()
        b = ParentWithChild()
        b.player.folder = "different"
        self.assertNotEqual(a, b)


class TestValidateShare(unittest.TestCase):
    """Tests for validate_share() — catches mismatched shared fields."""

    def test_share_field_missing_from_parent(self):
        with self.assertRaises(TypeError) as ctx:
            class BadParent(Settings):
                core = Group(CoreSettings, share=[CoreSettings.exposure])
            BadParent()
        self.assertIn("not found on parent", str(ctx.exception))

    def test_share_field_missing_from_child(self):
        with self.assertRaises(TypeError) as ctx:
            class ChildWithoutFps(Settings):
                name = Field("")

            class BadParent2(Settings):
                fps = Field(30.0, access=Field.INIT)
                item = Group(ChildWithoutFps, share=[fps])
            BadParent2()
        self.assertIn("not found on child", str(ctx.exception))

    def test_share_type_mismatch(self):
        with self.assertRaises(TypeError) as ctx:
            class ChildIntFps(Settings):
                fps = Field(30)  # int, not float

            class BadParent3(Settings):
                fps = Field(30.0)  # float
                item = Group(ChildIntFps, share=[fps])
            BadParent3()
        self.assertIn("type mismatch", str(ctx.exception))

    def test_valid_share_passes(self):
        # Should not raise — ParentWithChild has matching fps/color
        s = ParentWithChild()
        self.assertEqual(len(s.cores), 3)



if __name__ == "__main__":
    unittest.main()
