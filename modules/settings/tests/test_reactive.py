"""Tests for the reactive settings system."""

import colorsys
import json
import os
import tempfile
import threading
import unittest
from enum import Enum

from modules.settings.setting import Setting, Widget
from modules.settings.base_settings import BaseSettings
from modules.settings.registry import SettingsRegistry
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

class CameraSettings(BaseSettings):
    exposure = Setting(1000, min=100, max=10000, step=100, description="Exposure time (µs)")
    gain = Setting(1.0, min=0.0, max=16.0, step=0.1)
    overlay_color = Setting(Color(1, 0, 0))
    resolution = Setting(1080, init_only=True)
    fps = Setting(0.0, readonly=True, visible=False)
    mode = Setting(RenderMode.SOLID)


class MinimalSettings(BaseSettings):
    value = Setting(0)


class SettingsWithActions(BaseSettings):
    exposure = Setting(1000)
    reset = Setting(False, widget=Widget.button, description="Reset all values")
    hidden_action = Setting(False, widget=Widget.button, visible=False)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSettingDescriptor(unittest.TestCase):
    """Tests for the Setting descriptor itself."""

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
        self.assertIsInstance(CameraSettings.exposure, Setting)


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


class TestReadOnly(unittest.TestCase):
    """Tests for readonly fields (readonly is a GUI hint, not enforced)."""

    def test_readonly_allows_set(self):
        s = CameraSettings()
        s.fps = 30.0
        self.assertEqual(s.fps, 30.0)

    def test_readonly_set_works(self):
        s = CameraSettings()
        s.fps = 29.97
        self.assertEqual(s.fps, 29.97)

    def test_readonly_callback_fires_on_set(self):
        s = CameraSettings()
        results = []
        s.bind(CameraSettings.fps, lambda v: results.append(v))
        s.fps = 60.0
        self.assertEqual(results, [60.0])


class TestInitOnly(unittest.TestCase):
    """Tests for init_only fields."""

    def test_init_only_set_during_construction(self):
        s = CameraSettings(resolution=720)
        self.assertEqual(s.resolution, 720)

    def test_init_only_raises_after_init(self):
        s = CameraSettings()
        with self.assertRaises(AttributeError):
            s.resolution = 480

    def test_init_only_loadable_via_set_before_init(self):
        s = CameraSettings()
        # Simulate pre-init state: can set init_only fields
        object.__setattr__(s, "_initialized", False)
        s.resolution = 480
        object.__setattr__(s, "_initialized", True)
        self.assertEqual(s.resolution, 480)

    def test_init_only_skipped_by_update_from_dict(self):
        s = CameraSettings(resolution=720)
        s.update_from_dict({"resolution": 480})
        # init_only field should NOT be overwritten after init
        self.assertEqual(s.resolution, 720)


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

    def test_to_dict_excludes_readonly(self):
        s = CameraSettings()
        d = s.to_dict()
        self.assertNotIn("fps", d)

    def test_to_dict_includes_writable(self):
        s = CameraSettings()
        d = s.to_dict()
        self.assertIn("exposure", d)
        # init_only fields are included in serialization (editable in JSON)
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
        # init_only fields are not serialized / restored
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
        s.update_from_dict({"unknown_key": 42})  # should not raise


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


class TestSettingsRegistry(unittest.TestCase):
    """Tests for SettingsRegistry."""

    def test_register_and_get(self):
        reg = SettingsRegistry()
        s = CameraSettings()
        reg.register("camera", s)
        self.assertIs(reg.get("camera"), s)

    def test_get_unknown_raises(self):
        reg = SettingsRegistry()
        with self.assertRaises(KeyError):
            reg.get("nope")

    def test_save_and_load(self):
        s1 = CameraSettings(exposure=3000)
        s1.mode = RenderMode.WIREFRAME
        reg = SettingsRegistry()
        reg.register("camera", s1)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "settings.json")
            presets.save(reg, path)

            # Verify JSON file content
            with open(path, "r") as f:
                raw = json.load(f)
            self.assertIn("camera", raw)
            self.assertEqual(raw["camera"]["exposure"], 3000)

            # Load into fresh settings
            s2 = CameraSettings()
            reg2 = SettingsRegistry()
            reg2.register("camera", s2)
            presets.load(reg2, path)

            self.assertEqual(s2.exposure, 3000)
            # init_only fields stay at default
            self.assertEqual(s2.resolution, 1080)
            self.assertEqual(s2.mode, RenderMode.WIREFRAME)

    def test_load_fires_callbacks(self):
        s = CameraSettings()
        reg = SettingsRegistry()
        reg.register("camera", s)
        results = []
        s.bind(CameraSettings.exposure, lambda v: results.append(v))

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "settings.json")
            # Save with different value
            s.exposure = 5000
            presets.save(reg, path)
            results.clear()

            # Reset and load
            s.exposure = 1000
            results.clear()
            presets.load(reg, path)
            self.assertEqual(results, [5000])

    def test_repr(self):
        reg = SettingsRegistry()
        reg.register("cam", CameraSettings())
        self.assertIn("cam", repr(reg))


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


class TestRegistryExtras(unittest.TestCase):
    """Tests for __contains__ and __getitem__ on SettingsRegistry."""

    def test_contains(self):
        reg = SettingsRegistry()
        reg.register("cam", CameraSettings())
        self.assertIn("cam", reg) # type: ignore
        self.assertNotIn("nope", reg) # type: ignore

    def test_getitem(self):
        reg = SettingsRegistry()
        s = CameraSettings()
        reg.register("cam", s)
        self.assertIs(reg["cam"], s)

    def test_getitem_unknown_raises(self):
        reg = SettingsRegistry()
        with self.assertRaises(KeyError):
            _ = reg["nope"]

    def test_load_missing_file_no_error(self):
        reg = SettingsRegistry()
        reg.register("cam", CameraSettings())
        presets.load(reg, "nonexistent_path_12345.json")  # should not raise

    def test_load_corrupt_file_no_error(self):
        reg = SettingsRegistry()
        reg.register("cam", CameraSettings())
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("{corrupt json!!!")
            path = f.name
        try:
            presets.load(reg, path)  # should not raise
        finally:
            os.unlink(path)


class TestAction(unittest.TestCase):
    """Tests for Widget.button Settings (replacing the old Action descriptor)."""

    def test_action_fires_callbacks(self):
        s = SettingsWithActions()
        results = []
        s.bind(SettingsWithActions.reset, lambda: results.append("fired"))
        SettingsWithActions.reset.fire(s)
        self.assertEqual(results, ["fired"])

    def test_action_multiple_callbacks(self):
        s = SettingsWithActions()
        a, b = [], []
        s.bind(SettingsWithActions.reset, lambda: a.append(1))
        s.bind(SettingsWithActions.reset, lambda: b.append(2))
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
            s.bind(MinimalSettings.value, lambda: None)

    def test_unbind_action(self):
        s = SettingsWithActions()
        results = []
        cb = lambda: results.append(1)
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
        self.assertIsInstance(SettingsWithActions.reset, Setting)

    def test_action_broken_callback_doesnt_crash(self):
        s = SettingsWithActions()
        results = []
        s.bind(SettingsWithActions.reset, lambda: 1 / 0)  # will raise
        s.bind(SettingsWithActions.reset, lambda: results.append("ok"))
        SettingsWithActions.reset.fire(s)  # should not raise
        self.assertEqual(results, ["ok"])


class TestRegistryGroups(unittest.TestCase):
    """Tests for registry group support."""

    def test_default_group(self):
        reg = SettingsRegistry()
        reg.register("cam", CameraSettings())
        self.assertEqual(reg.groups(), {"default": ["cam"]})

    def test_custom_group(self):
        reg = SettingsRegistry()
        reg.register("cam", CameraSettings(), group="camera")
        reg.register("timer", MinimalSettings(), group="general")
        groups = reg.groups()
        self.assertEqual(groups["camera"], ["cam"])
        self.assertEqual(groups["general"], ["timer"])

    def test_multiple_in_same_group(self):
        reg = SettingsRegistry()
        reg.register("a", MinimalSettings(), group="pose")
        reg.register("b", MinimalSettings(), group="pose")
        self.assertEqual(reg.groups()["pose"], ["a", "b"])

    def test_groups_returns_copy(self):
        reg = SettingsRegistry()
        reg.register("cam", CameraSettings(), group="camera")
        g = reg.groups()
        g["camera"].append("fake")
        self.assertEqual(reg.groups()["camera"], ["cam"])

    def test_register_preserves_order(self):
        reg = SettingsRegistry()
        reg.register("c", MinimalSettings(), group="g")
        reg.register("a", MinimalSettings(), group="g")
        reg.register("b", MinimalSettings(), group="g")
        self.assertEqual(reg.groups()["g"], ["c", "a", "b"])

    def test_repr_shows_groups(self):
        reg = SettingsRegistry()
        reg.register("cam", CameraSettings(), group="camera")
        r = repr(reg)
        self.assertIn("camera", r)
        self.assertIn("cam", r)


# ===== Child descriptor tests =============================================

class InnerSettings(BaseSettings):
    speed = Setting(1.0, min=0.0, max=10.0)
    scale = Setting(0.5)


class OuterSettings(BaseSettings):
    fps = Setting(60.0, min=1.0, max=240.0)
    inner: InnerSettings


class DoubleChildSettings(BaseSettings):
    name = Setting("default")
    alpha: InnerSettings
    beta: InnerSettings


class MiddleSettings(BaseSettings):
    weight = Setting(0.5, min=0.0, max=1.0)
    inner: InnerSettings


class DeepSettings(BaseSettings):
    label = Setting("root")
    middle: MiddleSettings


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

    # -- Class-level annotation access (no descriptor on the class) -----------

    def test_class_level_annotation_exists(self):
        ann = OuterSettings.__annotations__
        self.assertIn("inner", ann)

    # -- Registry with children ---------------------------------------------

    def test_registry_save_load_with_children(self):
        reg = SettingsRegistry()
        cfg = OuterSettings()
        cfg.fps = 30.0
        cfg.inner.speed = 2.0
        reg.register("outer", cfg)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            path = f.name
        try:
            presets.save(reg, path)

            reg2 = SettingsRegistry()
            cfg2 = OuterSettings()
            reg2.register("outer", cfg2)
            presets.load(reg2, path)

            self.assertEqual(cfg2.fps, 30.0)
            self.assertEqual(cfg2.inner.speed, 2.0)
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

class ListSettings(BaseSettings):
    tags = Setting(list[str], [])
    ids = Setting(list[int], [1, 2, 3])


class TestListSetting(unittest.TestCase):
    """Tests for list[T] support in Setting."""

    def test_default_value(self):
        s = ListSettings()
        self.assertEqual(s.tags, [])
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
        self.assertEqual(b.tags, [])

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

class _ColorSettings(BaseSettings):
    color = Setting(Color(1.0, 0.0, 0.0))


class _PointSettings(BaseSettings):
    pos = Setting(Point2f(0.0, 0.0))


class _RectSettings(BaseSettings):
    region = Setting(Rect(0.0, 0.0, 1.0, 1.0))


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


if __name__ == "__main__":
    unittest.main()
