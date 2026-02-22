"""Tests for the reactive settings system."""

import json
import os
import tempfile
import threading
import unittest
from enum import Enum

from modules.settings.Setting_ import Setting
from modules.settings.Action import Action
from modules.settings.Child_ import Child
from modules.settings.base_settings import BaseSettings
from modules.settings.Registry_ import SettingsRegistry


# ---------------------------------------------------------------------------
# Helper types
# ---------------------------------------------------------------------------

class Color:
    """Immutable color type for testing type coercion and serialization."""

    def __init__(self, r: float, g: float, b: float) -> None:
        self.r = r
        self.g = g
        self.b = b

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Color):
            return NotImplemented
        return (self.r, self.g, self.b) == (other.r, other.g, other.b)

    def __repr__(self) -> str:
        return f"Color({self.r}, {self.g}, {self.b})"

    def to_dict(self) -> dict:
        return {"r": self.r, "g": self.g, "b": self.b}

    @classmethod
    def from_dict(cls, data: dict) -> "Color":
        return cls(data["r"], data["g"], data["b"])


class RenderMode(Enum):
    WIREFRAME = "wireframe"
    SOLID = "solid"
    TEXTURED = "textured"


# ---------------------------------------------------------------------------
# Test settings classes
# ---------------------------------------------------------------------------

class CameraSettings(BaseSettings):
    exposure = Setting(int, 1000, min=100, max=10000, step=100, description="Exposure time (µs)")
    gain = Setting(float, 1.0, min=0.0, max=16.0, step=0.1)
    overlay_color = Setting(Color, Color(1, 0, 0))
    resolution = Setting(int, 1080, init_only=True)
    fps = Setting(float, 0.0, readonly=True, visible=False)
    mode = Setting(RenderMode, RenderMode.SOLID)


class MinimalSettings(BaseSettings):
    value = Setting(int, 0)


class SettingsWithActions(BaseSettings):
    exposure = Setting(int, 1000)
    reset = Action(description="Reset all values")
    hidden_action = Action(visible=False)


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
        s.overlay_color = (0.5, 0.5, 0.5)
        self.assertEqual(s.overlay_color, Color(0.5, 0.5, 0.5))

    def test_list_coercion(self):
        s = CameraSettings()
        s.overlay_color = [0.2, 0.3, 0.4]
        self.assertEqual(s.overlay_color, Color(0.2, 0.3, 0.4))

    def test_dict_coercion_from_dict(self):
        s = CameraSettings()
        s.overlay_color = {"r": 0.1, "g": 0.2, "b": 0.3}
        self.assertEqual(s.overlay_color, Color(0.1, 0.2, 0.3))

    def test_wrong_type_raises(self):
        s = CameraSettings()
        with self.assertRaises(TypeError):
            s.exposure = "not a number"

    def test_wrong_tuple_raises(self):
        s = MinimalSettings()
        with self.assertRaises(TypeError):
            s.value = (1, 2, 3)  # int(*[1,2,3]) fails

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
        s.on_change("fps", lambda v: results.append(v))
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
        s.on_change("exposure", lambda v: results.append(v))
        s.exposure = 2000
        self.assertEqual(results, [2000])

    def test_callback_not_fired_on_same_value(self):
        s = CameraSettings()
        results = []
        s.on_change("exposure", lambda v: results.append(v))
        s.exposure = 1000  # same as default
        self.assertEqual(results, [])

    def test_multiple_callbacks(self):
        s = CameraSettings()
        a, b = [], []
        s.on_change("exposure", lambda v: a.append(v))
        s.on_change("exposure", lambda v: b.append(v))
        s.exposure = 3000
        self.assertEqual(a, [3000])
        self.assertEqual(b, [3000])

    def test_remove_callback(self):
        s = CameraSettings()
        results = []
        cb = lambda v: results.append(v)
        s.on_change("exposure", cb)
        s.remove_callback("exposure", cb)
        s.exposure = 5000
        self.assertEqual(results, [])

    def test_decorator_syntax(self):
        s = CameraSettings()
        results = []

        @s.on_change("exposure")
        def on_exp(value):
            results.append(value)

        s.exposure = 4000
        self.assertEqual(results, [4000])

    def test_decorator_returns_function(self):
        s = CameraSettings()

        @s.on_change("gain")
        def on_gain(value):
            pass

        self.assertTrue(callable(on_gain))

    def test_on_change_unknown_field_raises(self):
        s = CameraSettings()
        with self.assertRaises(KeyError):
            s.on_change("nonexistent", lambda v: None)


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
        a.on_change("exposure", lambda v: results_a.append(v))
        b.on_change("exposure", lambda v: results_b.append(v))
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
        # init_only fields are excluded from serialization
        self.assertNotIn("resolution", d)

    def test_to_dict_values(self):
        s = CameraSettings(exposure=2000)
        d = s.to_dict()
        self.assertEqual(d["exposure"], 2000)

    def test_to_dict_custom_type_uses_to_dict(self):
        s = CameraSettings()
        d = s.to_dict()
        self.assertEqual(d["overlay_color"], {"r": 1, "g": 0, "b": 0})

    def test_to_dict_enum_uses_value(self):
        s = CameraSettings()
        d = s.to_dict()
        self.assertEqual(d["mode"], "solid")

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
        s.on_change("exposure", lambda v: results.append(v))
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

        s.on_change("value", increment)

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
            reg.save(path)

            # Verify JSON file content
            with open(path, "r") as f:
                raw = json.load(f)
            self.assertIn("camera", raw)
            self.assertEqual(raw["camera"]["exposure"], 3000)

            # Load into fresh settings
            s2 = CameraSettings()
            reg2 = SettingsRegistry()
            reg2.register("camera", s2)
            reg2.load(path)

            self.assertEqual(s2.exposure, 3000)
            # init_only fields stay at default
            self.assertEqual(s2.resolution, 1080)
            self.assertEqual(s2.mode, RenderMode.WIREFRAME)

    def test_load_fires_callbacks(self):
        s = CameraSettings()
        reg = SettingsRegistry()
        reg.register("camera", s)
        results = []
        s.on_change("exposure", lambda v: results.append(v))

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "settings.json")
            # Save with different value
            s.exposure = 5000
            reg.save(path)
            results.clear()

            # Reset and load
            s.exposure = 1000
            results.clear()
            reg.load(path)
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
        reg.load("nonexistent_path_12345.json")  # should not raise

    def test_load_corrupt_file_no_error(self):
        reg = SettingsRegistry()
        reg.register("cam", CameraSettings())
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("{corrupt json!!!")
            path = f.name
        try:
            reg.load(path)  # should not raise
        finally:
            os.unlink(path)


class TestAction(unittest.TestCase):
    """Tests for the Action descriptor."""

    def test_action_fires_callbacks(self):
        s = SettingsWithActions()
        results = []
        s.on_action("reset", lambda: results.append("fired"))
        s._actions["reset"].fire(s)
        self.assertEqual(results, ["fired"])

    def test_action_multiple_callbacks(self):
        s = SettingsWithActions()
        a, b = [], []
        s.on_action("reset", lambda: a.append(1))
        s.on_action("reset", lambda: b.append(2))
        s._actions["reset"].fire(s)
        self.assertEqual(a, [1])
        self.assertEqual(b, [2])

    def test_action_no_callbacks_ok(self):
        s = SettingsWithActions()
        s._actions["reset"].fire(s)  # should not raise

    def test_action_not_assignable(self):
        s = SettingsWithActions()
        with self.assertRaises(AttributeError):
            s.reset = True

    def test_action_decorator_syntax(self):
        s = SettingsWithActions()
        results = []

        @s.on_action("reset")
        def on_reset():
            results.append("decorated")

        s._actions["reset"].fire(s)
        self.assertEqual(results, ["decorated"])

    def test_on_action_unknown_raises(self):
        s = SettingsWithActions()
        with self.assertRaises(KeyError):
            s.on_action("nonexistent", lambda: None)

    def test_remove_action_callback(self):
        s = SettingsWithActions()
        results = []
        cb = lambda: results.append(1)
        s.on_action("reset", cb)
        s.remove_action_callback("reset", cb)
        s._actions["reset"].fire(s)
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
        s = SettingsWithActions()
        r = repr(s._actions["reset"])
        self.assertIn("Reset all values", r)

    def test_action_class_access_returns_descriptor(self):
        self.assertIsInstance(SettingsWithActions.reset, Action)

    def test_action_broken_callback_doesnt_crash(self):
        s = SettingsWithActions()
        results = []
        s.on_action("reset", lambda: 1 / 0)  # will raise
        s.on_action("reset", lambda: results.append("ok"))
        s._actions["reset"].fire(s)  # should not raise
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

class InnerConfig(BaseSettings):
    speed = Setting(float, 1.0, min=0.0, max=10.0)
    scale = Setting(float, 0.5)


class OuterConfig(BaseSettings):
    fps = Setting(float, 60.0, min=1.0, max=240.0)
    inner = Child(InnerConfig)


class DoubleChildConfig(BaseSettings):
    name = Setting(str, "default")
    alpha = Child(InnerConfig)
    beta = Child(InnerConfig)


class MiddleConfig(BaseSettings):
    weight = Setting(float, 0.5, min=0.0, max=1.0)
    inner = Child(InnerConfig)


class DeepConfig(BaseSettings):
    label = Setting(str, "root")
    middle = Child(MiddleConfig)


class TestChild(unittest.TestCase):
    """Tests for the Child descriptor."""

    # -- Basic access -------------------------------------------------------

    def test_child_access(self):
        cfg = OuterConfig()
        self.assertIsInstance(cfg.inner, InnerConfig)
        self.assertEqual(cfg.inner.speed, 1.0)
        self.assertEqual(cfg.inner.scale, 0.5)

    def test_child_field_mutation(self):
        cfg = OuterConfig()
        cfg.inner.speed = 5.0
        self.assertEqual(cfg.inner.speed, 5.0)

    def test_child_not_replaceable(self):
        cfg = OuterConfig()
        with self.assertRaises(AttributeError):
            cfg.inner = InnerConfig()

    def test_each_instance_gets_own_child(self):
        a = OuterConfig()
        b = OuterConfig()
        a.inner.speed = 9.0
        self.assertEqual(b.inner.speed, 1.0)

    # -- Children property --------------------------------------------------

    def test_children_property(self):
        cfg = OuterConfig()
        children = cfg.children
        self.assertIn("inner", children)
        self.assertIsInstance(children["inner"], InnerConfig)

    def test_children_property_returns_copy(self):
        cfg = OuterConfig()
        children = cfg.children
        children["fake"] = None
        self.assertNotIn("fake", cfg.children)

    # -- Multiple children --------------------------------------------------

    def test_multiple_children_independent(self):
        cfg = DoubleChildConfig()
        cfg.alpha.speed = 3.0
        cfg.beta.speed = 7.0
        self.assertEqual(cfg.alpha.speed, 3.0)
        self.assertEqual(cfg.beta.speed, 7.0)

    # -- Serialization: to_dict ---------------------------------------------

    def test_to_dict_includes_children(self):
        cfg = OuterConfig()
        cfg.inner.speed = 2.5
        d = cfg.to_dict()
        self.assertIn("fps", d)
        self.assertIn("inner", d)
        self.assertEqual(d["inner"]["speed"], 2.5)
        self.assertEqual(d["inner"]["scale"], 0.5)

    def test_to_dict_nested_structure(self):
        cfg = DoubleChildConfig()
        cfg.alpha.speed = 1.0
        cfg.beta.speed = 2.0
        d = cfg.to_dict()
        self.assertEqual(d["alpha"]["speed"], 1.0)
        self.assertEqual(d["beta"]["speed"], 2.0)
        self.assertEqual(d["name"], "default")

    # -- Serialization: update_from_dict ------------------------------------

    def test_update_from_dict_restores_children(self):
        cfg = OuterConfig()
        cfg.update_from_dict({"fps": 120.0, "inner": {"speed": 8.0, "scale": 0.25}})
        self.assertEqual(cfg.fps, 120.0)
        self.assertEqual(cfg.inner.speed, 8.0)
        self.assertEqual(cfg.inner.scale, 0.25)

    def test_update_from_dict_partial_child(self):
        cfg = OuterConfig()
        cfg.update_from_dict({"inner": {"speed": 4.0}})
        self.assertEqual(cfg.inner.speed, 4.0)
        self.assertEqual(cfg.inner.scale, 0.5)  # Unchanged

    def test_update_from_dict_ignores_unknown_child_keys(self):
        cfg = OuterConfig()
        cfg.update_from_dict({"inner": {"speed": 3.0, "nonexistent": 99}})
        self.assertEqual(cfg.inner.speed, 3.0)

    # -- Round-trip ---------------------------------------------------------

    def test_json_round_trip(self):
        cfg1 = OuterConfig()
        cfg1.fps = 144.0
        cfg1.inner.speed = 3.5
        cfg1.inner.scale = 0.75

        serialized = json.dumps(cfg1.to_dict())
        data = json.loads(serialized)

        cfg2 = OuterConfig()
        cfg2.update_from_dict(data)
        self.assertEqual(cfg2.fps, 144.0)
        self.assertEqual(cfg2.inner.speed, 3.5)
        self.assertEqual(cfg2.inner.scale, 0.75)

    # -- Callbacks on child fields ------------------------------------------

    def test_child_field_callbacks_still_fire(self):
        cfg = OuterConfig()
        received = []
        cfg.inner.on_change("speed", lambda v: received.append(v))
        cfg.inner.speed = 7.0
        self.assertEqual(received, [7.0])

    # -- Repr ---------------------------------------------------------------

    def test_repr_shows_children(self):
        cfg = OuterConfig()
        r = repr(cfg)
        self.assertIn("InnerConfig(...)", r)
        self.assertIn("fps=", r)

    # -- Class-level descriptor access --------------------------------------

    def test_class_level_access_returns_descriptor(self):
        self.assertIsInstance(OuterConfig.inner, Child)

    # -- Registry with children ---------------------------------------------

    def test_registry_save_load_with_children(self):
        reg = SettingsRegistry()
        cfg = OuterConfig()
        cfg.fps = 30.0
        cfg.inner.speed = 2.0
        reg.register("outer", cfg)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            path = f.name
        try:
            reg.save(path)

            reg2 = SettingsRegistry()
            cfg2 = OuterConfig()
            reg2.register("outer", cfg2)
            reg2.load(path)

            self.assertEqual(cfg2.fps, 30.0)
            self.assertEqual(cfg2.inner.speed, 2.0)
        finally:
            os.unlink(path)

    # -- Deep nesting (3 levels) --------------------------------------------

    def test_deep_access(self):
        cfg = DeepConfig()
        self.assertEqual(cfg.label, "root")
        self.assertIsInstance(cfg.middle, MiddleConfig)
        self.assertEqual(cfg.middle.weight, 0.5)
        self.assertIsInstance(cfg.middle.inner, InnerConfig)
        self.assertEqual(cfg.middle.inner.speed, 1.0)
        self.assertEqual(cfg.middle.inner.scale, 0.5)

    def test_deep_mutation(self):
        cfg = DeepConfig()
        cfg.middle.inner.speed = 9.9
        self.assertEqual(cfg.middle.inner.speed, 9.9)
        cfg.middle.weight = 0.1
        self.assertEqual(cfg.middle.weight, 0.1)

    def test_deep_to_dict(self):
        cfg = DeepConfig()
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
        cfg = DeepConfig()
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
        cfg = DeepConfig()
        cfg.update_from_dict({"middle": {"inner": {"speed": 5.5}}})
        self.assertEqual(cfg.middle.inner.speed, 5.5)
        self.assertEqual(cfg.middle.inner.scale, 0.5)  # unchanged
        self.assertEqual(cfg.middle.weight, 0.5)        # unchanged
        self.assertEqual(cfg.label, "root")              # unchanged

    def test_deep_json_round_trip(self):
        cfg1 = DeepConfig()
        cfg1.label = "trip"
        cfg1.middle.weight = 0.3
        cfg1.middle.inner.speed = 4.4
        cfg1.middle.inner.scale = 0.9

        serialized = json.dumps(cfg1.to_dict())
        data = json.loads(serialized)

        cfg2 = DeepConfig()
        cfg2.update_from_dict(data)
        self.assertEqual(cfg2.label, "trip")
        self.assertEqual(cfg2.middle.weight, 0.3)
        self.assertEqual(cfg2.middle.inner.speed, 4.4)
        self.assertEqual(cfg2.middle.inner.scale, 0.9)

    def test_deep_callback(self):
        cfg = DeepConfig()
        received = []
        cfg.middle.inner.on_change("speed", lambda v: received.append(v))
        cfg.middle.inner.speed = 6.6
        self.assertEqual(received, [6.6])

    def test_deep_instances_independent(self):
        a = DeepConfig()
        b = DeepConfig()
        a.middle.inner.speed = 100.0
        self.assertEqual(b.middle.inner.speed, 1.0)

    def test_deep_children_property(self):
        cfg = DeepConfig()
        self.assertIn("middle", cfg.children)
        self.assertIn("inner", cfg.middle.children)


if __name__ == "__main__":
    unittest.main()
