"""A field: its type and coercion, its lists and enums, its hints and its repr."""

import json
import unittest
from enum import Enum

from modules.settings import Field, BaseSettings
from modules.utils import Color

from .fixtures import RenderMode, CameraSettings, MinimalSettings


class ListSettings(BaseSettings):
    tags = Field(["default"])
    ids = Field([1, 2, 3])


class EnumListSettings(BaseSettings):
    modes = Field([RenderMode.SOLID])


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
            s.ids = [1, 2.5, 3]

    def test_negative_non_integer_rejected(self):
        s = ListSettings()
        with self.assertRaises(TypeError):
            s.ids = [-0.7]  # type: ignore


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
        class PrioSettings(BaseSettings):
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
        class FloatListSettings(BaseSettings):
            vals = Field([1.0, 2.0])
        s = FloatListSettings()
        with self.assertRaises(TypeError):
            s.vals = [False, 2.0]


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


class TestDeepCopyDefaults(unittest.TestCase):
    """List defaults must be deep-copied between instances (#10)."""

    def test_list_not_shared(self):
        class TagSettings(BaseSettings):
            tags = Field(["a", "b"])

        s1 = TagSettings()
        s2 = TagSettings()
        s1.tags = s1.tags + ["c"]
        self.assertEqual(s2.tags, ["a", "b"])

    def test_nested_list_not_shared(self):
        class NestedList(BaseSettings):
            items = Field(["x"])

        a = NestedList()
        b = NestedList()
        # Directly mutate the internal default (simulates shallow-copy bug)
        self.assertIsNot(a._values["items"], b._values["items"])


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

    def test_newline_metadata(self):
        field = Field(0, newline=True)
        self.assertTrue(field.newline)

    def test_label_metadata(self):
        self.assertIsNone(Field(0).label)                       # the panel derives it from the name
        self.assertEqual(Field(0, label="Amount").label, "Amount")

    def test_row_label_and_width_metadata(self):
        plain = Field(0)
        self.assertIsNone(plain.row_label)
        self.assertIsNone(plain.width)
        hinted = Field(0, row_label="Interval", width=32)
        self.assertEqual(hinted.row_label, "Interval")
        self.assertEqual(hinted.width, 32)


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


class TestToJsonValueListElements(unittest.TestCase):
    """Bug 1: to_json_value should serialize list elements with to_dict()."""

    def test_list_of_colors_serialized(self):
        class ColorListSettings(BaseSettings):
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


if __name__ == "__main__":
    unittest.main()
