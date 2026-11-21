"""A settings instance: its attribute guard, isolation, equality, repr, properties and field order."""

import unittest

from modules.settings import Field, Widget, BaseSettings, Group

from .fixtures import RenderMode, CameraSettings, MinimalSettings, InnerSettings, OuterSettings


class SettingsWithProperty(BaseSettings):
    raw_val = Field(0)

    @property
    def doubled(self):
        return self.raw_val * 2

    @doubled.setter
    def doubled(self, value):
        self.raw_val = value // 2


class ExtendedCamera(CameraSettings):
    zoom = Field(1.0, min=1.0, max=10.0)


class _OrderBase(BaseSettings):
    reset = Field(False, widget=Widget.button)
    width = Field(1)


class _OrderMixin(BaseSettings):
    gain = Field(1.0)
    width = Field(2)
    inner = Group(MinimalSettings)


class _OrderChild(_OrderMixin, _OrderBase):
    own = Field(0)


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


class TestFieldOrder(unittest.TestCase):
    """Fields and groups are listed base classes first; a name declared again keeps the base's place
    and takes the most derived descriptor."""

    def test_base_fields_come_first(self):
        self.assertEqual(list(_OrderChild().fields), ['reset', 'width', 'gain', 'own'])

    def test_a_redeclared_field_takes_the_most_derived_descriptor(self):
        s = _OrderChild()
        self.assertIs(s.fields['width'], _OrderMixin.width)
        self.assertEqual(s.width, 2)

    def test_groups_are_collected(self):
        self.assertEqual(list(_OrderChild().children), ['inner'])


if __name__ == "__main__":
    unittest.main()
