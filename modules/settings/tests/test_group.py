"""Group: a child instance, single or several, its serialization, repr, equality and wiring checks."""

import json
import unittest

from modules.settings import Field, BaseSettings, Group

from .fixtures import InnerSettings, OuterSettings, CoreSettings, PlayerSettings, ParentWithChild


class DoubleChildSettings(BaseSettings):
    name = Field("default")
    alpha = Group(InnerSettings)
    beta = Group(InnerSettings)


class MiddleSettings(BaseSettings):
    weight = Field(0.5, min=0.0, max=1.0)
    inner = Group(InnerSettings)


class DeepSettings(BaseSettings):
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

    def test_not_replaceable(self):
        s = ParentWithChild()
        with self.assertRaises(AttributeError):
            s.core_0 = CoreSettings()


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
            class BadParent(BaseSettings):
                core = Group(CoreSettings, share=[CoreSettings.exposure])
            BadParent()
        self.assertIn("not found on parent", str(ctx.exception))

    def test_share_field_missing_from_child(self):
        with self.assertRaises(TypeError) as ctx:
            class ChildWithoutFps(BaseSettings):
                name = Field("")

            class BadParent2(BaseSettings):
                fps = Field(30.0, access=Field.INIT)
                item = Group(ChildWithoutFps, share=[fps])
            BadParent2()
        self.assertIn("not found on child", str(ctx.exception))

    def test_share_type_mismatch(self):
        with self.assertRaises(TypeError) as ctx:
            class ChildIntFps(BaseSettings):
                fps = Field(30)  # int, not float

            class BadParent3(BaseSettings):
                fps = Field(30.0)  # float
                item = Group(ChildIntFps, share=[fps])
            BadParent3()
        self.assertIn("type mismatch", str(ctx.exception))

    def test_valid_share_passes(self):
        # Should not raise — ParentWithChild has matching fps/color
        s = ParentWithChild()
        self.assertEqual(len(s.cores), 3)

    def test_share_access_mismatch_raises(self):
        """Non-INIT parent field pushed to INIT child field must raise at construction."""
        with self.assertRaises(TypeError) as ctx:
            class ChildWithInitFps(BaseSettings):
                fps = Field(30.0, access=Field.INIT)

            class BadAccessParent(BaseSettings):
                fps = Field(30.0)  # WRITE, not INIT
                item = Group(ChildWithInitFps, share=[fps])

            BadAccessParent()
        self.assertIn("access mismatch", str(ctx.exception))
        self.assertIn("INIT", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
