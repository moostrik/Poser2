"""The settings mirror: a tree described and rebuilt holds the same fields, hints, labels and wiring, and
values travel both ways."""

import unittest
from enum import Enum, IntEnum
from pathlib import Path

from modules.settings import BaseSettings, Field, Group, Widget, mirror
from modules.utils import Color, Point2f, Rect


class Mode(IntEnum):
    A = 1
    B = 2
    C = 3

    @property
    def label(self) -> str:
        return f"mode {self.value}"


class Blend(Enum):
    ADD = (1, 2)
    MULTIPLY = (3, 4)


class Child(BaseSettings):
    rate = Field(30.0)
    gain = Field(0.5, min=0.0, max=1.0, step=0.01, description="The gain", unit="x", width=32)


class Probe(BaseSettings):
    rate = Field(30.0)
    flag = Field(True, pinned=True)
    count = Field(3, access=Field.INIT)
    name = Field("abc", newline=True, row_label="Name")
    mode = Field(Mode.B, min=1, max=2)
    blend = Field(Blend.ADD)
    modes = Field([Mode.A, Mode.C], widget=Widget.playlist)
    numbers = Field([1.0, 2.0])
    tint = Field(Color(1.0, 0.0, 0.0), widget=Widget.color_alpha)
    spot = Field(Point2f(0.1, 0.2))
    area = Field(Rect(0.0, 0.0, 1.0, 1.0))
    fps = Field(0.0, access=Field.READ, saved=False)
    reset = Field(False, widget=Widget.button)
    position = Field([0.0, 10.0], visible=False)
    drag = Field(-1.0, visible=False)
    seek = Field(-1.0, visible=False)
    span = Field([0.0, 1.0], widget=Widget.timeline, options=position, drag=drag, seek=seek)
    where = Field(Path("here"))
    child = Group(Child, share=[rate])


def made() -> tuple[Probe, BaseSettings]:
    tree = Probe()
    tree.initialize()
    copy = mirror.build_mirror(mirror.describe(tree))
    mirror.apply(copy, mirror.values(tree))
    return tree, copy


class TestDescribe(unittest.TestCase):
    def test_the_same_instances_and_fields_in_order(self):
        tree, copy = made()
        paths = [(path, list(s.fields)) for path, s in mirror.walk(tree)]
        self.assertEqual(paths, [(path, list(s.fields)) for path, s in mirror.walk(copy)])
        self.assertEqual([path for path, _ in paths], ["", "child"])

    def test_the_hints_and_access(self):
        tree, copy = made()
        for (_, real), (_, other) in zip(mirror.walk(tree), mirror.walk(copy)):
            for name, field in real.fields.items():
                twin = other.fields[name]
                for hint in ("min", "max", "step", "description", "label", "unit", "row_label", "width", "access",
                             "visible", "pinned", "newline", "color", "saved"):
                    self.assertEqual(getattr(field, hint), getattr(twin, hint), f"{name}.{hint}")
                if name != "where":
                    self.assertIs(Widget.resolve(field), Widget.resolve(twin), name)
                    self.assertEqual(field.encode(field.default), twin.encode(twin.default), name)

    def test_an_enum_keeps_its_names_values_and_labels(self):
        _, copy = made()
        mode = copy.fields["mode"].type_
        self.assertEqual([(m.name, m.value, m.label) for m in mode], [("A", 1, "mode 1"), ("B", 2, "mode 2"), ("C", 3, "mode 3")])
        blend = copy.fields["blend"].type_
        self.assertEqual([(m.name, m.value) for m in blend], [("ADD", 0), ("MULTIPLY", 1)])
        self.assertIsNone(getattr(next(iter(blend)), "label", None))

    def test_one_enum_for_every_field_that_uses_it(self):
        _, copy = made()
        self.assertIs(copy.fields["mode"].type_, copy.fields["modes"]._element_type)

    def test_the_wiring(self):
        tree, copy = made()
        child = copy.children["child"]
        self.assertTrue(child.is_shared("rate"))
        self.assertFalse(child.is_shared("gain"))
        self.assertTrue(copy.is_share_source("rate"))
        self.assertEqual(child.wiring_label("rate"), tree.child.wiring_label("rate"))
        self.assertEqual(copy.wiring_label("rate"), tree.wiring_label("rate"))

    def test_a_timeline_names_its_fields(self):
        _, copy = made()
        span = copy.fields["span"]
        self.assertIs(span.options, copy.fields["position"])
        self.assertIs(span.drag, copy.fields["drag"])
        self.assertIs(span.seek, copy.fields["seek"])

    def test_a_type_without_a_kind_is_shown_as_text(self):
        _, copy = made()
        self.assertEqual(copy.where, "here")
        self.assertIs(Widget.resolve(copy.fields["where"]), Widget.default)

    def test_the_description_is_plain(self):
        description = mirror.describe(Probe())

        def plain(value) -> bool:
            if isinstance(value, dict):
                return all(isinstance(k, str) and plain(v) for k, v in value.items())
            if isinstance(value, list):
                return all(plain(v) for v in value)
            return value is None or type(value) in (bool, int, float, str)
        self.assertTrue(plain(description))


class TestValues(unittest.TestCase):
    def test_the_tree_values_reach_the_mirror(self):
        tree, copy = made()
        tree.rate = 60.0
        tree.mode = Mode.C
        tree.blend = Blend.MULTIPLY
        tree.modes = [Mode.B]
        tree.tint = Color(0.0, 0.5, 1.0, 0.25)
        tree.area = Rect(1.0, 2.0, 3.0, 4.0)
        tree.fps = 59.5
        mirror.apply(copy, mirror.values(tree))
        self.assertEqual(mirror.values(copy), mirror.values(tree))
        self.assertEqual(copy.children["child"].rate, 60.0)
        self.assertEqual(copy.count, 3)

    def test_a_mirror_write_reaches_the_tree(self):
        tree, copy = made()
        for name, value in (("mode", copy.fields["mode"].type_["A"]), ("tint", Color(0.2, 0.3, 0.4)),
                            ("spot", Point2f(5.0, 6.0)), ("modes", [copy.fields["mode"].type_["C"]]), ("name", "xyz")):
            setattr(copy, name, value)
            setattr(tree, name, mirror.encode_value(copy.fields[name], getattr(copy, name)))
        self.assertEqual(mirror.values(tree), mirror.values(copy))

    def test_a_refused_value_is_logged_and_skipped(self):
        _, copy = made()
        with self.assertLogs("modules.settings.mirror", level="WARNING"):
            mirror.apply(copy, {"": {"rate": "fast", "nothing": 1}, "nowhere": {"rate": 1.0}})
        self.assertEqual(copy.rate, 30.0)


class TestValueDiff(unittest.TestCase):
    def test_nothing_changed(self):
        tree = Probe()
        self.assertEqual(mirror.ValueDiff(tree).changes(), {})

    def test_a_change_is_sent_once(self):
        tree = Probe()
        diff = mirror.ValueDiff(tree)
        tree.rate = 60.0
        tree.tint = Color(0.0, 1.0, 0.0)
        self.assertEqual(diff.changes(), {"": {"rate": 60.0, "tint": {"r": 0.0, "g": 1.0, "b": 0.0, "a": 1.0}},
                                          "child": {"rate": 60.0}})
        self.assertEqual(diff.changes(), {})

    def test_a_written_value_counts_as_sent(self):
        tree = Probe()
        diff = mirror.ValueDiff(tree)
        tree.rate = 45.0
        diff.written("", "rate", 45.0)
        self.assertEqual(diff.changes(), {"child": {"rate": 45.0}})     # the share the other side cannot see

    def test_a_refused_write_goes_back(self):
        tree = Probe()
        diff = mirror.ValueDiff(tree)
        diff.written("", "name", "never set")
        self.assertEqual(diff.changes(), {"": {"name": "abc"}})


if __name__ == "__main__":
    unittest.main()
