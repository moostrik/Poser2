"""Widget: which field types each member accepts, and the member ``resolve`` picks for a field."""

import unittest
from enum import Enum

from modules.settings import Field, Widget, BaseSettings
from modules.utils import Color, Point2f, Rect


class Mode(Enum):
    A = 1
    B = 2


class Probe(BaseSettings):
    flag = Field(True)
    count = Field(3)
    ratio = Field(0.5, min=0.0, max=1.0)
    name = Field("x")
    mode = Field(Mode.A)
    tint = Field(Color(1.0, 0.0, 0.0))
    spot = Field(Point2f(0.0, 0.0))
    box = Field(Rect(0.0, 0.0, 1.0, 1.0))
    modes = Field([Mode.A])
    weights = Field([1.0, 2.0])
    knob = Field(0.5, widget=Widget.knob)


class TestAccepts(unittest.TestCase):
    def test_members_are_distinct(self):
        self.assertEqual(len({w.value for w in Widget}), len(list(Widget)))

    def test_default_accepts_anything(self):
        for t in (bool, int, str, Mode, Color, list[int], object):
            self.assertTrue(Widget.default.accepts(t))

    def test_bool_widgets(self):
        for w in (Widget.switch, Widget.toggle, Widget.button, Widget.status):
            self.assertTrue(w.accepts(bool))
            self.assertFalse(w.accepts(int))

    def test_numeric_widgets_reject_bool(self):
        for w in (Widget.slider, Widget.log_slider, Widget.number, Widget.knob, Widget.number_field):
            self.assertTrue(w.accepts(int))
            self.assertTrue(w.accepts(float))
            self.assertFalse(w.accepts(bool))
            self.assertFalse(w.accepts(str))

    def test_enum_widgets_accept_any_enum(self):
        for w in (Widget.select, Widget.radio):
            self.assertTrue(w.accepts(Mode))
            self.assertFalse(w.accepts(str))

    def test_string_widgets(self):
        for w in (Widget.input, Widget.ip_field, Widget.textarea, Widget.text_select):
            self.assertTrue(w.accepts(str))
            self.assertFalse(w.accepts(int))

    def test_value_object_widgets_take_one_exact_type(self):
        self.assertTrue(Widget.color.accepts(Color))
        self.assertTrue(Widget.color_alpha.accepts(Color))
        self.assertTrue(Widget.point2f.accepts(Point2f))
        self.assertTrue(Widget.rect.accepts(Rect))
        for w in (Widget.color, Widget.color_alpha, Widget.point2f, Widget.rect):
            self.assertFalse(w.accepts(str))
            self.assertFalse(w.accepts(object))

    def test_list_widgets_accept_a_typed_list(self):
        for w in (Widget.checklist, Widget.playlist, Widget.order, Widget.number_list):
            self.assertTrue(w.accepts(list[Mode]))
            self.assertTrue(w.accepts(list))
            self.assertFalse(w.accepts(Mode))


class TestResolve(unittest.TestCase):
    def setUp(self):
        self.fields = Probe().fields

    def test_picks_by_type(self):
        expected = {
            "flag": Widget.toggle,
            "count": Widget.number,
            "ratio": Widget.slider,
            "name": Widget.input,
            "mode": Widget.select,
            "tint": Widget.color,
            "spot": Widget.point2f,
            "box": Widget.rect,
            "modes": Widget.checklist,
            "weights": Widget.number_list,
        }
        for name, widget in expected.items():
            with self.subTest(name=name):
                self.assertIs(Widget.resolve(self.fields[name]), widget)

    def test_keeps_a_set_widget(self):
        self.assertIs(Widget.resolve(self.fields["knob"]), Widget.knob)


if __name__ == "__main__":
    unittest.main()
