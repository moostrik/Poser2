"""The Control base class through a fake subclass, the registry, and the pure helpers."""

import unittest
from enum import Enum

from modules.settings import BaseSettings, Field, Widget
from modules.utils import Color, Point2f, Rect

from ..controls import (CONTROLS, Control, Fallback, access_css_class, control_class, format_readout,
                        format_time, generate_label)


class Mode(Enum):
    A = 1
    B = 2


class Probe(BaseSettings):
    gain = Field(1.0, min=0.0, max=10.0)
    readout = Field(0.5, access=Field.READ)
    port = Field(80, access=Field.INIT)
    names = Field(["a", "b"])
    modes = Field([Mode.A])


class FakeControl(Control):
    """Records what it was asked to show; builds nothing."""

    def build(self) -> None:
        self.shows: list = []

    def show(self, value) -> None:
        self.shows.append(value)


class TestControl(unittest.TestCase):
    def setUp(self):
        self.probe = Probe()
        self.probe.initialize()

    def test_resolves_label_lock_and_poll(self):
        gain = FakeControl(self.probe, "gain", self.probe.fields["gain"])
        self.assertEqual(gain.label, "Gain")
        self.assertFalse(gain.locked)
        self.assertTrue(gain.polled)        # READWRITE: the app may write it
        self.assertEqual(gain.shown, 1.0)
        readout = FakeControl(self.probe, "readout", self.probe.fields["readout"])
        self.assertTrue(readout.locked)
        self.assertTrue(readout.polled)

    def test_commit_writes_and_shows_the_field_value(self):
        gain = FakeControl(self.probe, "gain", self.probe.fields["gain"])
        gain.commit(2)
        self.assertEqual(self.probe.gain, 2.0)
        self.assertEqual(gain.shows, [2.0])
        self.assertEqual(gain.shown, 2.0)

    def test_poll_shows_only_what_changed_outside(self):
        gain = FakeControl(self.probe, "gain", self.probe.fields["gain"])
        gain.poll()
        self.assertEqual(gain.shows, [])
        gain.commit(3.0)
        gain.poll()                          # the echo guard: a committed value is not pushed back
        self.assertEqual(gain.shows, [3.0])
        self.probe.gain = 4.0
        gain.poll()
        gain.poll()
        self.assertEqual(gain.shows, [3.0, 4.0])

    def test_a_list_value_is_compared_by_content(self):
        names = FakeControl(self.probe, "names", self.probe.fields["names"])
        names.poll()
        self.assertEqual(names.shows, [])
        self.probe.names = ["a", "b", "c"]
        names.poll()
        self.assertEqual(names.shows, [["a", "b", "c"]])


class TestRegistry(unittest.TestCase):
    def test_every_widget_has_a_control(self):
        self.assertEqual(set(CONTROLS), set(Widget))

    def test_fallback_for_a_list_of_anything_but_enums(self):
        probe = Probe()
        self.assertIs(control_class(probe.fields["names"]), Fallback)
        self.assertIs(control_class(probe.fields["modes"]), CONTROLS[Widget.checklist])
        self.assertIs(control_class(probe.fields["gain"]), CONTROLS[Widget.slider])


class TestHelpers(unittest.TestCase):
    def test_generate_label(self):
        self.assertEqual(generate_label("min_cutoff"), "Min Cutoff")
        self.assertEqual(generate_label("TCP_PORT"), "TCP PORT")
        self.assertEqual(generate_label("x"), "X")

    def test_access_css_class(self):
        probe = Probe()
        self.assertEqual(access_css_class(probe.fields["gain"]), "poser-input")
        self.assertEqual(access_css_class(probe.fields["readout"]), "poser-feedback")
        self.assertEqual(access_css_class(probe.fields["port"]), "poser-init")

    def test_format_time(self):
        self.assertEqual(format_time(0.0), "0:00.0")
        self.assertEqual(format_time(65.3), "1:05.3")

    def test_format_readout(self):
        self.assertEqual(format_readout(True), "on")
        self.assertEqual(format_readout(Mode.B), "B")
        self.assertEqual(format_readout(1.26), "1.3")
        self.assertEqual(format_readout(7), "7")
        self.assertEqual(format_readout(Color(1.0, 0.0, 0.0)), str(Color(1.0, 0.0, 0.0)))
        self.assertEqual(format_readout(Point2f(1.0, 2.0)), str(Point2f(1.0, 2.0)))
        self.assertEqual(format_readout(Rect(0.0, 0.0, 1.0, 1.0)), str(Rect(0.0, 0.0, 1.0, 1.0)))


if __name__ == "__main__":
    unittest.main()
