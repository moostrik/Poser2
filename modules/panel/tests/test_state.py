"""Tests of the panel state: what it remembers, its keys and its file."""

import json
import tempfile
import unittest
from pathlib import Path

from modules.panel.state import ACTIVE_TAB_KEY, STATE_FILE, PanelState


class TestKeys(unittest.TestCase):

    def test_card_key_joins_the_path(self):
        self.assertEqual(PanelState.card_key("tab", "card"), "tab.card")
        self.assertEqual(PanelState.card_key("tab.card", "inner"), "tab.card.inner")

    def test_card_key_at_the_top(self):
        self.assertEqual(PanelState.card_key("", "tab"), "tab")

    def test_toggle_key(self):
        self.assertEqual(PanelState.toggle_key("tab", "__hide_init__"), "tab.__hide_init__")


class TestDefaults(unittest.TestCase):

    def setUp(self):
        self.state = PanelState(Path(tempfile.mkdtemp()) / STATE_FILE)
        self.state.load()

    def test_a_missing_file_is_an_empty_state(self):
        self.assertFalse(self.state.card_open("tab.card"))
        self.assertEqual(self.state.active_tab(["a", "b"]), "a")
        self.assertTrue(self.state.hidden("a", "__hide_init__", True))
        self.assertFalse(self.state.hidden("a", "__hide_input__", False))

    def test_a_card_is_open_only_when_stored_true(self):
        self.state.set_card_open("tab.card", True)
        self.assertTrue(self.state.card_open("tab.card"))
        self.state.set_card_open("tab.card", False)
        self.assertFalse(self.state.card_open("tab.card"))

    def test_the_active_tab_must_exist(self):
        self.state.set_active_tab("gone")
        self.assertEqual(self.state.active_tab(["a", "b"]), "a")
        self.state.set_active_tab("b")
        self.assertEqual(self.state.active_tab(["a", "b"]), "b")

    def test_a_toggle_is_per_tab(self):
        self.state.set_hidden("a", "__hide_init__", False)
        self.assertFalse(self.state.hidden("a", "__hide_init__", True))
        self.assertTrue(self.state.hidden("b", "__hide_init__", True))


class TestFile(unittest.TestCase):

    def setUp(self):
        self.path = Path(tempfile.mkdtemp()) / STATE_FILE

    def test_round_trip_keeps_the_format(self):
        state = PanelState(self.path)
        state.load()
        state.set_card_open("tab.card", True)
        state.set_active_tab("tab")
        state.set_hidden("tab", "__hide_init__", False)
        self.assertEqual(json.loads(self.path.read_text()),
                         {"tab.card": True, ACTIVE_TAB_KEY: "tab", "tab.__hide_init__": False})
        again = PanelState(self.path)
        again.load()
        self.assertTrue(again.card_open("tab.card"))
        self.assertEqual(again.active_tab(["other", "tab"]), "tab")
        self.assertFalse(again.hidden("tab", "__hide_init__", True))

    def test_reads_the_old_file(self):
        self.path.write_text('{"tab.every": true, "tab.inner": false, "__active_tab__": "tab", "tab.__hide_feedback__": true}')
        state = PanelState(self.path)
        state.load()
        self.assertTrue(state.card_open("tab.every"))
        self.assertFalse(state.card_open("tab.inner"))
        self.assertEqual(state.active_tab(["tab"]), "tab")
        self.assertTrue(state.hidden("tab", "__hide_feedback__", False))

    def test_an_unreadable_file_is_a_warning_and_an_empty_state(self):
        self.path.write_text("{not json")
        state = PanelState(self.path)
        with self.assertLogs("modules.panel.state", level="WARNING"):
            state.load()
        self.assertEqual(state.active_tab(["a"]), "a")

    def test_a_file_without_a_dictionary_is_a_warning_and_an_empty_state(self):
        self.path.write_text("[1, 2]")
        state = PanelState(self.path)
        with self.assertLogs("modules.panel.state", level="WARNING"):
            state.load()
        self.assertFalse(state.card_open("x"))

    def test_a_failed_save_is_a_warning(self):
        state = PanelState(self.path.parent / "a_file" / STATE_FILE)
        (self.path.parent / "a_file").write_text("")   # a file where the folder should be
        with self.assertLogs("modules.panel.state", level="WARNING"):
            state.set_card_open("x", True)
        self.assertTrue(state.card_open("x"))
