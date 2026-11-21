"""The link: the app's end (PanelServer) and the panel's (AppLink) over a pipe, without a process or NiceGUI."""

import tempfile
import threading
import time
import unittest
from multiprocessing import Pipe
from typing import Callable

from modules.logging import LogRingBuffer
from modules.settings import BaseSettings, Field, Group, SettingsStore, Widget, mirror

from ..process import AppLink
from ..server import PanelServer, PanelSettings


class Probe(BaseSettings):
    rate = Field(30.0)
    fps = Field(0.0, access=Field.READ)
    reset = Field(False, widget=Widget.button)


class Flow(BaseSettings):
    gain = Field(1.0)


class Root(BaseSettings):
    setup = Group(Probe)
    flow = Group(Flow)


def until(condition: Callable[[], bool], timeout: float = 2.0) -> bool:
    end = time.perf_counter() + timeout
    while time.perf_counter() < end:
        if condition():
            return True
        time.sleep(0.01)
    return condition()


class TestLink(unittest.TestCase):
    def setUp(self):
        self.tree = Root()
        self.tree.initialize()
        self.presses: list[bool] = []
        self.tree.setup.bind(Probe.reset, self.presses.append)
        self.store = SettingsStore(self.enterContext(tempfile.TemporaryDirectory()))

        app_end, panel_end = Pipe()
        self.server = PanelServer(self.tree, PanelSettings(), self.store)
        self.server._conn = app_end
        self.thread = threading.Thread(target=self.server._link, args=(app_end,), daemon=True)
        self.thread.start()

        _, description, values, _, _ = panel_end.recv()
        self.copy = mirror.build_mirror(description)
        mirror.apply(self.copy, values)
        self.ended = threading.Event()
        self.link = AppLink(panel_end, LogRingBuffer(), on_loop=lambda call, *args: call(*args), on_end=self.ended.set)
        self.link.watch(self.copy)
        threading.Thread(target=self.link.receive, daemon=True).start()

    def tearDown(self):
        with self.assertLogs("modules.panel.server", level="INFO"):
            self.server.stop()
        self.thread.join(timeout=2.0)
        self.link.close()

    def test_a_mirror_write_reaches_the_tree_and_does_not_come_back(self):
        seen: list[float] = []
        setup = self.copy.children["setup"]
        setup.bind(setup.fields["rate"], seen.append)
        for rate in (1.0, 2.0, 3.0):        # a slider dragged
            setup.rate = rate
        self.assertTrue(until(lambda: self.tree.setup.rate == 3.0))
        time.sleep(0.3)                     # three of the link's sends
        self.assertEqual(seen, [1.0, 2.0, 3.0])

    def test_a_tree_change_reaches_the_mirror(self):
        self.tree.setup.fps = 59.5
        self.assertTrue(until(lambda: self.copy.children["setup"].fps == 59.5))

    def test_a_press_fires_the_action(self):
        setup = self.copy.children["setup"]
        setup.fields["reset"].fire(setup)
        self.assertTrue(until(lambda: self.presses == [True]))

    def test_a_refused_write_goes_back(self):
        setup = self.copy.children["setup"]
        self.link._from_app = True          # the mirror holds a value without sending it
        setup.rate = 99.0
        self.link._from_app = False
        with self.assertLogs("modules.panel.server", level="WARNING"):
            self.link.send(("set", "setup", "rate", "fast"))
            self.assertTrue(until(lambda: setup.rate == 30.0))
        self.assertEqual(self.tree.setup.rate, 30.0)

    def test_each_root_group_is_saved_and_loaded_over_the_link(self):
        self.tree.setup.rate = 10.0
        self.tree.flow.gain = 2.0
        self.assertTrue(self.link.save("setup", "studio"))
        self.assertTrue(self.link.save("flow", "calm"))
        self.assertEqual(self.store.scan("setup"), ["studio"])
        self.assertEqual(self.store.scan("flow"), ["calm"])
        self.tree.setup.rate = 20.0
        self.tree.flow.gain = 3.0
        self.assertTrue(self.link.load("flow", "calm"))
        self.assertEqual((self.tree.setup.rate, self.tree.flow.gain), (20.0, 2.0))     # the setup as it was
        self.assertTrue(self.link.load("setup", "studio"))
        self.assertEqual((self.tree.setup.rate, self.tree.flow.gain), (10.0, 2.0))

    def test_a_bad_request_is_refused(self):
        with self.assertLogs("modules.panel.server", level="WARNING"):
            self.assertFalse(self.link.load("setup", ".hidden"))     # an invalid name: refused without a file
        with self.assertLogs("modules.panel.server", level="WARNING"):
            self.assertFalse(self.link.save("camera", "studio"))      # no root group of that name

    def test_stop_ends_the_panel(self):
        self.server._stopped.set()
        self.server._send(("stop",))
        self.assertTrue(self.ended.wait(2.0))


if __name__ == "__main__":
    unittest.main()
