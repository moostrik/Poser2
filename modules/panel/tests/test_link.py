"""The link: the app's end (PanelServer) and the panel's (AppLink) over a pipe, without a process or NiceGUI."""

import threading
import time
import unittest
from multiprocessing import Pipe
from typing import Callable

from modules.log_config import LogRingBuffer
from modules.settings import BaseSettings, Field, Widget, mirror

from ..process import AppLink
from ..server import PanelServer, PanelSettings


class Probe(BaseSettings):
    rate = Field(30.0)
    fps = Field(0.0, access=Field.READ)
    reset = Field(False, widget=Widget.button)


def until(condition: Callable[[], bool], timeout: float = 2.0) -> bool:
    end = time.perf_counter() + timeout
    while time.perf_counter() < end:
        if condition():
            return True
        time.sleep(0.01)
    return condition()


class TestLink(unittest.TestCase):
    def setUp(self):
        self.tree = Probe()
        self.tree.initialize()
        self.presses: list[bool] = []
        self.tree.bind(Probe.reset, self.presses.append)

        app_end, panel_end = Pipe()
        self.server = PanelServer(self.tree, PanelSettings())
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
        self.copy.bind(self.copy.fields["rate"], seen.append)
        for rate in (1.0, 2.0, 3.0):        # a slider dragged
            self.copy.rate = rate
        self.assertTrue(until(lambda: self.tree.rate == 3.0))
        time.sleep(0.3)                     # three of the link's sends
        self.assertEqual(seen, [1.0, 2.0, 3.0])

    def test_a_tree_change_reaches_the_mirror(self):
        self.tree.fps = 59.5
        self.assertTrue(until(lambda: self.copy.fps == 59.5))

    def test_a_press_fires_the_action(self):
        self.copy.fields["reset"].fire(self.copy)
        self.assertTrue(until(lambda: self.presses == [True]))

    def test_a_refused_write_goes_back(self):
        self.link._from_app = True          # the mirror holds a value without sending it
        self.copy.rate = 99.0
        self.link._from_app = False
        with self.assertLogs("modules.panel.server", level="WARNING"):
            self.link.send(("set", "", "rate", "fast"))
            self.assertTrue(until(lambda: self.copy.rate == 30.0))
        self.assertEqual(self.tree.rate, 30.0)

    def test_a_preset_request_is_answered(self):
        with self.assertLogs("modules.panel.server", level="WARNING"):
            self.assertFalse(self.link.load(".hidden"))     # an invalid name: refused without a file

    def test_stop_ends_the_panel(self):
        self.server._stopped.set()
        self.server._send(("stop",))
        self.assertTrue(self.ended.wait(2.0))


if __name__ == "__main__":
    unittest.main()
