"""Tests for the OSC sender: what goes into a bundle, and a send that fails (a fake client, no network)."""

import time
import unittest

from modules.osc.Sender import Sender
from modules.osc.settings import OscSettings


class FailingClient:
    def __init__(self, failures: int) -> None:
        self.failures: int = failures
        self.sent: int = 0

    def send(self, bundle) -> None:
        if self.failures > 0:
            self.failures -= 1
            raise OSError('network is unreachable')
        self.sent += 1


class TestSender(unittest.TestCase):
    def setUp(self):
        self.settings = OscSettings()
        self.settings.fps = 120.0
        self.sender = Sender(self.settings)

    def tearDown(self):
        self.sender.stop()

    def test_a_state_sends_its_newest_value_and_an_event_every_one(self):
        data = [('/progress', 0.1), ('/bang', 0), ('/progress', 0.2), ('/bang', 2), ('/state', 'CAMERA_IN')]
        self.assertEqual(self.sender.get_last_data(data), [('/progress', 0.2), ('/bang', 0), ('/bang', 2), ('/state', 'CAMERA_IN')])

    def test_nothing_is_queued_while_the_sender_does_not_run(self):
        self.sender.send('/progress', 0.5)
        self.assertTrue(self.sender.queue.empty())

    def test_a_failed_send_does_not_stop_the_sender(self):
        self.sender.start()
        client = FailingClient(failures=1)
        self.sender.client = client
        with self.assertLogs('modules.osc.Sender', 'ERROR'):
            self.sender.send('/progress', 0.1)
            end = time.time() + 2.0
            while client.failures and time.time() < end:
                time.sleep(0.01)
        self.sender.send('/progress', 0.2)
        end = time.time() + 2.0
        while not client.sent and time.time() < end:
            time.sleep(0.01)
        self.assertEqual(client.sent, 1)
        self.assertTrue(self.sender.running)


if __name__ == '__main__':
    unittest.main()
