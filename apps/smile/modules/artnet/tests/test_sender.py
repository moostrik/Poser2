"""Tests for the Art-Net sender's packet and ring (no network: the sender is never started)."""

import json
import os
import tempfile
import unittest

import numpy as np

from modules.artnet.Sender import Sender
from modules.artnet.settings import ArtnetSettings
from modules.render.Sequencer import AnimationType


def sender_for(modules: dict) -> Sender:
    path = os.path.join(tempfile.mkdtemp(), 'artnet.json')
    with open(path, 'w') as f:
        json.dump(modules, f)
    return Sender(ArtnetSettings(), path)


class TestSender(unittest.TestCase):
    def test_the_packet_size_is_even(self):
        # StupidArtnet rounds its own size up to even and refuses a packet of another length
        cores = {f'core{i}': {'type': 'Core', 'num_rgb': 3} for i in range(4)}
        for modules in (cores, {**cores, 'mako': {'type': 'Camera', 'num_rgb': 24}}):
            sender = sender_for(modules)
            self.assertEqual(sender.packet_size % 2, 0)
            self.assertEqual(len(sender.packet), sender.packet_size)

    def test_the_ring_is_replaced_in_one_step(self):
        sender = sender_for({'mako': {'type': 'Camera', 'num_rgb': 24}})
        sender.set_progress(1.0, AnimationType.DIFFUSE_OUT)     # the full red ring
        before = sender.cam_lights
        self.assertTrue(np.all(before[:, 0] == 1.0))
        sender.set_progress(0.5, AnimationType.DESTRUCT_OUT)
        # a new array: the sender thread never sees the one it reads half filled in
        self.assertIsNot(sender.cam_lights, before)
        self.assertTrue(np.all(before[:, 0] == 1.0))

    def test_the_ring_reaches_the_packet(self):
        sender = sender_for({'pc': {'type': 'Core', 'num_rgb': 3}, 'mako': {'type': 'Camera', 'num_rgb': 24}})
        sender.set_camera_white(1.0)
        sender.update_cameras()
        start = sender.cameras['mako']
        self.assertEqual(bytes(sender.packet[start:start + 72]), b'\xff' * 72)


if __name__ == '__main__':
    unittest.main()
