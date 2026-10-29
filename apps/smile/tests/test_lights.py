"""Tests for the lights' packet and camera ring (no network: the sender is never started)."""

import json
import os
import tempfile
import unittest

import numpy as np

from apps.smile.lights import Lights, CAMERA_LEDS
from apps.smile.render.Sequencer import AnimationType
from modules.inout import ArtNetSenderSettings


def lights_for(modules: dict) -> Lights:
    path = os.path.join(tempfile.mkdtemp(), 'artnet.json')
    with open(path, 'w') as f:
        json.dump(modules, f)
    return Lights(ArtNetSenderSettings(), path)


class TestLights(unittest.TestCase):
    def test_the_modules_take_the_packet_in_order(self):
        lights = lights_for({'pc': {'type': 'Core', 'num_rgb': 3}, 'mako': {'type': 'Camera', 'num_rgb': 24}, 'audio': {'type': 'Core', 'num_rgb': 3}})
        self.assertEqual(lights.cores, {'pc': 0, 'audio': 9 + CAMERA_LEDS * 3})
        self.assertEqual(lights.cameras, {'mako': 9})
        self.assertGreaterEqual(lights.sender.packet_size, 9 + CAMERA_LEDS * 3 + 9 + 1)

    def test_the_ring_is_replaced_in_one_step(self):
        lights = lights_for({'mako': {'type': 'Camera', 'num_rgb': 24}})
        lights.set_progress(1.0, AnimationType.DIFFUSE_OUT)     # the full red ring
        before = lights.cam_lights
        self.assertTrue(np.all(before[:, 0] == 1.0))
        lights.set_progress(0.5, AnimationType.FADE)
        # a new array: the sender thread never sees the one it reads half filled in
        self.assertIsNot(lights.cam_lights, before)
        self.assertTrue(np.all(before[:, 0] == 1.0))

    def test_the_ring_reaches_the_packet(self):
        lights = lights_for({'pc': {'type': 'Core', 'num_rgb': 3}, 'mako': {'type': 'Camera', 'num_rgb': 24}})
        lights.set_camera_white(1.0)
        packet = bytearray(lights.sender.packet_size)
        lights.fill(packet)
        start = lights.cameras['mako']
        self.assertEqual(bytes(packet[start:start + 72]), b'\xff' * 72)


if __name__ == '__main__':
    unittest.main()
