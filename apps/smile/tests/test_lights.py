"""Tests for the lights' packet and camera ring (no network: the sender is never started)."""

import json
import math
import os
import tempfile
import unittest

import numpy as np

from apps.smile.lights import Lights, CAMERA_LEDS, lfor
from apps.smile.sequence import AnimationType
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

    def test_the_breath_is_a_sine_at_half_its_htz(self):
        for timestamp in (0.0, 1.3, 3.0, 7.5, 12.0, 100.0):
            for phase in (0.0, 0.25):
                sine = 0.5 * math.sin(timestamp * math.pi * 0.1666 + (0.75 + phase) * math.pi * 2) + 0.5
                self.assertAlmostEqual(lfor(timestamp, 0.1666, phase, 0.05, 1), sine * 0.95 + 0.05, places=12)


if __name__ == '__main__':
    unittest.main()
