"""The encoder's bytes are python-osc's for the same messages, so the wire stays as it was."""

import math
import unittest

import numpy as np
from pythonosc.osc_bundle_builder import OscBundleBuilder, IMMEDIATELY
from pythonosc.osc_message_builder import OscMessageBuilder

from modules.inout.osc_encoder import encode_bundle, encode_message
from modules.inout.osc_sender import _encode


def python_osc(address: str, args: list, typed: bool = True) -> bytes:
    """python-osc's message: each argument typed as the sound sent it, or untyped as the sender sent it."""
    builder = OscMessageBuilder(address=address)
    for arg in args:
        if not typed:
            builder.add_arg(arg)
        elif isinstance(arg, int):
            builder.add_arg(arg, OscMessageBuilder.ARG_TYPE_INT)
        elif isinstance(arg, float):
            builder.add_arg(arg, OscMessageBuilder.ARG_TYPE_FLOAT)
        else:
            builder.add_arg(arg, OscMessageBuilder.ARG_TYPE_STRING)
    return builder.build().dgram


class TestEncoder(unittest.TestCase):
    def test_a_float_row_is_its_floats(self):
        row = np.array([0.1, -2.5, np.nan, 3.0e-8, math.pi, -np.nan, np.inf], dtype=np.float32)
        self.assertEqual(encode_message('/pose/0/angle/rad', (row,)), python_osc('/pose/0/angle/rad', row.tolist()))

    def test_a_float64_row_rounds_as_python_osc_does(self):
        row = np.array([1 / 3, math.pi * 0.7, -1e-30, np.nan], dtype=np.float64)
        self.assertEqual(encode_message('/r', (row,)), python_osc('/r', row.tolist()))

    def test_scalars(self):
        for address, args in [('/global/state', [-1]), ('/global/progress', [0.25]), ('/a', [np.nan]),
                              ('/state', ['CAMERA_IN']), ('/abc', ['four']), ('/x', []),
                              ('/pose/12/similarity/leader', [0.0, 1, 'x', -7.5])]:
            with self.subTest(address=address):
                self.assertEqual(encode_message(address, tuple(args)), python_osc(address, args))

    def test_numpy_scalars_are_their_python_values(self):
        self.assertEqual(encode_message('/f', (np.float32(0.3),)), python_osc('/f', [float(np.float32(0.3))]))
        self.assertEqual(encode_message('/i', (np.int64(3),)), python_osc('/i', [3]))

    def test_a_bool_is_refused(self):
        with self.assertRaises(TypeError):
            encode_message('/b', (True,))

    def test_the_sender_s_mixed_messages(self):
        """What smile sends: untyped to python-osc, a tuple as separate arguments."""
        for address, args in [('/test_bang', (5, 1)), ('/progress', 0.5), ('/state', 'CAMERA_IN'), ('/music_id', 3),
                              ('/laughter_offset', -0.125), ('/bang', 0)]:
            with self.subTest(address=address):
                values = list(args) if isinstance(args, tuple) else [args]
                self.assertEqual(_encode(address, args), python_osc(address, values, typed=False))

    def test_a_bundle(self):
        messages = [('/global/state', [4]), ('/pose/0/bbox', [0.5, np.nan, 0.25, 1.0]), ('/state', ['IDLE'])]
        builder = OscBundleBuilder(IMMEDIATELY)
        for address, args in messages:
            builder.add_content(_built(address, args))
        expected: bytes = builder.build().dgram
        self.assertEqual(encode_bundle([encode_message(address, tuple(args)) for address, args in messages]), expected)


def _built(address: str, args: list):
    builder = OscMessageBuilder(address=address)
    for arg in args:
        builder.add_arg(arg, OscMessageBuilder.ARG_TYPE_INT if isinstance(arg, int) else
                        OscMessageBuilder.ARG_TYPE_FLOAT if isinstance(arg, float) else OscMessageBuilder.ARG_TYPE_STRING)
    return builder.build()


if __name__ == '__main__':
    unittest.main()
