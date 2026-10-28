"""DebugLayer is a hand-written mirror of LayerId (so type checkers see a real enum);
this test is what keeps the two from drifting apart."""

import unittest

from apps.white_space.light import DebugLayer, LayerId


class TestDebugLayerMirrorsLayerId(unittest.TestCase):

    def test_off_is_zero_and_first(self):
        self.assertEqual(DebugLayer.OFF, 0)
        self.assertIs(list(DebugLayer)[0], DebugLayer.OFF)
        self.assertNotIn(0, {m.value for m in LayerId})   # OFF never collides with a layer

    def test_members_match_layer_id_in_name_value_and_order(self):
        debug = [(m.name, m.value) for m in DebugLayer if m is not DebugLayer.OFF]
        pool  = [(m.name, m.value) for m in LayerId]
        self.assertEqual(debug, pool)

    def test_round_trip_through_int(self):
        for layer in LayerId:
            self.assertIs(LayerId(int(DebugLayer[layer.name])), layer)


if __name__ == '__main__':
    unittest.main()
