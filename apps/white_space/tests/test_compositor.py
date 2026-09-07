"""Tests for the Compositor — weighted blend of the composed look, the light_phase shift
for spun-content layers, explicit resets, and the manual/debug override."""

import unittest
from types import SimpleNamespace

import numpy as np

from apps.white_space.light import LayerId, Tick
from apps.white_space.light.frame import Frame
from apps.white_space.light.layers.compositor import Compositor

RES = 8


def config(**overrides) -> SimpleNamespace:
    base = dict(light_resolution=RES, light_phase=0.0, manual=False, manual_layers=[])
    base.update(overrides)
    return SimpleNamespace(**base)


class FakeLayer:
    """Writes a constant into the white channel; records resets."""
    def __init__(self, value: float) -> None:
        self.value = value
        self.resets = 0

    def render(self, frame: Frame) -> None:
        frame.white += self.value

    def reset(self) -> None:
        self.resets += 1


def frame() -> Frame:
    return Frame(RES, Tick(0.0, 0.0, 0.0, 0.0, 0))


class CompositorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.a = FakeLayer(1.0)
        self.b = FakeLayer(2.0)
        self.layers = {LayerId.playhead_lamp: self.a, LayerId.pose_waves: self.b}
        self.cfg = config()
        self.comp = Compositor(self.cfg, self.layers, shifted={LayerId.pose_waves})

    def test_weighted_blend(self) -> None:
        self.comp.set_look([(LayerId.playhead_lamp, 0.5), (LayerId.pose_waves, 0.25)])
        f = frame()
        self.comp.render(f)
        np.testing.assert_allclose(f.white, 0.5 * 1.0 + 0.25 * 2.0)

    def test_zero_weight_is_silent_but_never_resets(self) -> None:
        self.comp.set_look([(LayerId.playhead_lamp, 0.0)])
        f = frame()
        self.comp.render(f)
        np.testing.assert_allclose(f.white, 0.0)
        self.assertEqual(self.a.resets, 0)

    def test_absent_layer_is_not_reset_implicitly(self) -> None:
        self.comp.set_look([(LayerId.playhead_lamp, 1.0)])
        self.comp.render(frame())
        self.comp.set_look([(LayerId.pose_waves, 1.0)])   # lamp dropped from the look
        self.comp.render(frame())
        self.assertEqual(self.a.resets, 0)                 # resets are explicit only

    def test_explicit_reset_layers(self) -> None:
        self.comp.reset_layers([LayerId.playhead_lamp])
        self.assertEqual(self.a.resets, 1)
        self.assertEqual(self.b.resets, 0)

    def test_light_phase_shifts_only_spun_content(self) -> None:
        class Marker(FakeLayer):
            def render(self, frame: Frame) -> None:
                frame.white[0] += 1.0

        marker = Marker(1.0)
        lamp = Marker(1.0)
        comp = Compositor(config(light_phase=0.25),
                          {LayerId.pose_waves: marker, LayerId.playhead_lamp: lamp},
                          shifted={LayerId.pose_waves})
        f = frame()
        comp.set_look([(LayerId.pose_waves, 1.0), (LayerId.playhead_lamp, 1.0)])
        comp.render(f)
        self.assertEqual(f.white[RES // 4], 1.0)   # spun content rolled by a quarter turn
        self.assertEqual(f.white[0], 1.0)          # lamp content not rolled

    def test_manual_override_replaces_state_look(self) -> None:
        self.cfg.manual = True
        self.cfg.manual_layers = [LayerId.pose_waves]
        self.comp.set_look([(LayerId.playhead_lamp, 1.0)])
        f = frame()
        self.comp.render(f)
        np.testing.assert_allclose(f.white, 2.0)   # only the manual layer, full weight


if __name__ == "__main__":
    unittest.main()
