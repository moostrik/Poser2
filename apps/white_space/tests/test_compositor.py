"""Tests for the Compositor — weighted blend of the composed look, the light_phase shift
for spun-content layers, explicit resets, and the manual/debug override."""

import unittest
from types import SimpleNamespace

import numpy as np

from apps.white_space.light import DebugLayer, LayerId, Tick, BeamLightId
from apps.white_space.light.frame import Frame
from apps.white_space.light.layers.compositor import Compositor

RES = 8


def config(**overrides) -> SimpleNamespace:
    base = dict(light_resolution=RES, light_phase=0.0, debug=DebugLayer.OFF)
    base.update(overrides)
    return SimpleNamespace(**base)


class FakeLayer:
    """Writes a constant into the white channel; records resets."""
    SHIFTED = False   # beam-mode fake; set True on instances standing in for HighLayers

    def __init__(self, value: float) -> None:
        self.value = value
        self.resets = 0

    def render(self, frame: Frame) -> None:
        frame.white += self.value

    def reset(self) -> None:
        self.resets += 1


def frame() -> Frame:
    return Frame(RES, Tick(0.0, 0.0))


class CompositorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.a = FakeLayer(1.0)
        self.b = FakeLayer(2.0)
        self.layers = {LayerId.searchlight: self.a, LayerId.test_pose_waves: self.b}
        self.cfg = config()
        self.comp = Compositor(self.cfg, self.layers)

    def test_weighted_blend(self) -> None:
        self.comp.set_mix([(LayerId.searchlight, 0.5), (LayerId.test_pose_waves, 0.25)])
        f = frame()
        self.comp.render(f)
        np.testing.assert_allclose(f.white, 0.5 * 1.0 + 0.25 * 2.0)

    def test_zero_weight_is_silent_but_never_resets(self) -> None:
        self.comp.set_mix([(LayerId.searchlight, 0.0)])
        f = frame()
        self.comp.render(f)
        np.testing.assert_allclose(f.white, 0.0)
        self.assertEqual(self.a.resets, 0)

    def test_absent_layer_is_not_reset_implicitly(self) -> None:
        self.comp.set_mix([(LayerId.searchlight, 1.0)])
        self.comp.render(frame())
        self.comp.set_mix([(LayerId.test_pose_waves, 1.0)])   # lamp dropped from the look
        self.comp.render(frame())
        self.assertEqual(self.a.resets, 0)                 # resets are explicit only

    def test_per_channel_weights(self) -> None:
        class Both(FakeLayer):
            def render(self, frame: Frame) -> None:
                frame.white += 1.0
                frame.blue += 1.0

        comp = Compositor(config(), {LayerId.test_pose_waves: Both(1.0)})
        comp.set_mix([(LayerId.test_pose_waves, (1.0, 0.25))])   # white hard, blue partial
        f = frame()
        comp.render(f)
        np.testing.assert_allclose(f.white, 1.0)
        np.testing.assert_allclose(f.blue, 0.25)

    def test_explicit_reset_layers(self) -> None:
        self.comp.reset_layers([LayerId.searchlight])
        self.assertEqual(self.a.resets, 1)
        self.assertEqual(self.b.resets, 0)

    def test_light_phase_shifts_only_spun_content(self) -> None:
        class Marker(FakeLayer):
            def render(self, frame: Frame) -> None:
                frame.white[0] += 1.0

        marker = Marker(1.0)
        marker.SHIFTED = True             # stands in for a ProjectionLayer — rides the ring shift
        lamp = Marker(1.0)
        comp = Compositor(config(light_phase=0.25),
                          {LayerId.test_pose_waves: marker, LayerId.searchlight: lamp})
        f = frame()
        comp.set_mix([(LayerId.test_pose_waves, 1.0), (LayerId.searchlight, 1.0)])
        comp.render(f)
        self.assertEqual(f.white[RES // 4], 1.0)   # spun content rolled by a quarter turn
        self.assertEqual(f.white[0], 1.0)          # lamp content not rolled

    def test_debug_override_replaces_state_mix(self) -> None:
        # Selecting a layer IS turning debug on: the select replaces the state's mix solo.
        self.cfg.debug = DebugLayer.test_pose_waves
        self.comp.set_mix([(LayerId.searchlight, 1.0)])
        f = frame()
        self.comp.render(f)
        np.testing.assert_allclose(f.white, 2.0)   # only the debug layer, full weight


class LampMappingTest(unittest.TestCase):
    """BeamLayer's named writes land on the frame's explicit beam lights and touch no pixel —
    verified through playhead_test, beam mode's direct test tool."""

    def test_named_lamps_land_on_the_bar_lights(self) -> None:
        from apps.white_space.light.layers.beam.playhead_test import PlayheadTest, PlayheadTestSettings
        cfg = PlayheadTestSettings()
        cfg.front_white, cfg.back_white = 0.9, 0.6
        cfg.left_blue, cfg.right_blue = 0.4, 0.2
        layer = PlayheadTest(RES, cfg, board=None)
        f = frame()
        layer.render(f)
        self.assertAlmostEqual(f.beam_lights[BeamLightId.FRONT_WHITE], 0.9, places=6)
        self.assertAlmostEqual(f.beam_lights[BeamLightId.BACK_WHITE],  0.6, places=6)
        self.assertAlmostEqual(f.beam_lights[BeamLightId.LEFT_BLUE],   0.4, places=6)
        self.assertAlmostEqual(f.beam_lights[BeamLightId.RIGHT_BLUE],  0.2, places=6)
        self.assertEqual(float(f.light_img.sum()), 0.0)                   # a beam layer writes no pixels


class BarLightMixTest(unittest.TestCase):
    """The Compositor weighs the beam lights per channel (white weight on the whites, blue
    weight on the blues) and never ring-shifts them."""

    class BarLayer:
        SHIFTED = False

        def render(self, f: Frame) -> None:
            f.beam_lights += np.array([1.0, 0.5, 1.0, 0.5], dtype=np.float32)

        def reset(self) -> None:
            pass

    def test_per_channel_weights(self) -> None:
        comp = Compositor(config(), {LayerId.playhead_test: self.BarLayer()})
        f = frame()
        comp.set_mix([(LayerId.playhead_test, (0.5, 0.2))])
        comp.render(f)
        np.testing.assert_allclose(f.beam_lights, [0.5, 0.25, 0.2, 0.1], rtol=1e-6)
        self.assertEqual(float(f.light_img.sum()), 0.0)

    def test_light_phase_leaves_bar_lights_alone(self) -> None:
        comp = Compositor(config(light_phase=0.25), {LayerId.playhead_test: self.BarLayer()})
        f = frame()
        comp.set_mix([(LayerId.playhead_test, 1.0)])
        comp.render(f)
        np.testing.assert_allclose(f.beam_lights, [1.0, 0.5, 1.0, 0.5], rtol=1e-6)

    def test_zero_weight_keeps_bar_lights_dark(self) -> None:
        comp = Compositor(config(), {LayerId.playhead_test: self.BarLayer()})
        f = frame()
        comp.set_mix([(LayerId.playhead_test, 0.0)])
        comp.render(f)
        self.assertEqual(float(f.beam_lights.sum()), 0.0)

    def test_regime_flags(self) -> None:
        from apps.white_space.light import BeamLayer, ProjectionLayer
        self.assertFalse(BeamLayer.SHIFTED)
        self.assertTrue(ProjectionLayer.SHIFTED)


if __name__ == "__main__":
    unittest.main()
