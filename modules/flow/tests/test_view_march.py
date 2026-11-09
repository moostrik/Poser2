"""The composite's terms from the panel's: fully dyed water filling the view covers `opacity` of what is behind
it and is seen at its color times `emission`, whatever the layers."""

import math
import unittest

from modules.flow.fluid_3d.shaders import OPACITY_LIMIT, view_march


def seen(opacity: float, emission: float, layers: int) -> tuple[float, float]:
    """What a march over `layers` of fully dyed water gives: what shows through, and its brightness."""
    extinction, total = view_march(opacity, emission)
    through, brightness = 1.0, 0.0
    for _ in range(layers):
        brightness += through * total / layers
        through *= math.exp(-extinction / layers)
    return through, brightness


class ViewMarchTest(unittest.TestCase):
    def test_the_opacity_is_what_fully_dyed_water_covers(self) -> None:
        for layers in (4, 16, 64):
            through, _ = seen(0.6, 1.0, layers)
            self.assertAlmostEqual(1.0 - through, 0.6, places=6)

    def test_emission_1_shows_fully_dyed_water_near_its_color_at_any_opacity(self) -> None:
        for opacity in (0.0, 0.3, 0.6, 0.9):
            _, brightness = seen(opacity, 1.0, 256)
            self.assertAlmostEqual(brightness, 1.0, delta=0.01)

    def test_opacity_1_is_drawn_at_the_limit_not_infinite(self) -> None:
        extinction, emission = view_march(1.0, 1.0)
        self.assertAlmostEqual(extinction, -math.log(1.0 - OPACITY_LIMIT))
        self.assertTrue(math.isfinite(emission))


if __name__ == '__main__':
    unittest.main()
