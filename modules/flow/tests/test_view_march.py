"""The composite's terms from the panel's: `opacity` the widths of fully dyed water that hide half of what is
behind it, `fog` the widths of water that dim light to half, toward `tint`'s hue, and thick density seen at its
color times `emission`, whatever the layers and the box's depth."""

import math
import unittest

from modules.flow.fluid_3d.shaders import LUMINANCE, fog_extinction, march_look


def through_slab(absorption: float, emission: float, thickness: float, depth: float, layers: int) -> tuple[float, float]:
    """The composite's march over `layers` of a box `depth` widths deep whose first `thickness` widths are fully
    dyed, white: what shows through, and its brightness."""
    through, brightness = 1.0, 0.0
    sigma = absorption / layers
    for k in range(layers):
        amount = min(max(thickness / depth * layers - k, 0.0), 1.0)    # how much of this layer is dyed
        taken = sigma * amount
        sent = (1.0 - math.exp(-taken)) / amount if amount > 1e-6 else sigma
        brightness += through * emission * amount * sent
        through *= math.exp(-taken)
    return through, brightness


class MarchLookTest(unittest.TestCase):
    def test_opacity_is_the_thickness_that_hides_half(self) -> None:
        for depth in (0.2, 0.5, 1.0):
            look = march_look(1.0, 0.05, 1000.0, (1.0, 1.0, 1.0), depth)
            through, _ = through_slab(look.absorption, look.emission, 0.05, depth, 64)
            self.assertAlmostEqual(through, 0.5, places=6)

    def test_thick_density_is_seen_at_its_color_times_the_emission(self) -> None:
        for emission in (0.5, 1.0, 3.0):
            look = march_look(emission, 0.01, 1000.0, (1.0, 1.0, 1.0), 0.5)
            _, brightness = through_slab(look.absorption, look.emission, 0.5, 0.5, 256)
            self.assertAlmostEqual(brightness, emission, delta=0.01 * emission)

    def test_fog_is_the_distance_that_dims_light_to_half(self) -> None:
        white = fog_extinction(2.0, (1.0, 1.0, 1.0))
        for sigma in white:
            self.assertAlmostEqual(math.exp(-sigma * 2.0), 0.5)
        sea = (0.3, 0.8, 1.0)
        kept = [math.exp(-sigma * 2.0) for sigma in fog_extinction(2.0, sea)]
        self.assertAlmostEqual(sum(w * k for w, k in zip(LUMINANCE, kept)), 0.5)      # half as bright
        self.assertLess(kept[0], kept[1])                                               # red taken out first
        self.assertLess(kept[1], kept[2])


if __name__ == '__main__':
    unittest.main()
