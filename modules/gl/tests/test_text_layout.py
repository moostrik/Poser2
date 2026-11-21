import unittest

import numpy as np

from modules.gl.text.font_atlas import GlyphMetrics
from modules.gl.text.text_layout import text_layout

A = GlyphMetrics(u0=0.0, v0=0.0, u1=0.1, v1=0.2, width=8, height=10, bearing_x=1, bearing_y=9, advance=10)
LOW = GlyphMetrics(u0=0.1, v0=0.0, u1=0.2, v1=0.2, width=6, height=7, bearing_x=0, bearing_y=5, advance=8)


def atlas(char: str) -> GlyphMetrics | None:
    return {'A': A, 'a': LOW}.get(char)


class TextLayoutTest(unittest.TestCase):

    def test_quads_in_area_units(self) -> None:
        v = text_layout('A', 20.0, 30.0, 12.0, atlas, 200.0, 100.0)
        self.assertEqual(v.shape, (6, 4))
        x0, y0 = (20 + 1) / 200, (30 + 12 - 9) / 100
        x1, y1 = x0 + 8 / 200, y0 + 10 / 100
        np.testing.assert_allclose(v[0], [x0, y0, 0.0, 0.0], rtol=1e-6)
        np.testing.assert_allclose(v[2], [x1, y1, 0.1, 0.2], rtol=1e-6)
        np.testing.assert_allclose(v[5], [x0, y1, 0.0, 0.2], rtol=1e-6)

    def test_advance_and_baseline(self) -> None:
        v = text_layout('Aa', 0.0, 0.0, 12.0, atlas, 100.0, 100.0)
        self.assertEqual(v.shape, (12, 4))
        self.assertAlmostEqual(float(v[6, 0]), (10 + 0) / 100)       # the second glyph starts one advance on
        self.assertAlmostEqual(float(v[6, 1]), (12 - 5) / 100)       # lower by its smaller bearing

    def test_missing_glyph_skipped(self) -> None:
        self.assertEqual(text_layout('?A', 0.0, 0.0, 12.0, atlas, 100.0, 100.0).shape, (6, 4))
        self.assertEqual(text_layout('??', 0.0, 0.0, 12.0, atlas, 100.0, 100.0).shape, (0, 4))


if __name__ == '__main__':
    unittest.main()
