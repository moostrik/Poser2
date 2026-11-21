"""Color, Point2f and Rect as field values: their own behavior, serialization and settings."""

import json
import unittest

from modules.settings import Field, BaseSettings
from modules.utils import Color, Point2f, Rect


class _ColorSettings(BaseSettings):
    color = Field(Color(1.0, 0.0, 0.0))


class _PointSettings(BaseSettings):
    pos = Field(Point2f(0.0, 0.0))


class _RectSettings(BaseSettings):
    region = Field(Rect(0.0, 0.0, 1.0, 1.0))


class TestColorClass(unittest.TestCase):
    """Tests for the Color utility class from modules.utils.color."""

    # -- Construction --------------------------------------------------------

    def test_rgba_defaults(self):
        c = Color(0.5, 0.2, 0.1)
        self.assertAlmostEqual(c.r, 0.5)
        self.assertAlmostEqual(c.g, 0.2)
        self.assertAlmostEqual(c.b, 0.1)
        self.assertAlmostEqual(c.a, 1.0)

    def test_rgba_explicit_alpha(self):
        c = Color(0.5, 0.2, 0.1, 0.8)
        self.assertAlmostEqual(c.a, 0.8)

    # -- from_int / to_int ---------------------------------------------------

    def test_from_int(self):
        c = Color.from_int(128, 64, 32)
        self.assertAlmostEqual(c.r, 128 / 255, places=3)
        self.assertAlmostEqual(c.g, 64 / 255, places=3)
        self.assertAlmostEqual(c.b, 32 / 255, places=3)
        self.assertAlmostEqual(c.a, 1.0)

    def test_from_int_with_alpha(self):
        c = Color.from_int(255, 0, 0, 128)
        self.assertAlmostEqual(c.a, 128 / 255, places=3)

    def test_to_int_round_trip(self):
        c = Color.from_int(200, 100, 50, 255)
        ri, gi, bi, ai = c.to_int()
        self.assertEqual((ri, gi, bi, ai), (200, 100, 50, 255))

    # -- from_hex / to_hex ---------------------------------------------------

    def test_from_hex_6(self):
        c = Color.from_hex("#FF8040")
        self.assertAlmostEqual(c.r, 1.0, places=2)
        self.assertAlmostEqual(c.g, 128 / 255, places=2)
        self.assertAlmostEqual(c.b, 64 / 255, places=2)
        self.assertAlmostEqual(c.a, 1.0)

    def test_from_hex_8(self):
        c = Color.from_hex("#FF804080")
        self.assertAlmostEqual(c.a, 128 / 255, places=2)

    def test_from_hex_3(self):
        c = Color.from_hex("#F84")
        self.assertAlmostEqual(c.r, 0xFF / 255, places=2)
        self.assertAlmostEqual(c.g, 0x88 / 255, places=2)
        self.assertAlmostEqual(c.b, 0x44 / 255, places=2)

    def test_from_hex_4(self):
        c = Color.from_hex("#F84A")
        self.assertAlmostEqual(c.r, 0xFF / 255, places=2)
        self.assertAlmostEqual(c.a, 0xAA / 255, places=2)

    def test_from_hex_no_hash(self):
        c = Color.from_hex("FF0000")
        self.assertAlmostEqual(c.r, 1.0)

    def test_to_hex(self):
        c = Color(1.0, 0.0, 0.0)
        self.assertEqual(c.to_hex(), "#FF0000")

    def test_to_hex_with_alpha(self):
        c = Color(1.0, 0.0, 0.0, 0.5)
        h = c.to_hex(include_alpha=True)
        self.assertTrue(h.startswith("#FF0000"))
        self.assertEqual(len(h), 9)

    def test_hex_round_trip(self):
        original = Color(0.8, 0.4, 0.2)
        rebuilt = Color.from_hex(original.to_hex())
        # Hex is 8-bit so we lose some precision
        for ch in ("r", "g", "b"):
            self.assertAlmostEqual(getattr(original, ch), getattr(rebuilt, ch), places=2)

    def test_from_hex_invalid(self):
        with self.assertRaises(ValueError):
            Color.from_hex("#ZZZZZZ")

    def test_from_hex_bad_length(self):
        with self.assertRaises(ValueError):
            Color.from_hex("#12345")

    # -- from_hsv / to_hsv ---------------------------------------------------

    def test_hsv_round_trip(self):
        c = Color(0.8, 0.3, 0.1)
        h, s, v, a = c.to_hsv()
        rebuilt = Color.from_hsv(h, s, v, a)
        self.assertAlmostEqual(c.r, rebuilt.r, places=5)
        self.assertAlmostEqual(c.g, rebuilt.g, places=5)
        self.assertAlmostEqual(c.b, rebuilt.b, places=5)

    def test_from_hsv_red(self):
        c = Color.from_hsv(0.0, 1.0, 1.0)
        self.assertAlmostEqual(c.r, 1.0)
        self.assertAlmostEqual(c.g, 0.0)
        self.assertAlmostEqual(c.b, 0.0)

    # -- from_hsl / to_hsl ---------------------------------------------------

    def test_hsl_round_trip(self):
        c = Color(0.6, 0.2, 0.9)
        h, s, l, a = c.to_hsl()
        rebuilt = Color.from_hsl(h, s, l, a)
        self.assertAlmostEqual(c.r, rebuilt.r, places=5)
        self.assertAlmostEqual(c.g, rebuilt.g, places=5)
        self.assertAlmostEqual(c.b, rebuilt.b, places=5)

    # -- Serialization -------------------------------------------------------

    def test_to_dict(self):
        c = Color(0.1, 0.2, 0.3, 0.4)
        self.assertEqual(c.to_dict(), {"r": 0.1, "g": 0.2, "b": 0.3, "a": 0.4})

    def test_from_dict(self):
        c = Color.from_dict({"r": 0.5, "g": 0.6, "b": 0.7, "a": 0.8})
        self.assertEqual(c, Color(0.5, 0.6, 0.7, 0.8))

    def test_from_dict_no_alpha(self):
        c = Color.from_dict({"r": 0.1, "g": 0.2, "b": 0.3})
        self.assertEqual(c, Color(0.1, 0.2, 0.3))
        self.assertAlmostEqual(c.a, 1.0)

    def test_to_tuple(self):
        c = Color(0.1, 0.2, 0.3, 0.4)
        self.assertEqual(c.to_tuple(), (0.1, 0.2, 0.3, 0.4))

    def test_from_tuple_3(self):
        c = Color.from_tuple((0.1, 0.2, 0.3))
        self.assertEqual(c, Color(0.1, 0.2, 0.3))

    def test_from_tuple_4(self):
        c = Color.from_tuple((0.1, 0.2, 0.3, 0.4))
        self.assertEqual(c, Color(0.1, 0.2, 0.3, 0.4))

    # -- Protocol methods ----------------------------------------------------

    def test_iter(self):
        c = Color(0.1, 0.2, 0.3, 0.4)
        self.assertEqual(list(c), [0.1, 0.2, 0.3, 0.4])

    def test_getitem(self):
        c = Color(0.1, 0.2, 0.3, 0.4)
        self.assertAlmostEqual(c[0], 0.1)
        self.assertAlmostEqual(c[1], 0.2)
        self.assertAlmostEqual(c[2], 0.3)
        self.assertAlmostEqual(c[3], 0.4)

    def test_getitem_out_of_range(self):
        c = Color(0.1, 0.2, 0.3)
        with self.assertRaises(IndexError):
            _ = c[4]

    def test_len(self):
        self.assertEqual(len(Color(0, 0, 0)), 4)

    def test_equality(self):
        self.assertEqual(Color(0.1, 0.2, 0.3), Color(0.1, 0.2, 0.3))
        self.assertNotEqual(Color(0.1, 0.2, 0.3), Color(0.1, 0.2, 0.4))
        self.assertNotEqual(Color(0.1, 0.2, 0.3, 0.5), Color(0.1, 0.2, 0.3, 0.6))

    def test_eq_not_implemented_for_other_types(self):
        c = Color(0, 0, 0)
        self.assertIs(c.__eq__("not a color"), NotImplemented)

    def test_repr_no_alpha(self):
        self.assertEqual(repr(Color(0.1, 0.2, 0.3)), "Color(0.1, 0.2, 0.3)")

    def test_repr_with_alpha(self):
        self.assertIn("0.5", repr(Color(0.1, 0.2, 0.3, 0.5)))

    def test_copy(self):
        c = Color(0.1, 0.2, 0.3, 0.4)
        c2 = c.copy()
        self.assertEqual(c, c2)
        self.assertIsNot(c, c2)

    # -- Operations ----------------------------------------------------------

    def test_lerp(self):
        a = Color(0.0, 0.0, 0.0, 0.0)
        b = Color(1.0, 1.0, 1.0, 1.0)
        mid = a.lerp(b, 0.5)
        for ch in (mid.r, mid.g, mid.b, mid.a):
            self.assertAlmostEqual(ch, 0.5)

    def test_clamped(self):
        c = Color(-0.1, 1.5, 0.5, 2.0)
        cl = c.clamped()
        self.assertAlmostEqual(cl.r, 0.0)
        self.assertAlmostEqual(cl.g, 1.0)
        self.assertAlmostEqual(cl.b, 0.5)
        self.assertAlmostEqual(cl.a, 1.0)

    def test_with_alpha(self):
        c = Color(0.1, 0.2, 0.3)
        c2 = c.with_alpha(0.5)
        self.assertAlmostEqual(c2.a, 0.5)
        self.assertAlmostEqual(c2.r, 0.1)

    def test_luminance(self):
        white = Color(1.0, 1.0, 1.0)
        self.assertAlmostEqual(white.luminance, 1.0, places=3)
        black = Color(0.0, 0.0, 0.0)
        self.assertAlmostEqual(black.luminance, 0.0)

    def test_rgb_property(self):
        c = Color(0.1, 0.2, 0.3, 0.4)
        self.assertEqual(c.rgb, (0.1, 0.2, 0.3))


class TestPoint2fSerialization(unittest.TestCase):
    """Tests for Point2f to_dict / from_dict and Setting integration."""

    def test_to_dict(self):
        p = Point2f(1.5, 2.5)
        self.assertEqual(p.to_dict(), {"x": 1.5, "y": 2.5})

    def test_from_dict(self):
        p = Point2f.from_dict({"x": 3.0, "y": 4.0})
        self.assertEqual(p, Point2f(3.0, 4.0))

    def test_round_trip(self):
        p = Point2f(1.23, 4.56)
        self.assertEqual(Point2f.from_dict(p.to_dict()), p)


class TestRectSerialization(unittest.TestCase):
    """Tests for Rect to_dict / from_dict and Setting integration."""

    def test_to_dict(self):
        r = Rect(1.0, 2.0, 3.0, 4.0)
        self.assertEqual(r.to_dict(), {"x": 1.0, "y": 2.0, "width": 3.0, "height": 4.0})

    def test_from_dict(self):
        r = Rect.from_dict({"x": 10.0, "y": 20.0, "width": 30.0, "height": 40.0})
        self.assertEqual(r, Rect(10.0, 20.0, 30.0, 40.0))

    def test_round_trip(self):
        r = Rect(1.5, 2.5, 100.0, 200.0)
        self.assertEqual(Rect.from_dict(r.to_dict()), r)


class TestColorSettingIntegration(unittest.TestCase):
    """Tests for Color as a Setting type."""

    def test_default(self):
        s = _ColorSettings()
        self.assertEqual(s.color, Color(1.0, 0.0, 0.0))

    def test_direct_set(self):
        s = _ColorSettings()
        s.color = Color(0.0, 1.0, 0.0, 0.5)
        self.assertEqual(s.color, Color(0.0, 1.0, 0.0, 0.5))

    def test_tuple_coercion_3(self):
        s = _ColorSettings()
        s.color = (0.5, 0.5, 0.5)  # type: ignore
        self.assertEqual(s.color, Color(0.5, 0.5, 0.5))

    def test_tuple_coercion_4(self):
        s = _ColorSettings()
        s.color = (0.1, 0.2, 0.3, 0.4)  # type: ignore
        self.assertEqual(s.color, Color(0.1, 0.2, 0.3, 0.4))

    def test_list_coercion(self):
        s = _ColorSettings()
        s.color = [0.2, 0.3, 0.4]  # type: ignore
        self.assertEqual(s.color, Color(0.2, 0.3, 0.4))

    def test_dict_coercion(self):
        s = _ColorSettings()
        s.color = {"r": 0.1, "g": 0.2, "b": 0.3, "a": 0.9}  # type: ignore
        self.assertEqual(s.color, Color(0.1, 0.2, 0.3, 0.9))

    def test_dict_coercion_no_alpha(self):
        s = _ColorSettings()
        s.color = {"r": 0.5, "g": 0.6, "b": 0.7}  # type: ignore
        self.assertEqual(s.color, Color(0.5, 0.6, 0.7))

    def test_to_dict(self):
        s = _ColorSettings()
        d = s.to_dict()
        self.assertEqual(d["color"], {"r": 1.0, "g": 0.0, "b": 0.0, "a": 1.0})

    def test_json_round_trip(self):
        s1 = _ColorSettings()
        s1.color = Color(0.1, 0.2, 0.3, 0.4)
        data = json.loads(json.dumps(s1.to_dict()))
        s2 = _ColorSettings()
        s2.update_from_dict(data)
        self.assertEqual(s2.color, Color(0.1, 0.2, 0.3, 0.4))

    def test_callback(self):
        s = _ColorSettings()
        results = []
        s.bind(_ColorSettings.color, lambda v: results.append(v))
        s.color = Color(0, 0, 1)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0], Color(0, 0, 1))


class TestPoint2fSettingIntegration(unittest.TestCase):
    """Tests for Point2f as a Setting type."""

    def test_default(self):
        s = _PointSettings()
        self.assertEqual(s.pos, Point2f(0.0, 0.0))

    def test_direct_set(self):
        s = _PointSettings()
        s.pos = Point2f(5.0, 10.0)
        self.assertEqual(s.pos, Point2f(5.0, 10.0))

    def test_tuple_coercion(self):
        s = _PointSettings()
        s.pos = (3.0, 4.0)  # type: ignore
        self.assertEqual(s.pos, Point2f(3.0, 4.0))

    def test_list_coercion(self):
        s = _PointSettings()
        s.pos = [1.0, 2.0]  # type: ignore
        self.assertEqual(s.pos, Point2f(1.0, 2.0))

    def test_dict_coercion(self):
        s = _PointSettings()
        s.pos = {"x": 7.0, "y": 8.0}  # type: ignore
        self.assertEqual(s.pos, Point2f(7.0, 8.0))

    def test_to_dict(self):
        s = _PointSettings()
        s.pos = Point2f(1.5, 2.5)
        self.assertEqual(s.to_dict()["pos"], {"x": 1.5, "y": 2.5})

    def test_json_round_trip(self):
        s1 = _PointSettings()
        s1.pos = Point2f(99.0, 88.0)
        data = json.loads(json.dumps(s1.to_dict()))
        s2 = _PointSettings()
        s2.update_from_dict(data)
        self.assertEqual(s2.pos, Point2f(99.0, 88.0))


class TestRectSettingIntegration(unittest.TestCase):
    """Tests for Rect as a Setting type."""

    def test_default(self):
        s = _RectSettings()
        self.assertEqual(s.region, Rect(0.0, 0.0, 1.0, 1.0))

    def test_direct_set(self):
        s = _RectSettings()
        s.region = Rect(10, 20, 100, 200)
        self.assertEqual(s.region, Rect(10, 20, 100, 200))

    def test_tuple_coercion(self):
        s = _RectSettings()
        s.region = (5, 10, 50, 100)  # type: ignore
        self.assertEqual(s.region, Rect(5, 10, 50, 100))

    def test_list_coercion(self):
        s = _RectSettings()
        s.region = [1, 2, 3, 4]  # type: ignore
        self.assertEqual(s.region, Rect(1, 2, 3, 4))

    def test_dict_coercion(self):
        s = _RectSettings()
        s.region = {"x": 0, "y": 0, "width": 640, "height": 480}  # type: ignore
        self.assertEqual(s.region, Rect(0, 0, 640, 480))

    def test_to_dict(self):
        s = _RectSettings()
        self.assertEqual(
            s.to_dict()["region"],
            {"x": 0.0, "y": 0.0, "width": 1.0, "height": 1.0}
        )

    def test_json_round_trip(self):
        s1 = _RectSettings()
        s1.region = Rect(10, 20, 320, 240)
        data = json.loads(json.dumps(s1.to_dict()))
        s2 = _RectSettings()
        s2.update_from_dict(data)
        self.assertEqual(s2.region, Rect(10, 20, 320, 240))


if __name__ == "__main__":
    unittest.main()
