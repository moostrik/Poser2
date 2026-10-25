"""Tests for the artist's credit: the QR code scans, the line wraps evenly, and it fades with the result (no GL)."""

import unittest

import cv2

from modules.credit import LINE, URL, qr_image, shown_url, wrap_balanced
from modules.render.layers.CreditLayer import FADE_S, SHOWN, fade_step
from modules.render.Sequencer import SequenceType


class TestQr(unittest.TestCase):
    def test_the_code_scans_back_to_the_url(self):
        image = qr_image(URL, 240)
        self.assertEqual(cv2.QRCodeDetector().detectAndDecode(image)[0], URL)

    def test_whole_pixels_per_module_within_the_size(self):
        image = qr_image(URL, 240)
        self.assertEqual(image.shape[0], image.shape[1])
        self.assertLessEqual(image.shape[0], 240)
        self.assertGreater(image.shape[0], 240 * 0.8)
        self.assertEqual(image.shape[2], 3)
        self.assertTrue((image[:8] == 255).all())       # the quiet zone is white

    def test_shown_url_drops_the_scheme(self):
        self.assertEqual(shown_url("https://oostrik.net/smile/"), "oostrik.net/smile")
        self.assertEqual(shown_url("oostrik.net"), "oostrik.net")


class TestWrap(unittest.TestCase):
    measure = staticmethod(lambda line: 10 * len(line))      # a monospaced font, 10 px a character

    def test_one_line_when_it_fits(self):
        self.assertEqual(wrap_balanced(LINE, self.measure, 1000), [LINE])

    def test_two_even_lines(self):
        self.assertEqual(wrap_balanced("Matthias Oostrik likes your smile!", self.measure, 250),
                         ["Matthias Oostrik", "likes your smile!"])

    def test_a_long_word_keeps_a_line_of_its_own(self):
        self.assertEqual(wrap_balanced("a supercalifragilistic b", self.measure, 50), ["a", "supercalifragilistic", "b"])

    def test_empty(self):
        self.assertEqual(wrap_balanced("  ", self.measure, 100), [])


class TestFade(unittest.TestCase):
    def test_shown_with_the_grin_and_the_result(self):
        self.assertEqual(SHOWN, {SequenceType.DIFF_LOOP, SequenceType.TIMEOUT})

    def test_fades_in_and_out(self):
        alpha = 0.0
        alpha = fade_step(alpha, True, FADE_S / 2)
        self.assertAlmostEqual(alpha, 0.5)
        alpha = fade_step(alpha, True, FADE_S)
        self.assertEqual(alpha, 1.0)
        alpha = fade_step(alpha, False, FADE_S * 2)
        self.assertEqual(alpha, 0.0)


if __name__ == '__main__':
    unittest.main()
