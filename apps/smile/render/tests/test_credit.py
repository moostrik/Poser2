"""Tests for the artist's credit: the QR code scans, the line wraps evenly, and it fades with the result (no GL)."""

import os
import tempfile
import unittest
from unittest import mock

import cv2

from apps.smile.credit import (LINE, LINES, QUIET_MODULES, TITLE_FONT, URL, CreditLines, qr_image, quiet_px, title_image,
                               wrap_balanced)
from apps.smile.render.layers.credit_layer import FADE_S, SHOWN, caption_shown, fade_step, new_line_due
from apps.smile.render.sequencer import SequenceType


class TestTitle(unittest.TestCase):
    def test_the_title_alone_is_white_letters_with_a_black_edge_on_transparent(self):
        image = title_image('', 107)
        self.assertEqual(image.shape[2], 4)
        self.assertGreater(image.shape[1], image.shape[0])          # SMILE and the ring side by side
        self.assertTrue((image[:, :, 3] == 0).any())
        self.assertEqual(image[:, :, :3].max(), 255)

    def test_the_height_follows_the_font_size(self):
        self.assertAlmostEqual(title_image('', 200).shape[0] / title_image('', 100).shape[0], 2.0, delta=0.05)
        self.assertEqual(round(96 / TITLE_FONT), 107)

    def test_a_name_comes_after_it(self):
        self.assertGreater(title_image('Matthias Oostrik', 28).shape[1], title_image('', 28).shape[1])

    def test_the_outline_is_part_of_the_image(self):
        self.assertGreater(title_image('', 107, outline_ratio=0.05).shape[1], title_image('', 107).shape[1])

    def test_alpha_scales_the_whole_image(self):
        self.assertLessEqual(title_image('', 28, alpha=0.5)[:, :, 3].max(), 127)


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

    def test_quiet_px_is_the_white_border(self):
        image = qr_image(URL, 500)
        quiet = quiet_px(image)
        self.assertEqual(quiet % QUIET_MODULES, 0)
        self.assertTrue((image[:quiet] == 255).all())
        self.assertTrue((image[quiet] < 255).any())


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


class TestLines(unittest.TestCase):
    def write(self, text: str) -> str:
        path = os.path.join(tempfile.mkdtemp(), 'lines.txt')
        with open(path, 'w', encoding='utf-8') as file:
            file.write(text)
        return path

    def test_the_file_in_the_repo_has_lines_that_fit(self):
        lines = CreditLines(LINES).lines
        self.assertGreaterEqual(len(lines), 100)
        self.assertEqual(len(lines), len(set(lines)))
        self.assertIn(LINE, lines)

    def test_skips_blanks_and_comments(self):
        lines = CreditLines(self.write('# a comment\n\nloves your smile!\n  says: cheese!  \n')).lines
        self.assertEqual(lines, ['loves your smile!', 'says: cheese!'])

    def test_never_the_same_line_twice_in_a_row(self):
        credit = CreditLines(self.write('a\nb\nc\n'))
        picks = [credit.pick() for _ in range(50)]
        self.assertEqual(set(picks), {'a', 'b', 'c'})
        self.assertTrue(all(x != y for x, y in zip(picks, picks[1:])))

    def test_a_missing_or_empty_file_gives_the_default_line(self):
        self.assertEqual(CreditLines('/nowhere/lines.txt').pick(), LINE)
        self.assertEqual(CreditLines(self.write('# nothing\n')).pick(), LINE)

    def test_the_file_is_read_again_when_it_changes(self):
        path = self.write('a\n')
        credit = CreditLines(path)
        self.assertEqual(credit.pick(), 'a')
        with open(path, 'w') as file:
            file.write('b\n')
        os.utime(path, (1, 1))     # a changed mtime, however fast the test writes
        self.assertEqual(credit.pick(), 'b')


class TestFade(unittest.TestCase):
    def test_shown_from_the_first_generated_image_to_the_result(self):
        self.assertEqual(SHOWN, {SequenceType.DIFFUSION, SequenceType.DIFF_LOOP, SequenceType.TIMEOUT})

    def test_on_the_camera_only_with_the_switch(self):
        self.assertFalse(caption_shown(SequenceType.CAMERA_SHOW, False))
        self.assertTrue(caption_shown(SequenceType.CAMERA_DETECT, True))
        self.assertTrue(caption_shown(SequenceType.TIMEOUT, False))
        self.assertFalse(caption_shown(SequenceType.FLASH, True))

    def test_a_new_line_at_the_capture_and_when_the_camera_view_begins(self):
        S = SequenceType
        self.assertTrue(new_line_due(S.CAPTURE, S.FLASH))
        self.assertFalse(new_line_due(S.CAPTURE, S.CAPTURE))
        self.assertTrue(new_line_due(S.CAMERA_SHOW, S.FADE))
        self.assertTrue(new_line_due(S.CAMERA_SHOW, S.NONE))
        # a face that comes and goes within the camera view keeps the line
        self.assertFalse(new_line_due(S.CAMERA_SHOW, S.CAMERA_DETECT))
        self.assertFalse(new_line_due(S.CAMERA_DETECT, S.CAMERA_SHOW))

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
