"""Tests for the sphere's wiring: which screen a player's cell is, the rows and the angles."""

import unittest
from pathlib import Path

from apps.smile import sphere


class TestWiring(unittest.TestCase):
    def test_every_screen_is_one_cell_and_back(self):
        seen = set()
        for screen in range(1, sphere.SCREENS + 1):
            player, cell = sphere.cell_of_screen(screen)
            self.assertIn(player, (0, 1))
            self.assertTrue(0 <= cell < sphere.CELLS)
            self.assertEqual(sphere.screen_of_cell(player, cell), screen)
            seen.add((player, cell))
        self.assertEqual(len(seen), sphere.SCREENS)

    def test_the_readme_wiring(self):
        # unit n = column n - 1, output k = row k - 1; screen s = unit ceil(s / 4), output ((s - 1) mod 4) + 1
        self.assertEqual(sphere.cell_of_screen(1), (0, 0))      # unit 1 output 1
        self.assertEqual(sphere.cell_of_screen(2), (0, 6))      # unit 1 output 2: row 1
        self.assertEqual(sphere.cell_of_screen(5), (0, 1))      # unit 2 output 1: column 1
        self.assertEqual(sphere.cell_of_screen(24), (0, 23))
        self.assertEqual(sphere.cell_of_screen(25), (1, 0))
        self.assertEqual(sphere.cell_of_screen(48), (1, 23))
        # grid row 1 is height 1 of the six ribs in order, grid row 0 the tops and bottoms of the 5-ribs
        self.assertEqual([sphere.screen_of_cell(0, c) for c in range(6, 12)], [2, 6, 10, 14, 18, 22])
        self.assertEqual([sphere.screen_of_cell(0, c) for c in range(6)], [1, 5, 9, 13, 17, 21])

    def test_ribs_and_heights(self):
        self.assertEqual(sphere.rib_height(1), (0, 0))
        self.assertEqual(sphere.rib_height(5), (0, 4))
        self.assertEqual(sphere.rib_height(6), (1, 1))
        self.assertEqual(sphere.rib_height(8), (1, 3))
        self.assertEqual(sphere.rib_height(9), (2, 0))
        self.assertEqual(sphere.rib_height(48), (11, 3))
        self.assertEqual(sphere.screen_at(1, 0), None)
        self.assertEqual(sphere.screen_at(12, 0), 1)      # round the sphere

    def test_rows(self):
        self.assertEqual(sphere.row_screens(0), [1, 9, 17, 25, 33, 41])
        self.assertEqual(sphere.row_screens(1), [2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46])
        self.assertEqual(sphere.row_screens(3), [4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48])
        self.assertEqual(sorted(s for h in range(5) for s in sphere.row_screens(h)), list(range(1, 49)))

    def test_angles(self):
        angles = sphere.cell_angles()
        self.assertEqual(len(angles), 48)
        # the ribs count to the left: rib 1 is at -30 degrees
        self.assertEqual(angles[:6], [0.0, 0.0, -60.0, -60.0, -120.0, -120.0])
        self.assertEqual(angles[6:12], [0.0, -30.0, -60.0, -90.0, -120.0, -150.0])
        self.assertEqual(angles[24:30], [180.0, 180.0, 120.0, 120.0, 60.0, 60.0])
        self.assertEqual(angles[30:36], [180.0, 150.0, 120.0, 90.0, 60.0, 30.0])

    def test_the_angles_file_is_the_wiring(self):
        data = Path('apps/smile/data/audio/screen_angles.txt').read_bytes()
        self.assertFalse(data.startswith(b'\xef\xbb\xbf'), 'a byte order mark breaks the first entry in Max')
        # the line endings are the checkout's (a dev PC with autocrlf has CRLF): the lines must match, the endings may differ
        self.assertEqual(data.decode('ascii').replace('\r\n', '\n'), sphere.screen_angles_text(),
                         'regenerate the file: python -c "from apps.smile.sphere import screen_angles_text as t; print(t(), end=\'\')"')


if __name__ == '__main__':
    unittest.main()
