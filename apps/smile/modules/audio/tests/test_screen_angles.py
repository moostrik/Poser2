"""Tests for data/audio/screen_angles.txt: the angle of every screen on the sphere, which the Max patch's
coll reads (a line per screen: index, angle;). Index = the player's cell (row * 6 + column), + 24 for the
player on port 4116."""

import re
import unittest
from pathlib import Path

SCREENS: int = 48
ENTRY = re.compile(r'^(\d+), (-?\d+(?:\.\d*)?);$')


class TestScreenAngles(unittest.TestCase):
    def test_every_screen_has_an_angle(self):
        data = Path('data/audio/screen_angles.txt').read_bytes()
        self.assertFalse(data.startswith(b'\xef\xbb\xbf'), 'a byte order mark breaks the first entry in Max')
        lines = data.decode('ascii').splitlines()
        self.assertEqual(len(lines), SCREENS)
        for index, line in enumerate(lines):
            entry = ENTRY.match(line)
            self.assertIsNotNone(entry, f'line {index + 1} is not "index, angle;": {line!r}')
            assert entry is not None
            self.assertEqual(int(entry.group(1)), index, line)
            self.assertTrue(-180.0 <= float(entry.group(2)) <= 180.0, line)


if __name__ == '__main__':
    unittest.main()
