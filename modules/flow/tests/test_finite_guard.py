"""The explosion guard without GL: its report from its counts."""

import unittest

from modules.flow import guard_report


class TestGuardReport(unittest.TestCase):

    def test_quiet_when_every_count_is_zero(self) -> None:
        self.assertEqual(guard_report('fluid', ('velocity', 'density'), [0, 0], 1.0), [])

    def test_one_warning_per_field_with_bad_values(self) -> None:
        messages = guard_report('fluid', ('velocity', 'density', 'temperature'), [1240, 0, 3], 1.02)
        self.assertEqual(messages, ['fluid: 1240 non-finite velocity values set to 0 in the last 1.0 s',
                                    'fluid: 3 non-finite temperature values set to 0 in the last 1.0 s'])


if __name__ == '__main__':
    unittest.main()
