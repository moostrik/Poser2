"""Tests for FixtureSpeedLatch — the host-side model of the fixture's speed latch.

The firmware acts on a received rpm only when it differs from the last one it applied, and its
packet watchdog stops the motor after >5 revolutions of packet silence without recording that
stop (see the FixtureSpeedLatch docstring). After any send silence the sender must therefore
re-establish the speed with a 0→target edge. These tests pin down exactly when the latch
demands that edge — injected time only, no sockets, no sleeps.
"""

import unittest
from itertools import takewhile

from apps.white_space.inout.osc_light_sender import FixtureSpeedLatch

FRAME = 1.0 / 32.0      # send cadence (s); a power-of-two fraction, so the time sums stay exact
BEAM = 72
PROJECTION = 1920
HOLD = 0.3              # resync hold under test (s)


def leading_zeros(sent: list[int]) -> int:
    return len(list(takewhile(lambda rpm: rpm == 0, sent)))


class LatchDriver:
    """Drives a latch the way the sender thread does: one rpm_to_send + note_sent per frame."""

    def __init__(self) -> None:
        self.latch = FixtureSpeedLatch()
        self.now = 1000.0   # arbitrary monotonic origin

    def run(self, target: int, seconds: float) -> list[int]:
        """Send frames at the regular cadence for `seconds`; returns the rpms put on the wire."""
        sent = []
        for _ in range(round(seconds / FRAME)):
            rpm = self.latch.rpm_to_send(target, HOLD, self.now)
            self.latch.note_sent(rpm, self.now)
            sent.append(rpm)
            self.now += FRAME
        return sent

    def gap(self, seconds: float) -> None:
        """Send nothing for `seconds` (simulation, a host hitch, a stalled sender)."""
        self.now += seconds


class BootHoldTest(unittest.TestCase):
    def test_boot_hold_runs_at_the_first_send_not_at_arming(self) -> None:
        """Starting with simulation on, the boot hold must survive until sending begins."""
        driver = LatchDriver()
        driver.latch.begin_hold(2.0)
        driver.gap(60.0)                    # armed long ago; nothing sent meanwhile
        sent = driver.run(BEAM, 2.5)
        hold_frames = round(2.0 / FRAME)
        self.assertEqual(sent[:hold_frames], [0] * hold_frames)
        self.assertEqual(set(sent[hold_frames:]), {BEAM})

    def test_arming_never_shortens_a_pending_hold(self) -> None:
        driver = LatchDriver()
        driver.latch.begin_hold(2.0)
        driver.latch.begin_hold(HOLD)
        sent = driver.run(BEAM, 2.5)
        self.assertEqual(leading_zeros(sent), round(2.0 / FRAME))


class SendGapTest(unittest.TestCase):
    def test_projection_gap_forces_an_edge(self) -> None:
        """200 ms of silence at 1920 rpm can have tripped the watchdog (>5 revs ≈ 190 ms)."""
        driver = LatchDriver()
        driver.run(PROJECTION, 2.0)
        driver.gap(0.2)
        sent = driver.run(PROJECTION, 1.0)
        zeros = leading_zeros(sent)
        self.assertGreater(zeros, 0)                        # the 0 that re-arms the firmware latch
        self.assertAlmostEqual(zeros * FRAME, HOLD, delta=2 * FRAME)
        self.assertEqual(set(sent[zeros:]), {PROJECTION})   # and the release to the target

    def test_beam_short_gap_causes_no_hold(self) -> None:
        """At 72 rpm the watchdog needs ~5 s; a 2 s gap must not disturb the motor."""
        driver = LatchDriver()
        driver.run(BEAM, 2.0)
        driver.gap(2.0)
        self.assertEqual(set(driver.run(BEAM, 0.5)), {BEAM})

    def test_beam_long_gap_forces_an_edge(self) -> None:
        driver = LatchDriver()
        driver.run(BEAM, 2.0)
        driver.gap(4.0)                     # > 4 revs at 72 rpm (3.33 s)
        sent = driver.run(BEAM, 1.0)
        self.assertGreater(leading_zeros(sent), 0)
        self.assertEqual(sent[-1], BEAM)

    def test_never_spun_has_no_gap_bound(self) -> None:
        """A bar never commanded to spin makes no revolutions; the first spin is its own edge."""
        driver = LatchDriver()
        driver.run(0, 1.0)
        driver.gap(60.0)
        self.assertEqual(driver.run(BEAM, 0.5)[0], BEAM)


class PhysicalCeilingTest(unittest.TestCase):
    """The watchdog counts physical revolutions: right after PROJECTION the bar is still fast,
    whatever rpm is being commanded, so the gap bound must stay tight until it has spun down."""

    def test_gap_while_braking_from_projection_forces_an_edge(self) -> None:
        driver = LatchDriver()
        driver.run(PROJECTION, 2.0)
        driver.run(BEAM, 1.0)               # commanded down; the bar is still braking
        driver.gap(1.0)                     # harmless at 72 rpm, fatal at ~1700
        sent = driver.run(BEAM, 1.0)
        self.assertGreater(leading_zeros(sent), 0)

    def test_same_gap_after_spindown_causes_no_hold(self) -> None:
        driver = LatchDriver()
        driver.run(PROJECTION, 2.0)
        driver.run(BEAM, 10.0)              # ceiling decays to the beam rpm
        driver.gap(1.0)
        self.assertEqual(set(driver.run(BEAM, 0.5)), {BEAM})


class InvalidateTest(unittest.TestCase):
    def test_invalidate_forces_an_edge(self) -> None:
        """A reconnect (new endpoint, unknown latch) must re-establish the speed."""
        driver = LatchDriver()
        driver.run(BEAM, 1.0)
        driver.latch.invalidate()
        sent = driver.run(BEAM, 1.0)
        self.assertGreater(leading_zeros(sent), 0)
        self.assertEqual(sent[-1], BEAM)


if __name__ == "__main__":
    unittest.main()
