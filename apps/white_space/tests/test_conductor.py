"""The Conductor's per-tick order, where the mix and the motor command must agree.

`tick()` runs before the update callbacks because the playhead it feeds is what the state
machine reads — but the machine commands the motor and hands over its mix from inside those
callbacks. Both have to reach the *same* frame: the fixture's readout mode follows the rpm it
receives, so a frame whose rpm says ring mode while its mix wrote only bar lights is a black
one (which is what entering S8/S9 used to show, for exactly one frame).
"""

import unittest

from apps.white_space.board import Board
from apps.white_space.light import Conductor, Frame, LayerId, MotorMode, FIXTURE_SLOW_RPM
from apps.white_space.light.clock import Tick
from apps.white_space.settings import Settings, Stage

DT = 1.0 / 30.0


class RegimeSwitchTest(unittest.TestCase):

    def setUp(self) -> None:
        settings = Settings()
        self.conductor = Conductor(settings.light, distortion=settings.camera.tracker.distortion,
                                   board=Board(), pose_stage=int(Stage.LERP))
        self.frames: list[Frame] = []
        self.conductor.add_render_callback(self.frames.append)
        self.time: float = 0.0

    def _tick(self) -> Frame:
        self.conductor._update(Tick(self.time, DT))
        self.time += DT
        return self.frames[-1]

    def _enter_wind_down(self) -> None:
        """What a show state's entry does: the new motor command and the new mix, together."""
        self.conductor.set_motor_mode(MotorMode.LOW)
        self.conductor.set_mix([(LayerId.wind_down, 1.0)])

    def test_a_command_from_an_update_callback_reaches_the_same_frame(self) -> None:
        self.conductor.set_motor_mode(MotorMode.HIGH)
        self.conductor.set_mix([(LayerId.flood, 1.0)])
        self.assertGreaterEqual(self._tick().motor.target_rpm, FIXTURE_SLOW_RPM)

        self.conductor.add_update_callback(self._enter_wind_down)
        frame = self._tick()
        self.assertEqual(frame.motor.mode, MotorMode.LOW)
        self.assertLess(frame.motor.target_rpm, FIXTURE_SLOW_RPM)

    def test_the_switch_frame_is_not_black(self) -> None:
        # The regression: END's wall (a ring layer at HIGH) handing over to END_INTRO's wall
        # (bar lights at LOW). The frame the fixture is sent must be readable in the regime its
        # own rpm selects — here slot mode, with the bar lights lit.
        self.conductor.set_motor_mode(MotorMode.HIGH)
        self.conductor.set_mix([(LayerId.flood, 1.0)])
        self.assertGreater(float(self._tick().white.sum()), 0.0)      # END: a lit ring

        self.conductor.add_update_callback(self._enter_wind_down)
        frame = self._tick()
        self.assertLess(frame.motor.target_rpm, FIXTURE_SLOW_RPM)     # the fixture reads the slots …
        self.assertGreater(float(frame.bar_lights.sum()), 0.0)        # … and they are lit

    def test_a_steady_state_is_unaffected(self) -> None:
        self.conductor.set_motor_mode(MotorMode.LOW)
        self.conductor.set_mix([(LayerId.playhead_low, 1.0)])
        for _ in range(3):
            frame = self._tick()
            self.assertEqual(frame.motor.mode, MotorMode.LOW)
            self.assertGreater(float(frame.bar_lights.sum()), 0.0)


if __name__ == '__main__':
    unittest.main()
