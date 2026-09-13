"""The Conductor's per-tick order: a frame is sent with the command its mix was drawn for.

The motor is measured and the playhead advanced at the top of the tick (the state machine
reads that playhead the same tick), the machine commands the motor and hands over its mix
from the update callbacks, and the frame takes the command as it stands *after* them. The
fixture's readout mode follows the rpm it receives, so a frame whose rpm said projection mode
while its mix wrote only beam lights would be a black one.
"""

import unittest

from apps.white_space.board import Board
from apps.white_space.light import Conductor, Frame, LayerId, MotorMode, FIXTURE_PROJECTION_RPM
from apps.white_space.light.clock import Tick
from apps.white_space.settings import Settings, Stage

DT = 1.0 / 30.0


class RegimeSwitchTest(unittest.TestCase):

    def setUp(self) -> None:
        settings = Settings()
        self.conductor = Conductor(settings.light, board=Board(), pose_stage=int(Stage.LERP))
        self.frames: list[Frame] = []
        self.conductor.add_render_callback(self.frames.append)
        self.time: float = 0.0

    def _tick(self) -> Frame:
        self.conductor._update(Tick(self.time, DT))
        self.time += DT
        return self.frames[-1]

    def _enter_wind_down(self) -> None:
        """What a show state's entry does: the new motor command and the new mix, together."""
        self.conductor.set_motor_mode(MotorMode.BEAM)
        self.conductor.set_mix([(LayerId.beam_wind_down, 1.0)])

    def test_a_command_from_an_update_callback_reaches_the_same_frame(self) -> None:
        self.conductor.set_motor_mode(MotorMode.PROJECTION)
        self.conductor.set_mix([(LayerId.flood, 1.0)])
        self.assertGreaterEqual(self._tick().motor_command.target_rpm, FIXTURE_PROJECTION_RPM)

        self.conductor.add_update_callback(self._enter_wind_down)
        frame = self._tick()
        self.assertEqual(frame.motor_command.mode, MotorMode.BEAM)
        self.assertLess(frame.motor_command.target_rpm, FIXTURE_PROJECTION_RPM)

    def test_the_switch_frame_is_not_black(self) -> None:
        # END's wall (a projection layer at PROJECTION) handing over to END_INTRO's wall (beam lights at BEAM):
        # the frame the fixture is sent must be readable in the mode its own rpm selects —
        # here beam mode, with the beam lights lit.
        self.conductor.set_motor_mode(MotorMode.PROJECTION)
        self.conductor.set_mix([(LayerId.flood, 1.0)])
        self.assertGreater(float(self._tick().white.sum()), 0.0)      # END: a lit projection

        self.conductor.add_update_callback(self._enter_wind_down)
        frame = self._tick()
        self.assertLess(frame.motor_command.target_rpm, FIXTURE_PROJECTION_RPM)   # the fixture reads the slots …
        self.assertGreater(float(frame.beam_lights.sum()), 0.0)              # … and they are lit

    def test_a_steady_state_is_unaffected(self) -> None:
        self.conductor.set_motor_mode(MotorMode.BEAM)
        self.conductor.set_mix([(LayerId.beam_playhead, 1.0)])
        for _ in range(3):
            frame = self._tick()
            self.assertEqual(frame.motor_command.mode, MotorMode.BEAM)
            self.assertGreater(float(frame.beam_lights.sum()), 0.0)


if __name__ == '__main__':
    unittest.main()
