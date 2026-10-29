"""Tests for the Vimba sensor control against a fake camera (no Vimba)."""

import unittest
from unittest import mock

from modules.camera.vimba import SensorControl, VimbaCameraSettings


class FakeFeature:
    def __init__(self, camera: 'FakeCamera', name: str, value, range_=None) -> None:
        self.camera = camera
        self.name = name
        self.value = value
        self.range = range_

    def get(self):
        if self.name in self.camera.unreadable:
            raise RuntimeError('<VmbError.Resources: -14>')
        if self.name == 'BalanceRatioAbs':
            return self.camera.ratios[self.camera.features['BalanceRatioSelector'].value]
        return self.value

    def get_range(self):
        return self.range

    def set(self, value) -> None:
        if self.name in self.camera.refuses:
            raise RuntimeError('timeout')
        self.camera.writes.append((self.name, value))
        if self.name == 'BalanceRatioAbs':
            self.camera.ratios[self.camera.features['BalanceRatioSelector'].value] = value
        else:
            self.value = value


class FakeCamera:
    def __init__(self, without: tuple[str, ...] = ()) -> None:
        self.writes: list[tuple[str, object]] = []
        self.refuses: set[str] = set()      # features whose write fails for now
        self.unreadable: set[str] = set()   # features whose read fails for now (the camera is away)
        self.ratios: dict[str, float] = {'Red': 1.0, 'Blue': 1.0}
        self.features: dict[str, FakeFeature] = {}
        for name, value, range_ in (
            ('AcquisitionFrameRateAbs', 20.0, (0.01, 30.3)),
            ('GainAuto', 'Off', None), ('GainAutoTarget', 50, (0, 100)), ('GainAutoMax', 15.0, (0.0, 30.0)),
            ('Gain', 0.0, (0.0, 30.0)),
            ('ExposureAuto', 'Off', None), ('ExposureAutoTarget', 50, (0, 100)), ('ExposureAutoMax', 15000, (12, 84577817)),
            ('ExposureTimeAbs', 33000.0, (12.0, 84577817.0)),
            ('Saturation', 1.0, (0.0, 2.0)), ('Hue', 0.0, (-40.0, 40.0)), ('Gamma', 1.0, (0.45, 1.5)),
            ('BalanceWhiteAuto', 'Off', None), ('BalanceRatioSelector', 'Red', None),
            ('BalanceRatioAbs', 1.0, (0.8, 3.0)),
        ):
            if name not in without:
                self.features[name] = FakeFeature(self, name, value, range_)

    def get_feature_by_name(self, name: str) -> FakeFeature:
        if name not in self.features:
            raise KeyError(name)
        return self.features[name]

    def written(self, name: str) -> list:
        return [v for n, v in self.writes if n == name]


class TestSensorControl(unittest.TestCase):
    def setUp(self):
        self.settings = VimbaCameraSettings()
        self.settings.initialize()
        self.control = SensorControl(self.settings)
        self.camera = FakeCamera()

    def tearDown(self):
        self.control.close()

    def test_sends_everything_once_then_only_what_changed(self):
        self.settings.manual_WB_red = 1.5
        self.control.update(self.camera)
        self.assertEqual(self.camera.written('GainAuto'), ['Off'])
        self.assertEqual(self.camera.written('AcquisitionFrameRateAbs'), [30.0])     # `fps`
        self.assertEqual(self.camera.written('Saturation'), [1.0])
        self.assertEqual(self.camera.written('BalanceWhiteAuto'), ['Off'])
        self.assertEqual(self.camera.ratios, {'Red': 1.5, 'Blue': 1.0})

        self.camera.writes.clear()
        self.control.update(self.camera)
        self.assertEqual(self.camera.writes, [])

        self.settings.saturation = 1.2
        self.control.update(self.camera)
        self.assertEqual(self.camera.writes, [('Saturation', 1.2)])

    def test_the_auto_target_goes_to_exposure_and_gain(self):
        self.settings.target_brightness = 30
        self.settings.exposure_max = 20000.0
        self.control.update(self.camera)
        self.assertEqual(self.camera.written('ExposureAutoTarget'), [30])
        self.assertEqual(self.camera.written('GainAutoTarget'), [30])
        written = self.camera.written('ExposureAutoMax')
        self.assertEqual(written, [20000])
        self.assertIsInstance(written[0], int)      # the feature's own number type

    def test_manual_gain_and_exposure_are_the_cameras_while_auto_is_on(self):
        self.settings.auto_gain = True
        self.settings.auto_exposure = True
        self.settings.manual_gain = 6.0
        self.settings.manual_exposure = 10000.0
        self.control.update(self.camera)
        self.assertEqual(self.camera.written('GainAuto'), ['Continuous'])
        self.assertEqual(self.camera.written('ExposureAuto'), ['Continuous'])
        self.assertEqual(self.camera.written('Gain'), [])
        self.assertEqual(self.camera.written('ExposureTimeAbs'), [])

        self.settings.auto_gain = False
        self.settings.auto_exposure = False
        self.control.update(self.camera)
        self.assertEqual(self.camera.written('Gain'), [6.0])
        self.assertEqual(self.camera.written('ExposureTimeAbs'), [10000.0])

    def test_the_actual_values_follow_the_camera(self):
        self.settings.auto_gain = True
        self.control.update(self.camera)
        self.assertEqual(self.settings.exposure, 33000.0)
        self.camera.features['Gain'].value = 7.26
        self.control._read_next = 0.0
        self.control.update(self.camera)
        self.assertEqual(self.settings.gain, 7.3)

        self.camera.ratios = {'Red': 1.61, 'Blue': 2.04}
        self.control._read_next = 0.0
        self.control.update(self.camera)
        self.assertEqual(self.settings.WB_red, 1.61)
        self.assertEqual(self.settings.WB_blue, 2.04)

    def test_the_frame_rate_and_the_dropped_frames_are_counted(self):
        self.control.update(self.camera)
        self.assertEqual(self.settings.readings.video_fps, 0.0)     # no interval yet
        for _ in range(15):
            self.control.count_frame()
        for _ in range(3):
            self.control.count_frame(False)                  # incomplete: skipped
        self.control._frames_time -= 0.5                     # half a second ago
        self.control._read_next = 0.0
        self.control.update(self.camera)
        self.assertAlmostEqual(self.settings.readings.video_fps, 30.0, delta=0.5)
        self.assertEqual(self.settings.dropped_frames, 3)

    def test_the_worst_gap_between_frames_is_the_minimum_rate(self):
        with mock.patch('modules.camera.vimba.control.perf_counter', side_effect=[0.0, 0.02, 0.12, 0.14]):
            for _ in range(4):
                self.control.count_frame()                   # gaps of 20, 100 and 20 ms
        self.control._frames_time = 0.0                      # a read interval has passed
        self.control._read_next = 0.0
        self.control.update(self.camera)
        self.assertEqual(self.settings.readings.video_min_fps, 10.0)
        self.control._frames_time -= 0.5                    # another interval, without frames
        self.control._read_next = 0.0
        self.control.update(self.camera)
        self.assertEqual(self.settings.readings.video_min_fps, 0.0)    # no frames since: no gap

    def test_clamps_to_the_range_of_the_feature(self):
        self.settings.gamma = 3.0
        self.settings.manual_WB_blue = 0.5
        self.control.update(self.camera)
        self.assertEqual(self.camera.written('Gamma'), [1.5])
        self.assertEqual(self.camera.ratios['Blue'], 0.8)

    def test_a_missing_feature_is_skipped(self):
        camera = FakeCamera(without=('Hue', 'BalanceRatioAbs'))
        self.settings.hue = 10.0
        self.control.update(camera)
        self.assertEqual(camera.written('Saturation'), [1.0])
        self.settings.hue = 20.0
        self.settings.manual_WB_red = 2.0
        self.control.update(camera)
        self.assertEqual(camera.written('Hue'), [])
        self.assertEqual(camera.ratios['Red'], 1.0)

    def test_a_refused_write_is_tried_again_and_does_not_switch_the_feature_off(self):
        self.camera.refuses = {'Hue', 'BalanceRatioSelector'}
        self.settings.hue = 10.0
        with self.assertLogs('modules.camera.vimba.control', 'WARNING') as logs:
            self.control.update(self.camera)
            self.control.update(self.camera)                 # not again before the next read interval
        self.assertEqual(self.camera.written('Hue'), [])
        self.assertEqual(len([line for line in logs.output if 'Hue' in line]), 1)   # logged once
        self.assertEqual(self.control._missing, set())

        self.camera.refuses = set()                         # the camera takes it again
        self.control._retry = {key: 0.0 for key in self.control._retry}
        self.control.update(self.camera)
        self.assertEqual(self.camera.written('Hue'), [10.0])
        self.settings.manual_WB_red = 2.0
        self.control.update(self.camera)
        self.assertEqual(self.camera.ratios['Red'], 2.0)

    def test_a_read_that_keeps_failing_is_logged_once(self):
        self.control.update(self.camera)
        self.camera.unreadable = {'BalanceRatioAbs'}
        with self.assertLogs('modules.camera.vimba.control', 'WARNING') as logs:
            for _ in range(10):                             # twenty reads: five seconds of a camera that is away
                self.control._read_next = 0.0
                self.control.update(self.camera)
        self.assertEqual(len(logs.output), 1)
        self.assertIn('BalanceRatioAbs not readable', logs.output[0])

        self.camera.unreadable = set()                      # back, then away again: logged again
        self.control._read_next = 0.0
        self.control.update(self.camera)
        self.camera.unreadable = {'BalanceRatioAbs'}
        with self.assertLogs('modules.camera.vimba.control', 'WARNING') as logs:
            self.control._read_next = 0.0
            self.control.update(self.camera)
        self.assertEqual(len(logs.output), 1)

    def test_auto_owns_the_ratios_and_the_panel_takes_them_back(self):
        self.settings.auto_WB = True
        self.settings.manual_WB_red = 2.0
        self.control.update(self.camera)
        self.assertEqual(self.camera.written('BalanceWhiteAuto'), ['Continuous'])
        self.assertEqual(self.camera.written('BalanceRatioAbs'), [])

        self.settings.auto_WB = False
        self.control.update(self.camera)
        self.assertEqual(self.camera.written('BalanceWhiteAuto'), ['Continuous', 'Off'])
        self.assertEqual(self.camera.ratios['Red'], 2.0)

    def test_set_white_balance_measures_once_and_stores_the_ratios(self):
        self.control.update(self.camera)
        VimbaCameraSettings.set_WB.fire(self.settings)
        self.control.update(self.camera)
        self.assertEqual(self.camera.features['BalanceWhiteAuto'].value, 'Once')

        # still measuring: the panel's ratios are not sent
        self.settings.manual_WB_red = 2.5
        self.control.update(self.camera)
        self.assertEqual(self.camera.ratios['Red'], 1.0)

        # the camera is done
        self.camera.ratios = {'Red': 1.83, 'Blue': 2.41}
        self.camera.features['BalanceWhiteAuto'].value = 'Off'
        self.control.update(self.camera)
        self.assertEqual(self.settings.manual_WB_red, 1.83)
        self.assertEqual(self.settings.manual_WB_blue, 2.41)

        self.camera.writes.clear()
        self.control.update(self.camera)
        self.assertEqual(self.camera.written('BalanceRatioAbs'), [])


if __name__ == '__main__':
    unittest.main()
