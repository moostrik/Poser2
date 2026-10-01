"""Tests for the camera feature writer against a fake camera (no Vimba)."""

import unittest

from modules.cam.FeatureWriter import FeatureWriter
from modules.cam.settings import CameraSettings


class FakeFeature:
    def __init__(self, camera: 'FakeCamera', name: str, value, range_=None) -> None:
        self.camera = camera
        self.name = name
        self.value = value
        self.range = range_

    def get(self):
        if self.name == 'BalanceRatioAbs':
            return self.camera.ratios[self.camera.features['BalanceRatioSelector'].value]
        return self.value

    def get_range(self):
        return self.range

    def set(self, value) -> None:
        self.camera.writes.append((self.name, value))
        if self.name == 'BalanceRatioAbs':
            self.camera.ratios[self.camera.features['BalanceRatioSelector'].value] = value
        else:
            self.value = value


class FakeCamera:
    def __init__(self, without: tuple[str, ...] = ()) -> None:
        self.writes: list[tuple[str, object]] = []
        self.ratios: dict[str, float] = {'Red': 1.0, 'Blue': 1.0}
        self.features: dict[str, FakeFeature] = {}
        for name, value, range_ in (
            ('GainAuto', 'Off', None), ('GainAutoTarget', 50, (0, 100)),
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


class TestFeatureWriter(unittest.TestCase):
    def setUp(self):
        self.settings = CameraSettings()
        self.settings.initialize()
        self.writer = FeatureWriter(self.settings)
        self.camera = FakeCamera()

    def tearDown(self):
        self.writer.close()

    def test_sends_everything_once_then_only_what_changed(self):
        self.settings.wb_red = 1.5
        self.writer.update(self.camera)
        self.assertEqual(self.camera.written('GainAuto'), ['Off'])
        self.assertEqual(self.camera.written('Saturation'), [1.0])
        self.assertEqual(self.camera.written('BalanceWhiteAuto'), ['Off'])
        self.assertEqual(self.camera.ratios, {'Red': 1.5, 'Blue': 1.0})

        self.camera.writes.clear()
        self.writer.update(self.camera)
        self.assertEqual(self.camera.writes, [])

        self.settings.saturation = 1.2
        self.writer.update(self.camera)
        self.assertEqual(self.camera.writes, [('Saturation', 1.2)])

    def test_clamps_to_the_range_of_the_feature(self):
        self.settings.gamma = 3.0
        self.settings.wb_blue = 0.5
        self.writer.update(self.camera)
        self.assertEqual(self.camera.written('Gamma'), [1.5])
        self.assertEqual(self.camera.ratios['Blue'], 0.8)

    def test_a_missing_feature_is_skipped(self):
        camera = FakeCamera(without=('Hue', 'BalanceRatioAbs'))
        self.settings.hue = 10.0
        self.writer.update(camera)
        self.assertEqual(camera.written('Saturation'), [1.0])
        self.settings.hue = 20.0
        self.settings.wb_red = 2.0
        self.writer.update(camera)
        self.assertEqual(camera.written('Hue'), [])
        self.assertEqual(camera.ratios['Red'], 1.0)

    def test_auto_owns_the_ratios_and_the_panel_takes_them_back(self):
        self.settings.wb_auto = True
        self.settings.wb_red = 2.0
        self.writer.update(self.camera)
        self.assertEqual(self.camera.written('BalanceWhiteAuto'), ['Continuous'])
        self.assertEqual(self.camera.written('BalanceRatioAbs'), [])

        self.settings.wb_auto = False
        self.writer.update(self.camera)
        self.assertEqual(self.camera.written('BalanceWhiteAuto'), ['Continuous', 'Off'])
        self.assertEqual(self.camera.ratios['Red'], 2.0)

    def test_set_white_balance_measures_once_and_stores_the_ratios(self):
        self.writer.update(self.camera)
        CameraSettings.wb_set.fire(self.settings)
        self.writer.update(self.camera)
        self.assertEqual(self.camera.features['BalanceWhiteAuto'].value, 'Once')

        # still measuring: the panel's ratios are not sent
        self.settings.wb_red = 2.5
        self.writer.update(self.camera)
        self.assertEqual(self.camera.ratios['Red'], 1.0)

        # the camera is done
        self.camera.ratios = {'Red': 1.83, 'Blue': 2.41}
        self.camera.features['BalanceWhiteAuto'].value = 'Off'
        self.writer.update(self.camera)
        self.assertEqual(self.settings.wb_red, 1.83)
        self.assertEqual(self.settings.wb_blue, 2.41)

        self.camera.writes.clear()
        self.writer.update(self.camera)
        self.assertEqual(self.camera.written('BalanceRatioAbs'), [])


if __name__ == '__main__':
    unittest.main()
