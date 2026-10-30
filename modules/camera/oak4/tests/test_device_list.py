"""The OAK4 backend's device list leaves out the OAK-D cameras depthai 3 also sees. Runs on a depthai 3 install."""

import unittest
from types import SimpleNamespace
from unittest import mock

import depthai as dai

from modules.camera.oak4.definitions import get_device_list


def _device(device_id: str, platform: dai.XLinkPlatform) -> SimpleNamespace:
    return SimpleNamespace(getDeviceId=lambda: device_id, name=device_id, platform=platform, state='UNBOOTED')


class DeviceListTest(unittest.TestCase):
    def test_an_oak_d_is_never_listed(self) -> None:
        found = [_device('oak_d', dai.XLinkPlatform.X_LINK_MYRIAD_X), _device('oak4', dai.XLinkPlatform.X_LINK_RVC4)]
        with mock.patch.object(dai.Device, 'getAllAvailableDevices', return_value=found):
            self.assertEqual(get_device_list(), ['oak4'])


if __name__ == '__main__':
    unittest.main()
