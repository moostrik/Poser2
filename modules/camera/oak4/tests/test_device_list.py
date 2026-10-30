"""The OAK4 backend's device list leaves out the OAK-D cameras depthai 3 also sees, and lists an OAK4
once whichever links it is on; `find_device` picks the link. Runs on a depthai 3 install."""

import unittest
from types import SimpleNamespace
from unittest import mock

import depthai as dai

from modules.camera.oak4 import CameraLink
from modules.camera import CameraResolution
from modules.camera.oak4.definitions import get_device_list, find_device, link_resolution

USB = dai.XLinkProtocol.X_LINK_USB_EP
ETHERNET = dai.XLinkProtocol.X_LINK_TCP_IP


def _device(device_id: str, platform: dai.XLinkPlatform, protocol: dai.XLinkProtocol = ETHERNET) -> SimpleNamespace:
    return SimpleNamespace(getDeviceId=lambda: device_id, name=device_id, platform=platform, protocol=protocol, state='UNBOOTED')


def _oak4(protocol: dai.XLinkProtocol) -> SimpleNamespace:
    return _device('oak4', dai.XLinkPlatform.X_LINK_RVC4, protocol)


class DeviceListTest(unittest.TestCase):
    def test_an_oak_d_is_never_listed(self) -> None:
        found = [_device('oak_d', dai.XLinkPlatform.X_LINK_MYRIAD_X), _oak4(ETHERNET)]
        with mock.patch.object(dai.Device, 'getAllAvailableDevices', return_value=found):
            self.assertEqual(get_device_list(), ['oak4'])

    def test_an_oak4_on_both_links_is_listed_once(self) -> None:
        with mock.patch.object(dai.Device, 'getAllAvailableDevices', return_value=[_oak4(ETHERNET), _oak4(USB)]):
            self.assertEqual(get_device_list(), ['oak4'])


class FindDeviceTest(unittest.TestCase):
    def test_the_asked_link_wins_whatever_order_depthai_lists_them(self) -> None:
        for found in ([_oak4(ETHERNET), _oak4(USB)], [_oak4(USB), _oak4(ETHERNET)]):
            with mock.patch.object(dai.Device, 'getAllAvailableDevices', return_value=found):
                self.assertEqual(find_device('oak4', CameraLink.USB).protocol, USB)
                self.assertEqual(find_device('oak4', CameraLink.ETHERNET).protocol, ETHERNET)

    def test_a_missing_link_falls_back_to_the_other_with_a_warning(self) -> None:
        with mock.patch.object(dai.Device, 'getAllAvailableDevices', return_value=[_oak4(ETHERNET)]):
            with self.assertLogs('modules.camera.oak4.definitions', level='WARNING'):
                self.assertEqual(find_device('oak4', CameraLink.USB).protocol, ETHERNET)

    def test_a_camera_on_no_link_is_none(self) -> None:
        with mock.patch.object(dai.Device, 'getAllAvailableDevices', return_value=[]):
            self.assertIsNone(find_device('oak4', CameraLink.USB))


class LinkResolutionTest(unittest.TestCase):
    def test_each_link_gets_its_own_size(self) -> None:
        self.assertEqual(link_resolution(USB, CameraResolution.P1080, CameraResolution.P800), CameraResolution.P1080)
        self.assertEqual(link_resolution(ETHERNET, CameraResolution.P1080, CameraResolution.P800), CameraResolution.P800)


if __name__ == '__main__':
    unittest.main()
