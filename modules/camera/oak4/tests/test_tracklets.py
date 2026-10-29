"""The hand-over of depthai's tracklets to the contract: every device status maps. Runs on a
depthai 3 install (`install.bat oak4`); the oak backend has the same test for depthai 2."""

import unittest

import depthai as dai

from modules.camera import TrackingStatus
from modules.camera.oak4.definitions import TRACKING_STATUS


class TrackingStatusMapTest(unittest.TestCase):
    def test_every_device_status_maps(self) -> None:
        for status in dai.Tracklet.TrackingStatus.__members__.values():
            self.assertIn(status, TRACKING_STATUS)
        self.assertEqual(set(TRACKING_STATUS.values()), set(TrackingStatus))


if __name__ == '__main__':
    unittest.main()
