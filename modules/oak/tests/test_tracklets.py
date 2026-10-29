"""The hand-over of depthai's tracklets to the contract: every device status maps, nothing else."""

import unittest

import depthai as dai

from modules.camera import TrackingStatus
from modules.oak.camera.definitions import TRACKING_STATUS


class TrackingStatusMapTest(unittest.TestCase):
    def test_every_device_status_maps(self) -> None:
        for status in dai.Tracklet.TrackingStatus.__members__.values():
            self.assertIn(status, TRACKING_STATUS)

    def test_the_names_agree(self) -> None:
        for device_status, status in TRACKING_STATUS.items():
            self.assertEqual(device_status.name, status.name)
        self.assertEqual(set(TRACKING_STATUS.values()), set(TrackingStatus))


if __name__ == '__main__':
    unittest.main()
