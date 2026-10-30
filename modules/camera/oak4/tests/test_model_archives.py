"""The archives each model chain names: `build_pipeline` tells the chains apart by which are set.
Runs on a depthai 3 install."""

import unittest

from modules.camera.oak4 import PoseChain
from modules.camera.oak4.definitions import model_archives, DETECTOR_ARCHIVE, PERSON_DETECTOR_ARCHIVE, POSE_ARCHIVE, \
    TOP_DOWN_ARCHIVE


class ModelArchivesTest(unittest.TestCase):
    def test_each_chain_names_its_archives(self) -> None:
        self.assertEqual(model_archives(PoseChain.ONE_MODEL), (POSE_ARCHIVE, None))
        self.assertEqual(model_archives(PoseChain.TWO_MODELS), (TOP_DOWN_ARCHIVE, DETECTOR_ARCHIVE))
        self.assertEqual(model_archives(PoseChain.DETECTOR), (None, PERSON_DETECTOR_ARCHIVE))


if __name__ == '__main__':
    unittest.main()
