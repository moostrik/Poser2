"""The one builder links the same nodes the same way for every configuration the apps use.

depthai builds a pipeline on the host, so the graph can be read back (`getConnectionMap`) with no
device. The blob is the one thing a host without the model files cannot read, so its input size
and `setBlobPath` are stubbed; everything else is the real depthai.
"""

import unittest
from unittest.mock import patch

import depthai as dai

from modules.camera import CameraResolution
from modules.camera.oak import pipeline as oak_pipeline
from modules.camera.oak.pipeline import WarpConfig, build_pipeline, VIDEO_STREAM, TRACKLETS_STREAM, IMU_STREAM, \
    CONTROL_STREAM, CLIPS_STREAM


MOUNT = WarpConfig(flip_h=True, flip_v=False, tilt=0.0, keystone=0.0, fov_h=127.0)


def build(**kwargs) -> dai.Pipeline:
    options = dict(models='data/models', fps=30.0, color=False, square=True, yolo=True,
                   resolution=CameraResolution.P800, mount=MOUNT)
    options.update(kwargs)
    pipeline = dai.Pipeline()
    with patch.object(oak_pipeline, 'detector_input_size', return_value=(416, 416)), \
         patch.object(dai.node.YoloDetectionNetwork, 'setBlobPath', lambda self, path: None):
        build_pipeline(pipeline, **options)
    return pipeline


def node_names(pipeline: dai.Pipeline) -> set[str]:
    return {type(node).__name__ for node in pipeline.getAllNodes()}


def streams(pipeline: dai.Pipeline, kind) -> set[str]:
    return {node.getStreamName() for node in pipeline.getAllNodes() if isinstance(node, kind)}


def links(pipeline: dai.Pipeline) -> set[tuple[str, str, str, str]]:
    """(from node, output, to node, input), by node type: the shape of the graph."""
    by_id = {node.id: type(node).__name__ for node in pipeline.getAllNodes()}
    out: set[tuple[str, str, str, str]] = set()
    for node_id, connections in pipeline.getConnectionMap().items():
        for c in connections:
            out.add((by_id[c.outputId], c.outputName, by_id[c.inputId], c.inputName))
    return out


class DevicePipelineTest(unittest.TestCase):

    def test_mono_yolo_is_sensor_warp_video_detector_imu(self) -> None:
        # White Space: mono, square, YOLO.
        p = build(color=False)
        self.assertEqual(node_names(p), {'MonoCamera', 'XLinkIn', 'Warp', 'XLinkOut', 'ImageManip',
                                         'YoloDetectionNetwork', 'ObjectTracker', 'IMU'})
        self.assertEqual(streams(p, dai.node.XLinkOut), {VIDEO_STREAM, TRACKLETS_STREAM, IMU_STREAM})
        self.assertEqual(streams(p, dai.node.XLinkIn), {CONTROL_STREAM})
        self.assertEqual(links(p), {
            ('XLinkIn', 'out', 'MonoCamera', 'inputControl'),
            ('MonoCamera', 'out', 'Warp', 'inputImage'),
            ('Warp', 'out', 'XLinkOut', 'in'),
            ('Warp', 'out', 'ImageManip', 'inputImage'),
            ('ImageManip', 'out', 'YoloDetectionNetwork', 'in'),
            ('YoloDetectionNetwork', 'passthrough', 'ObjectTracker', 'inputTrackerFrame'),
            ('YoloDetectionNetwork', 'passthrough', 'ObjectTracker', 'inputDetectionFrame'),
            ('YoloDetectionNetwork', 'out', 'ObjectTracker', 'inputDetections'),
            ('ObjectTracker', 'out', 'XLinkOut', 'in'),
            ('IMU', 'out', 'XLinkOut', 'in'),
        })

    def test_color_takes_the_preview_through_the_same_graph(self) -> None:
        # HD Trio and Deep Flow: colour, square, YOLO.
        p = build(color=True, resolution=CameraResolution.P1080)
        self.assertIn('ColorCamera', node_names(p))
        self.assertNotIn('MonoCamera', node_names(p))
        self.assertIn(('ColorCamera', 'preview', 'Warp', 'inputImage'), links(p))
        self.assertIn(('XLinkIn', 'out', 'ColorCamera', 'inputControl'), links(p))

    def test_without_yolo_there_is_no_detector_branch(self) -> None:
        p = build(yolo=False)
        self.assertEqual(node_names(p), {'MonoCamera', 'XLinkIn', 'Warp', 'XLinkOut', 'IMU'})
        self.assertEqual(streams(p, dai.node.XLinkOut), {VIDEO_STREAM, IMU_STREAM})

    def test_wide_stretches_the_frame_into_the_blob(self) -> None:
        manips = [n for n in build(square=False).getAllNodes() if isinstance(n, dai.node.ImageManip)]
        self.assertFalse(manips[0].initialConfig.getResizeConfig().keepAspectRatio)
        manips = [n for n in build(square=True).getAllNodes() if isinstance(n, dai.node.ImageManip)]
        self.assertTrue(manips[0].initialConfig.getResizeConfig().keepAspectRatio)


class ClipPipelineTest(unittest.TestCase):

    def test_clips_replace_the_sensor_and_skip_the_warp(self) -> None:
        # -sim: the clip is the delivered frame, so it goes straight to the video output and the
        # detector; no sensor, no controls, no IMU. One link to the video output (the 4.1 bug).
        p = build(clips=True)
        self.assertEqual(node_names(p), {'XLinkIn', 'XLinkOut', 'ImageManip', 'YoloDetectionNetwork', 'ObjectTracker'})
        self.assertEqual(streams(p, dai.node.XLinkIn), {CLIPS_STREAM})
        self.assertEqual(streams(p, dai.node.XLinkOut), {VIDEO_STREAM, TRACKLETS_STREAM})
        graph = links(p)
        self.assertIn(('XLinkIn', 'out', 'XLinkOut', 'in'), graph)
        self.assertIn(('XLinkIn', 'out', 'ImageManip', 'inputImage'), graph)
        self.assertNotIn('Warp', node_names(p))
        self.assertEqual(len([c for c in graph if c[0] == 'XLinkIn']), 2)

    def test_warped_clips_go_through_the_warp_like_the_sensor(self) -> None:
        p = build(clips=True, warp_clips=True)
        graph = links(p)
        self.assertIn(('XLinkIn', 'out', 'Warp', 'inputImage'), graph)
        self.assertIn(('Warp', 'out', 'XLinkOut', 'in'), graph)
        self.assertIn(('Warp', 'out', 'ImageManip', 'inputImage'), graph)
        self.assertNotIn(('XLinkIn', 'out', 'XLinkOut', 'in'), graph)

    def test_clip_input_takes_a_sensor_sized_frame(self) -> None:
        # 1280x800 mono at P800, square 800: the ceiling is the larger of the two.
        clips = [n for n in build(clips=True).getAllNodes() if isinstance(n, dai.node.XLinkIn)][0]
        self.assertEqual(clips.getMaxDataSize(), 1280 * 800)
        clips = [n for n in build(clips=True, color=True).getAllNodes() if isinstance(n, dai.node.XLinkIn)][0]
        self.assertEqual(clips.getMaxDataSize(), 1280 * 800 * 3)


class ModelPathTest(unittest.TestCase):

    def test_six_shaves_on_a_device_seven_for_clips(self) -> None:
        self.assertEqual(oak_pipeline.model_path('m', True, False).name, 'yolov8n_coco_416x416_6S.blob')
        self.assertEqual(oak_pipeline.model_path('m', True, True).name, 'yolov8n_coco_416x416_7S.blob')
        self.assertEqual(oak_pipeline.model_path('m', False, False).name, 'yolov8n_coco_512x448_6S.blob')
        self.assertEqual(oak_pipeline.model_path('m', False, True).name, 'yolov8n_coco_512x448_7S.blob')


if __name__ == '__main__':
    unittest.main()
