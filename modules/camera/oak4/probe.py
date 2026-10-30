"""The OAK4 bring-up: the real backend on one device for a few seconds, reporting what arrives.

    python -m modules.camera.oak4.probe [device_id] [-s SECONDS] [--fps FPS] [--resolution P720|P800|P1080]
        [--link USB|ETHERNET] [--color] [--flip-h] [--keystone K] [--tilt-offset T]
        [--no-pose | --two-models | --detector] [--depth NEURAL_288|NEURAL_384|NEURAL_480] [--projector I]
        [--flood-light I]

Lists the devices depthai sees, opens the first (or the one named) through `Camera` exactly as an
app would, and prints the frame rate, the frame type and size, the tracklets with their keypoints
when the pose model's archive is in `data/models` (`--no-pose` leaves it out, `--two-models` runs
the detector plus the pose model on the crops, `--detector` the detector alone), the depth with
`--depth`, the readings (exposure readback, tilt and roll from the IMU, the lens) at the end. A
failure names the depthai call, not an app.
"""

import argparse
import logging
import sys
import time

from .. import CameraFrame, CameraResolution, CameraTracklet, FrameType
from .camera import Camera
from .definitions import get_device_list, CameraLink, DepthModel, PoseChain
from .settings import Oak4CameraSettings

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='stream one OAK4 through the oak4 backend and report')
    parser.add_argument('device_id', nargs='?', default='', help='device id or IP; the first found when left out')
    parser.add_argument('-s', '--seconds', type=float, default=10.0)
    parser.add_argument('--fps', type=float, default=30.0)
    parser.add_argument('--resolution', choices=[r.name for r in CameraResolution], default=CameraResolution.P800.name)
    parser.add_argument('--link', choices=[l.name for l in CameraLink], default=CameraLink.USB.name)
    parser.add_argument('--color', action='store_true', help='colour frames')
    parser.add_argument('--depth', choices=[m.name for m in DepthModel], default=DepthModel.OFF.name)
    parser.add_argument('--projector', type=float, default=0.0, help='dot projector brightness, 0 to 1')
    parser.add_argument('--flood-light', type=float, default=0.0, help='IR flood light brightness, 0 to 1')
    parser.add_argument('--flip-h', action='store_true')
    parser.add_argument('--keystone', type=float, default=0.0)
    parser.add_argument('--tilt-offset', type=float, default=None, help='turn the tilt correction on, with this offset (°)')
    chain = parser.add_mutually_exclusive_group()
    chain.add_argument('--no-pose', action='store_true', help='frames only, no model on the device')
    chain.add_argument('--two-models', action='store_true', help='the detector and the pose model on each crop')
    chain.add_argument('--detector', action='store_true', help='the detector alone: boxes, no keypoints')
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(name)s: %(message)s')

    devices: list[str] = get_device_list(verbose=True)
    if not devices:
        logger.error('no device found: is the camera on the switch, with PoE+?')
        return 1
    settings = Oak4CameraSettings()
    settings.device_id = args.device_id or devices[0]
    settings.fps = args.fps
    settings.resolution = CameraResolution[args.resolution]
    settings.link = CameraLink[args.link]
    settings.color = args.color
    settings.depth_model = DepthModel[args.depth]
    settings.sensor.dot_projector = args.projector
    settings.sensor.flood_light = args.flood_light
    settings.flip_h = args.flip_h
    settings.keystone = args.keystone
    settings.tilt.enabled = args.tilt_offset is not None
    settings.tilt.offset = args.tilt_offset or 0.0
    settings.yolo = not args.no_pose
    settings.pose_chain = PoseChain.TWO_MODELS if args.two_models else PoseChain.DETECTOR if args.detector else PoseChain.ONE_MODEL
    settings.initialize()

    frames: list[CameraFrame] = []
    depths: list[CameraFrame] = []
    tracklets: list[list[CameraTracklet]] = []
    camera = Camera(settings)
    camera.add_frame_callback(lambda frame: (frames if frame.kind == FrameType.VIDEO else depths).append(frame))
    camera.add_tracklet_callback(lambda cam_id, message, timestamp: tracklets.append(message))
    camera.start()
    started: float = time.perf_counter()
    video_latency: list[float] = []
    tracker_latency: list[float] = []
    depth_latency: list[float] = []
    while time.perf_counter() - started < args.seconds and camera.is_alive():
        time.sleep(0.5)
        video_latency.append(settings.readings.video_latency)
        tracker_latency.append(settings.readings.tracker_latency)
        depth_latency.append(settings.readings.depth_latency)
        logger.info(f'{len(frames)} frames, {settings.readings.video_fps:.1f} fps, {settings.readings.video_latency:.1f} ms, '
                    f'{len(tracklets)} tracklet messages, {settings.readings.tracker_fps:.1f}/s, {settings.readings.tracker_latency:.1f} ms, '
                    f'{len(depths)} depth frames, {settings.readings.depth_fps:.1f} fps, {settings.readings.depth_latency:.1f} ms, '
                    f'exposure {settings.sensor.exposure} us iso {settings.sensor.iso}, '
                    f'tilt {settings.readings.tilt_measured:+.1f} roll {settings.readings.roll_measured:+.1f}')
    camera.stop()
    camera.join(5.0)

    if not frames:
        logger.error('no frame arrived')
        return 1
    first: CameraFrame = frames[0]
    settled_video, settled_tracker = video_latency[len(video_latency) // 2:], tracker_latency[len(tracker_latency) // 2:]
    logger.info(f'capture to arrival, second half of the run: frames mean {sum(settled_video) / len(settled_video):.1f} ms, '
                f'max {max(settled_video):.1f}; boxes mean {sum(settled_tracker) / len(settled_tracker):.1f} ms, max {max(settled_tracker):.1f}')
    if depths:
        settled_depth = depth_latency[len(depth_latency) // 2:]
        depth_gaps = [b.timestamp - a.timestamp for a, b in zip(depths, depths[1:])]
        valid: float = float((depths[-1].image > 0).mean()) * 100.0
        logger.info(f'{len(depths)} depth frames of {depths[-1].image.shape} {depths[-1].image.dtype}, {valid:.1f} % valid; '
                    f'{settings.readings.depth_fps:.1f} fps, worst gap {max(depth_gaps, default=0.0) * 1000:.1f} ms, '
                    f'latency mean {sum(settled_depth) / len(settled_depth):.1f} ms, max {max(settled_depth):.1f}')
    gaps = [b.timestamp - a.timestamp for a, b in zip(frames, frames[1:])]
    logger.info(f'{len(frames)} frames of {first.image.shape} {first.image.dtype}; '
                f'{settings.readings.video_fps:.1f} fps average, worst gap {max(gaps) * 1000:.1f} ms' if gaps else 'one frame')
    people = [t for message in tracklets for t in message]
    with_points = [t for t in people if t.keypoints is not None]
    if people:
        sample = with_points[-1] if with_points else people[-1]
        logger.info(f'{len(people)} tracklets in {len(tracklets)} messages, {len(with_points)} with keypoints; '
                    f'last: id {sample.id} {sample.status.name} box {sample.roi.x:.2f},{sample.roi.y:.2f} '
                    f'{sample.roi.width:.2f}x{sample.roi.height:.2f}'
                    + (f', keypoints {sample.keypoints.shape} scores {sample.keypoints[:, 2].min():.2f}..'
                       f'{sample.keypoints[:, 2].max():.2f}' if sample.keypoints is not None else ''))
    elif not args.no_pose:
        logger.info('no tracklet arrived: nobody in view, or the model did not run')
    if camera.pose_join is not None:
        logger.info(f'two-model chain: {camera.pose_join.joined} frames joined, {camera.pose_join.dropped} dropped incomplete')
    logger.info(f'lens: field {settings.readings.fov_factory:.1f} deg, {settings.readings.lens_error:.2f} deg off the preset lens; '
                f'tilt {settings.readings.tilt_measured:+.1f} roll {settings.readings.roll_measured:+.1f} deg')
    return 0


if __name__ == '__main__':
    sys.exit(main())
