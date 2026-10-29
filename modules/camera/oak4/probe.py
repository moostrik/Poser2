"""The OAK4 bring-up: the real backend on one device for a few seconds, reporting what arrives.

    python -m modules.camera.oak4.probe [device_id] [-s SECONDS] [--color] [--keystone K | --tilt T] [--no-pose | --two-models]

Lists the devices depthai sees, opens the first (or the one named) through `Camera` exactly as an
app would, and prints the frame rate, the frame type and size, the tracklets with their keypoints
when the pose model's archive is in `data/models` (`--no-pose` leaves it out, `--two-models` runs
the detector plus the pose model on the crops), the readings
(exposure readback, tilt and roll from the IMU, the lens) at the end. A failure names the depthai
call, not an app.
"""

import argparse
import logging
import sys
import time

from .. import CameraFrame, CameraTracklet
from .camera import Camera
from .definitions import get_device_list, PoseChain
from .settings import Oak4CameraSettings

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='stream one OAK4 through the oak4 backend and report')
    parser.add_argument('device_id', nargs='?', default='', help='device id or IP; the first found when left out')
    parser.add_argument('-s', '--seconds', type=float, default=10.0)
    parser.add_argument('--color', action='store_true', help='colour frames (the warp then needs keystone, not tilt)')
    parser.add_argument('--square', action='store_true')
    parser.add_argument('--keystone', type=float, default=0.0)
    parser.add_argument('--tilt', type=float, default=0.0)
    parser.add_argument('--no-pose', action='store_true', help='frames only, no pose model on the device')
    parser.add_argument('--two-models', action='store_true', help='the detector and the pose model on each crop')
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(name)s: %(message)s')

    devices: list[str] = get_device_list(verbose=True)
    if not devices:
        logger.error('no device found: is the camera on the switch, with PoE+?')
        return 1
    settings = Oak4CameraSettings()
    settings.device_id = args.device_id or devices[0]
    settings.color = args.color
    settings.square = args.square
    settings.keystone = args.keystone
    settings.tilt = args.tilt
    settings.yolo = not args.no_pose
    settings.pose_chain = PoseChain.TWO_MODELS if args.two_models else PoseChain.ONE_MODEL
    settings.initialize()

    frames: list[CameraFrame] = []
    tracklets: list[list[CameraTracklet]] = []
    camera = Camera(settings)
    camera.add_frame_callback(frames.append)
    camera.add_tracklet_callback(lambda cam_id, message, timestamp: tracklets.append(message))
    camera.start()
    started: float = time.perf_counter()
    while time.perf_counter() - started < args.seconds and camera.is_alive():
        time.sleep(0.5)
        logger.info(f'{len(frames)} frames, {settings.readings.video_fps:.1f} fps, '
                    f'{len(tracklets)} tracklet messages, {settings.readings.tracker_fps:.1f}/s, '
                    f'exposure {settings.sensor.exposure} us iso {settings.sensor.iso}, '
                    f'tilt {settings.readings.tilt_measured:+.1f} roll {settings.readings.roll_measured:+.1f}')
    camera.stop()
    camera.join(5.0)

    if not frames:
        logger.error('no frame arrived')
        return 1
    first: CameraFrame = frames[0]
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
        logger.info('no tracklet arrived: nobody in view, or the pose model did not run')
    if camera.join is not None:
        logger.info(f'two-model chain: {camera.join.joined} frames joined, {camera.join.dropped} dropped incomplete')
    logger.info(f'lens: field {settings.readings.fov_factory:.1f} deg, {settings.readings.lens_error:.2f} deg off the preset lens; '
                f'tilt {settings.readings.tilt_measured:+.1f} roll {settings.readings.roll_measured:+.1f} deg')
    return 0


if __name__ == '__main__':
    sys.exit(main())
