"""The depthai 3 pipeline of one OAK4 camera, built in one pass on the camera's own device.

The sensor at the requested size and type, the correction that turns the sensor's frame into the
contract's, and the IMU. Two corrections, as on the OAK-D: `tilt` (and tilt 0, the cylindrical
reprojection that makes a column one azimuth) goes through the `Warp` node with the mesh from
`geometry.warp_mesh_points`, which on this chip warps grey frames only; `keystone` (hd_trio,
deep_flow) is a four-point transform in `ImageManip`, in colour or grey. The detector branch
comes with the model archive (plan step 9.3). Nothing here is an XLink stream: the host reads a
node's output through the queue `Camera` creates on it.
"""

from dataclasses import dataclass
import logging

import depthai as dai

from .. import CameraResolution, frame_size, WARP_MESH, warp_mesh_points, keystone_corners, square_crop

from .definitions import SENSOR_SOCKET, frame_type, requested_size, IMU_RATE_HZ

logger = logging.getLogger(__name__)


@dataclass
class MountConfig:
    """How a camera is mounted and what its lens is, as the correction needs it: the OAK-D's
    `WarpConfig`, under the name of what it holds. `tilt` and `keystone` are exclusive per camera
    (`add_correction`); `fov_h` is the azimuth span of the delivered frame, quoted for the
    un-cropped sensor width; the lens fields are `geometry.source_lens`'s."""
    flip_h: bool
    flip_v: bool
    tilt: float
    keystone: float
    fov_h: float
    lens_fov: float = 0.0
    lens_center_x: float = 0.0
    lens_center_y: float = 0.0

    @property
    def lens_center(self) -> tuple[float, float]:
        return (self.lens_center_x, self.lens_center_y)


@dataclass
class PipelineOutputs:
    """What the host reads and writes: the delivered frames, the sensor's control input, the IMU."""
    video: dai.Node.Output
    control: dai.Node.Input
    imu: dai.Node.Output


def build_pipeline(
    pipeline: dai.Pipeline,
    fps: float,
    color: bool,
    square: bool,
    resolution: CameraResolution,
    mount: MountConfig,
    frame_height: int = 0,
) -> PipelineOutputs:
    """Build the camera's pipeline into `pipeline`, the one bound to its device.

    `frame_height` is the delivered height the tilt correction produces (`delivered_height`),
    0 for the sensor's rows. Raises `ValueError` for a colour frame with `tilt`, which the chip
    cannot warp: the camera then does not open, rather than deliver a frame the tracker would
    read wrong.
    """
    src_size: tuple[int, int] = requested_size(color, resolution)
    out_size: tuple[int, int] = frame_size(color, resolution, square, frame_height)
    logger.info("OAK4 pipeline: %s, %s, %s", 'square' if square else 'wide', 'color' if color else 'grey',
                'keystone' if mount.keystone != 0.0 and mount.tilt == 0.0 else 'tilt')

    camera: dai.node.Camera = pipeline.create(dai.node.Camera).build(SENSOR_SOCKET)
    source: dai.Node.Output = camera.requestOutput(src_size, frame_type(color), fps=fps)
    frames: dai.Node.Output = add_correction(pipeline, source, src_size, out_size, color, mount)
    imu: dai.Node.Output = add_imu(pipeline)
    return PipelineOutputs(video=frames, control=camera.inputControl, imu=imu)


def add_correction(pipeline: dai.Pipeline, source: dai.Node.Output, src_size: tuple[int, int],
                   out_size: tuple[int, int], color: bool, mount: MountConfig) -> dai.Node.Output:
    """The one correction a camera uses, picked as `oak.pipeline.build_warp_mesh` picks it: both
    set warns and takes `tilt`; both zero is the tilt path, which is not the identity but the
    reprojection; only a non-zero `keystone` is the four-point transform."""
    if mount.tilt != 0.0 and mount.keystone != 0.0:
        logger.warning("tilt (%.1f) and keystone (%.2f) are both set; they are exclusive, using tilt",
                       mount.tilt, mount.keystone)
    if mount.tilt != 0.0 or mount.keystone == 0.0:
        if color:
            raise ValueError('the OAK4 warps grey frames only: tilt needs color off, or keystone instead')
        return add_warp(pipeline, source, src_size, out_size, mount)
    return add_keystone(pipeline, source, src_size, out_size, color, mount)


def add_warp(pipeline: dai.Pipeline, source: dai.Node.Output, src_size: tuple[int, int],
             out_size: tuple[int, int], mount: MountConfig) -> dai.Node.Output:
    """The warp from the sensor's frame to the delivered one, the mesh from the shared geometry."""
    mesh: list[tuple[float, float]] = [(float(x), float(y)) for x, y in warp_mesh_points(
        src_size, out_size, src_size[0], mount.fov_h, mount.tilt, mount.flip_h, mount.flip_v,
        WARP_MESH, WARP_MESH, mount.lens_fov, mount.lens_center)]
    warp: dai.node.Warp = pipeline.create(dai.node.Warp)
    warp.setMaxOutputFrameSize(out_size[0] * out_size[1])
    warp.setOutputSize(*out_size)
    warp.setWarpMesh(mesh, WARP_MESH, WARP_MESH)
    source.link(warp.inputImage)
    return warp.out


def add_keystone(pipeline: dai.Pipeline, source: dai.Node.Output, src_size: tuple[int, int],
                 out_size: tuple[int, int], color: bool, mount: MountConfig) -> dai.Node.Output:
    """The keystone as a four-point transform: the frame's corners pinned to the points the
    OAK-D's homography pins (`geometry.keystone_corners`), the square cut from the centre first."""
    manip: dai.node.ImageManip = pipeline.create(dai.node.ImageManip)
    initial: dai.ImageManipConfig = manip.initialConfig
    if out_size[0] == src_size[0]:                          # wide: the whole frame
        src, dst = keystone_corners(src_size[0], src_size[1], mount.keystone, mount.flip_h, mount.flip_v)
    else:                                                   # square: cut from the centre
        size: int = out_size[1]
        initial.addCrop(*square_crop(src_size, size))
        src, dst = keystone_corners(size, size, mount.keystone, mount.flip_h, mount.flip_v, offset_base=src_size[1])
    initial.addTransformFourPoints([dai.Point2f(x, y) for x, y in src], [dai.Point2f(x, y) for x, y in dst], False)
    initial.setOutputSize(out_size[0], out_size[1], dai.ImageManipConfig.ResizeMode.STRETCH)
    initial.setFrameType(frame_type(color))
    manip.setMaxOutputFrameSize(out_size[0] * out_size[1] * (3 if color else 1))
    source.link(manip.inputImage)
    return manip.out


def add_imu(pipeline: dai.Pipeline) -> dai.Node.Output:
    """A slow accelerometer feed, so the camera can report how it is mounted. Raw rather than the
    fused gravity, averaged on the host: a camera on a tripod is static, so the average is the
    gravity vector, and this needs nothing of the IMU's own fusion."""
    imu: dai.node.IMU = pipeline.create(dai.node.IMU)
    accelerometer = dai.IMUSensorConfig()
    accelerometer.sensorId = dai.IMUSensor.ACCELEROMETER_RAW
    accelerometer.reportRate = IMU_RATE_HZ
    imu.enableIMUSensor(accelerometer)
    imu.setBatchReportThreshold(1)
    imu.setMaxBatchReports(10)
    return imu.out
