# The camera package

What `modules/camera` is and why it works as it does. Recording and playback have their own document
([RECORDING.md](../recording/RECORDING.md)), the OAK4 its own ([OAK4.md](oak4/OAK4.md)).

## What it is for

One job: hand an app a camera. That is three things.

| Part         | What                                                                                     |
|--------------|------------------------------------------------------------------------------------------|
| the contract | `CameraSource`, and what it hands over: `CameraFrame`, `CameraTracklets`                 |
| the devices  | a backend per device: `oak`, `oak4`, `vimba`, `webcam`                                   |
| the geometry | what the frame's pixels mean: the sensor modes, the lens, the tilt, the depth's alignment |

## The contract

- A camera's callbacks are a `utils.Broadcast`: they run on the backend's own thread in registration order, a
  callback can be registered at any time, and one that raises is logged while the others still run.
- A frame or tracklet message is stamped once, at hand-over, with its capture where the backend knows it (the end
  of the exposure, oak4; the device's timestamp on depthai's clock, oak), else its arrival, on `perf_counter`; where
  the two differ it also carries its arrival, so its latency needs no clock read downstream. A depth frame and the video frame of one capture carry timestamps
  within `CAPTURE_TOLERANCE` (2 ms; on the OAK4 0.12 ms apart), so time pairs them. A depth frame's `confidence`
  rides on it, on the depth's own view, never as a frame of its own; a backend without one gives None.
- A camera hands its frames and its model's boxes and keypoints as delivered: the uploader straightens them on the
  GPU with the frame's `homography`, then turns them by `rotate90` (a transpose) and flips them by `flip_v` and
  `flip_h`, in that order; off the GPU `oriented_image` applies `rotate90` and the flips in the frame's own layout
  (NV12's luma as a plain array, its chroma as one pixel a pair), `bgr_frame` converts an NV12 frame (the OAK4's
  color) to BGR, and `oriented_corners` gives a drawing the same `rotate90` and flips. One
  `homography` array stands for one correction, so a consumer keeps what it derives from it until the array
  changes. A frame's `readings` and `intrinsics` are what a recording needs to redo the correction.
- The tracklet callback gets every message, an empty one too: no player is a tick as well; a camera without a
  detector never calls it.
- `connected` says whether the device is open now: a backend that keeps looking for its device (Vimba) is started
  without one and connects when it is there.
- A backend's settings extend `CameraSettings` or `RigCameraSettings` with its device's fields, so the settings file keys
  stay flat: `cam_0.fov` and `cam_0.readings.roll_offset` are the same keys for every rig backend. A backend whose
  device takes a field live redeclares it without `Field.INIT`; the panel shows the backend's fields first.
- `CameraSettings` holds the mount every camera has: `flip_h`, `flip_v` and `rotate90`, a camera on its side, which go
  on its frames as flags. The OAK-D bakes its flips into its warp, whose columns are the panoramic tracker's
  bearings, and applies no turn: it logs once that `rotate90` is not applied.
- Nothing in the show reads a rig camera's readings: they are for setting the rig up. A backend fills what its
  device measures and leaves the rest at its default, NaN for the angles. `roll_offset` sits with the readings,
  beside the roll it corrects.

## The sensor mode and the frame's size

One setting, the sensor mode (`resolution`, `CameraResolution`), sets every width, height, aspect and buffer size
in the pipeline and the render, with the mount. It is a setting and not a constant because everything
frame-relative in a settings file is tuned against one frame, and a constant in code with its dependants in a settings file is
two sources that can disagree; it also lets a recording play right: a clip shot at 720 rows runs through an 800-row
pipeline with `vfov`, the distance and the panorama all wrong, and a playback settings file saying `P720` fixes them at
once. Nothing downstream restates a frame dimension as a literal. It is `CameraResolution`, not `Resolution`,
because `modules/pose` has a `resolution` of its own and the apps import both.

| Mode    | Size        | Sensors                                                |
|---------|-------------|--------------------------------------------------------|
| `P720`  | 1280 x 720  | both                                                   |
| `P800`  | 1280 x 800  | both; the OV9282 W's full readout                      |
| `P1080` | 1920 x 1072 | color only; 1080 cropped to the warp's alignment of 16 |
| `P2160` | 3840 x 2160 | color only, the 4K mode; on the OAK4 USB only           |

Both sensors have `THE_720_P` and `THE_800_P` in depthai 2.30, which lets one label span them; mono also has 400,
480 and 1200 rows (the last for AR0234-class sensors) and color modes above 4K, none offered until something needs
them. A mono camera asked for `P1080` or `P2160` runs `P800` and says so once.

The delivered frame's height is derived, not set: its rows are tangents of elevation ([the rows](#the-rows)),
which spend rows at the top, so a tilted camera's full reach needs more rows than the sensor has: 1152 at `P800`
and tilt 16, 960 at `P720` and tilt 15 (`delivered_height`, printed by the open log).

Not derived from the mode, and re-tuned on the rig with it, because they are set by eye against the frame (P800 to
P720 scales them by 800/720 = 1.111): `track.height_filter` (a fraction of the frame's height) and
`track.foot_offset` (the detector's pad measures in pixels). `track.seam.*` is in degrees and fractions of a height,
and the vertical field and the panorama's row derive their own, so neither is in that list.

## The lens

Every figure below is for the sensor's full readout, true only in the mode that reads every line: a field paired
with another mode is wrong.

| Device      | Sensor                 | Native      | DFOV | HFOV | VFOV |
|-------------|------------------------|-------------|------|------|------|
| OAK-D Pro W | OV9282 mono pair       | 1280 x 800  | 150  | 127  | 79.5 |
| OAK-D Pro W | IMX378 color (stock)   | 4056 x 3040 | 120  | 95   | 72   |
| OAK-D Pro W | OV9782 color (upgrade) | 1280 x 800  | 150  | 127  | 79.5 |
| OAK-1 W     | IMX378 color           | 4056 x 3040 | 120  | 95   | 72   |
| OAK-1 W     | OV9782 color           | 1280 x 800  | 150  | 127  | 79.5 |

From [OAK-D Pro W](https://docs.luxonis.com/hardware/products/OAK-D%20Pro%20W) and
[OAK-1 W](https://docs.luxonis.com/hardware/products/OAK-1%20W); `get_device_list(verbose=True)` logs the sensor
behind each socket.

`fov` is the field the delivered frame spans, a contract the tracker builds on (`cam_fov`, the overlaps, a column's
azimuth). The lens is what the sensor sees, and it is not `fov`. The four White Space units' factory calibrations
(mono CAM_B, 1280 x 800) are equidistant to 0.5 % out to 75° off axis, at 565 to 575 px/rad where `fov = 127`
implies 577.5, with the optical center 5 to 24 px left of and about 10 px below the frame's center:

| Unit          | Focal px/rad | cx    | cy    | Field across 1280 px |
|---------------|--------------|-------|-------|----------------------|
| ...F124D9D600 | 575.4        | 615.2 | 409.9 | 127.8                |
| ...110AD3D200 | 567.2        | 634.3 | 407.6 | 128.8                |
| ...31DDD2D200 | 566.7        | 632.6 | 411.7 | 128.7                |
| ...1136D1D200 | 565.0        | 634.5 | 410.3 | 129.6                |

The lens taken as `fov` read bearings about 1° short toward the seams and put the horizon 10 px (1°) too high, a
meter of distance at 5 m. So the lens is three settings of its own, shared by an installation's cameras:
`lens_fov` (the field across the full sensor width, 0 the same as `fov`), `lens_center_x` and `lens_center_y` (the
optical center's offset from the frame's center in px, at the full sensor mode, the same under the 720-row center
crop). One shared lens and not one per unit is a trade: the mean leaves the outlier unit 1.4° off at its azimuth
zero, and `lens_deviation` reports each unit's residual at open. `fov` must not exceed the lens's field, or the
edge columns read outside the sensor and go black; `build_warp_mesh` warns.

`degrees_per_pixel` takes the width of the un-cropped frame, because `fov` is quoted for it: a later crop (the
square crop, the 1080 to 1072 trim) removes pixels without changing the scale.

## The rows

The rows are cylindrical: `horizon_px - row = focal * tan(elevation)`, with the columns' `focal`, so the frame is
locally a level pinhole camera panned to each column's bearing. That keeps a body's proportions true at any height,
for the pose model, which was trained on pinhole photographs; rows linear in elevation shrink a meter at 40°
elevation by 41 % and at 55° by 67 % against eye level.

The window is pinned at the bottom: the last row is the sensor's lowest reach on the center column, and the rows
run up from there, because the feet are what the tilt rule pins (the distance reads off the feet) and the tangent
spends rows at the top: 800 rows at tilt 16 reach 43°, 1152 rows the sensor's own 57°. The horizon may lie outside
the frame. Rows are tangents, so `(row - horizon_row) * k` is never an angle: a consumer turns a row into an
elevation through `FrameWindow`. With `flip_v` the delivered row of `horizon_px` is `out_h - 1 - horizon_px`.

## The warp

The lenses are equidistant: a pixel's distance from the optical center is proportional to its angle off the axis.
In that projection a column is one azimuth only on the horizontal center line; off it a vertical pole bows toward
the center, so a standing player's box center reads short of the bearing: 5° at 45° bearing 1.35 m out, 8° at the
seam. No correction on x alone fixes that. So the warp reads each output pixel as a bearing (the column) and a
tangent of elevation (the row), undoes the tilt, and reprojects into the equidistant source: every column is one
azimuth at every height, every row one elevation at every column.

It is not a homography: a pinhole lens uses the tangent on both axes, which makes a tilted view a trapezoid with
straight edges, but on an equidistant lens a tilt bends straight lines: at 15° on a 127° frame the center of a row
moves 151 px and its ends 74 px, a 77 px bow. The warp node interpolates linearly between mesh points, so 2 columns
express only a straight source line per row, the homography's family; about 16 is already sub-pixel, and 32 x 32
differs from 64 x 64 by under 0.3 px at 30°. The error grows with the frame's width and falls with the square of the
mesh count, so 32 holds 0.23 px on a 4056 px frame.

Where the sensor did not look, the output is black: the sensor is a rectangle in the lens's projection and not in
bearing and elevation. With the window pinned at the bottom, the center column is covered top to bottom and the
top goes black toward the sides in an arch, because a sideways ray turned about the horizontal axis rises by less
than the tilt. `frame_coverage` gives the covered rows per column, what the tracker and the panorama judge a raised
arm near a seam against; a column at the very edge of the field can lose a pixel near the horizon where the lens's
reach falls short of `fov`, which `_report_lens_reach` logs at open. The fixed point is the lens's center: the
center column at the row of elevation `tilt` reads `(cx, cy)`, so at tilt 0 the warp is not the identity but the
equidistant to cylindrical reprojection.

## The mount readout

Every camera constant is measured with a tape (the camera's radius, the lens's height) or read off a datasheet
(`fov`), except two: `tilt`, a hand adjustment on a tripod head, and roll, which nothing models: the warp takes a
tilt and assumes the camera level about its optical axis. A rolled camera tilts the horizon, which in the panorama
looks exactly like a wrong `tilt`; the IMU tells them apart. It is a setup aid, a few hertz, every value allowed to be
missing, and nothing in the show reads it.

`orientation_from_gravity` is separable: pure tilt gives roll 0 and pure roll tilt 0 at any size, so a mount error is
put on one axis. The IMU's vector reaches the camera's frame through the rotation of the device's IMU-to-camera
calibration (a missing matrix passes the vector as it is, right where the frames agree) and the board's mounting
turn about the optical axis (`IMU_BOARD_ROLL`, measured on the OAK-D; turning by the angle maps a roll φ to φ minus
the angle). `geometry.MountReadout` does this for both OAK backends: each sample into the camera's frame, an
exponential average (`IMU_SMOOTHING`) at `IMU_RATE_HZ`, since a tripod does not move and one sample is mostly noise,
and its tilt and its roll less `roll_offset` into the readings. Each backend passes its `IMU_BOARD_ROLL` and its
board's extrinsics.

## The tilt correction

A tilt is a rotation about the camera, so where a pixel lands does not depend on its distance: one homography
straightens a frame and the depth aligned to it. It holds for a rectilinear frame only, so an undistorted one. The
picture is scaled so its height fills the frame's and centered across it, tilt 0 is the identity, and the third
coordinate the homography gives a pixel is its depth scale: a distance along the tilted axis times it is the distance
along the level one. The tilt must move `TILT_STEP` (0.25°) before the frames are straightened anew: the reading is
noisy, and a picture that followed it would shimmer. How the OAK4 uses it is in [OAK4.md](oak4/OAK4.md#tilt-correction).

## The depth's alignment

Depth comes from a stereo camera a few centimeters beside the video camera, so a point lands in the video frame
shifted by an amount that depends on its distance. The GPU moves each depth pixel there and keeps the nearest where
two land on one pixel; the background a near object uncovers stays empty, because the depth camera never saw it.

## The clip camera

`oak.ClipCamera` sends recorded clips through the OAK-D's own pipeline, so the device's detector and tracker run on
them as on the rig. A clip that is warped (`clip_warp`, for raw clips only, unwarped at tilt 0) feeds the warp's
input and must be the sensor's size; one that passes through must be the delivered size, its derived height
included. A clip of another size still runs (the XLinkIn's size is a ceiling, the frame carries its own size, the
detector resizes what comes), but every frame-relative setting is off by the ratio: the tracker's vfov and height
gates, the pose distance thresholds. The fix is in the settings file, `camera.resolution` set to the clip's label; the
clip camera warns once per size.

## Vimba

The Vimba backend runs one Allied Vision Mako G-125C over GigE through the VimbaPython binding, vendored in
`vimba/vimbapython/`, which needs the Vimba 6 SDK (`VIMBA_HOME`, VimbaC). Its thread keeps the camera: `start()`
makes the first try, the thread searches again every `RETRY_INTERVAL` while there is none, and opens it again (user
set, packet size, the panel's values) when no complete frame came for `FRAME_TIMEOUT`; `connected` follows it. The
frames come on Vimba's own thread, which copies them, marks them with the settings' `rotate90` and flips as
flags, stamps them with their arrival and hands them over; the camera's thread sends the panel's values to the camera and watches that frames arrive.
Without the SDK the binding cannot load: `start()` logs an error and the camera stays unconnected, so an app with
another image source still runs.

The panel is the source of truth: every feature is sent once after the camera opens, over what its user set
loaded. The rate and the flips are live: the camera takes a new frame rate at any time (`AcquisitionFrameRateAbs`)
and the flips are flags on the frame, applied where it is shown. A feature the camera model does not have is a warning, once; a write the
camera refuses (a value it does not take, a control channel that timed out) or a failed read is a warning once per
reason, and is tried again, not switched off. Exposure and gain each have auto, a manual value sent while auto is
off, a limit for auto and a readout of what the camera uses; both autos aim for `target_brightness`, exposure
first and gain after it. `measure_balance` lets the camera measure its white balance once and takes the ratios into the
panel.

## The cameras' health

`CameraHealth` turns the cameras' readings into verdicts pinned on the panel, because the readings themselves sit
on each camera's settings, deep in the panel, where a readout goes to be ignored; which camera and value is off is
drawn on that camera's own view (`CameraReadingsDraw`).

| Verdict      | Holds when                                                                      |
|--------------|---------------------------------------------------------------------------------|
| `mount`      | every tilt and roll deviation (`mount_deviation`) is within `mount_tolerance`   |
| `camera_fps` | every camera's frame rate is within `FPS_TOLERANCE` (5 %) of its `fps`          |
| `video_rate` | every camera's momentary rate (1 / its worst frame gap) is above half its `fps` |

- A camera without an IMU is left out of `mount`: that is the board, nothing to fix on site; its view says `no IMU`.
- The tilt deviation is against the tilt the warp was built with, so a camera off it is straightened by the wrong
  amount; there is no configured roll, as the warp has no roll term, so any roll is a deviation.
- A camera that delivers no frames reads 0 and fails `camera_fps`.
- `video_rate` catches what the average hides: single late frames, which a viewer sees as stutter.
- The slowest camera's latencies and each link are published without a verdict.
- `fov_factory` and `lens_error` are not judged: they describe the build, not how the tripod was set this
  morning, and a warning at every launch teaches the operator to ignore warnings.

## The OAK-D

**The IMU's turn.** The OAK-D's IMU sits a quarter turn about the optical axis, which
`getImuToCameraExtrinsics` does not encode (`oak.imu.IMU_BOARD_ROLL`, -90). It is measured: on the rig all four
cameras read a roll of -90° with a correct tilt, and only a turn about the optical axis does that, since it leaves
`gz` and `hypot(gx, gy)` alone, so the tilt survives and the roll moves by a constant. It is a property of the board,
so a constant and not a setting: four tripods set apart cannot agree on a quarter turn by chance. It claims the
quarter turn and nothing more: a spirit level left a residual of about -0.85° on three of four units, which is not
folded in, because -90 is the board's layout and -0.85 the mean of three samples. Below a degree is per unit and
belongs in `roll_offset`; the fourth unit needs -2.25 sitting level, a sensor fault, probably from a fall, which no
shared constant could absorb.

**The factory calibration.** Its model is a pinhole with a rational distortion polynomial (`Perspective`, 14
coefficients); its `fx` is the scale at the optical axis, and on these lenses the distortion keeps radius linear in
angle to 0.5 % out to 75°, so `fx` is the equidistant focal and `(cx, cy)` the optical center. `getCameraIntrinsics`
at the mode's size accounts for the 720-row center crop (it moves `cy` by 40). `lens_field` of `fx` tells the two
sensor variants apart (129° against about 97°).

**The IMU.** The pipeline reads `ACCELEROMETER_RAW` and not `GRAVITY`: gravity is a fused output that needs the
IMU's own DSP, which the BNO086 boards have and the BMI270 ones do not; a camera on a tripod is still, so the
averaged raw accelerometer is the gravity vector, on every board. A few hertz is enough, as nothing in the show
reads it. The node is always created; a board without an IMU never sends, and the camera subscribes only when
`getConnectedIMU` names a sensor.

**Tilt or keystone.** A camera uses one correction. The warp ([the warp](#the-warp)) makes a column one azimuth,
also at tilt 0, where it is the equidistant to cylindrical reprojection. The keystone (`keystone`, a fraction of the
frame) is a homography that pins the output's four corners to the frame's: a rotation composed with a stretch to
fit, the wrong model for re-aiming a wide lens but what hd_trio relies on, a player seen from a down-tilted camera
in normal proportions with the whole frame kept. Its mesh is 2 x 64: two columns, since the map is affine in x per
row, and 64 rows for the vertical curve; it does not change without the installations that use it. The two are
exclusive, since a keystone spreads columns by height, which the tilt removes, and the tilt shifts the frame, which
the keystone avoids; both set warns and takes the tilt.

**Raw clips.** A clip comes off the video output, after the warp, so it carries the flips and the square crop it
was shot with. A raw clip (`clip_warp`: equidistant, shot at tilt 0, before the warp reprojected) gets only the
difference: no flips, the clip as the source at its own size and the delivered size out, the tilt's sign turned
under `flip_v` (a vertical mirror reverses a rotation about the horizontal axis, a horizontal one commutes with
it), and the lens center mirrored with the pixels. A clip recorded since the warp is reprojected already and would
be reprojected twice; the geometry a clip was shot with is not stored with it.

**The lens's reach.** `fov` is the frame's span, and the lens must reach half of it on each side of its own center,
which is not the frame's; where it falls short, the edge columns near the horizon read past the sensor and go black.
A pixel or two is the price of a shared lens center and is logged as information; more is a settings file to fix, and
warns.

**The detector.** The wide blobs are built on tools.luxonis.com from the stock ultralytics yolov8n weights,
OpenVINO 2022.1, at 512 x 448 for the tall cylindrical frame (1280 x 1152 at P800 is 1.111; 512 x 448 is 1.143; the
P720 frame of 1280 x 960 is 1.333, a 17 % mismatch a 512 x 384 blob would close). They have the pixel count of the
640 x 352 they replaced (229 k against 225 k), so the same inference cost, confirmed on the rig with `tracker_fps` at
the input rate on all four cameras. The 640 x 352 files and the 5-shave builds stay in `data/models` for reference;
nothing selects them. The frame is stretched into the blob (`setKeepAspectRatio(False)`), never cropped or
letterboxed: nothing at the frame's edges goes undetected, and the detections' normalized coordinates map 1:1 onto
the full frame, so an aspect mismatch costs detection quality, never geometry, and the tracker's azimuths are right
whatever blob is in use. The input size is read off the blob, so a file name and a size cannot disagree.

## The package's parts

| Part                 | Holds                                                                                        |
|----------------------|----------------------------------------------------------------------------------------------|
| `source.py`          | `CameraSource`, `ClipSource` and their callbacks                                             |
| `frame.py`           | `CameraFrame`, `FrameType`, `FrameReadings`, `DepthAlignment`                                |
| `tracklet.py`        | `CameraTracklet`, `CameraTracklets`                                                          |
| `settings.py`        | `CameraSettings`, `RigCameraSettings` and their readings                                     |
| `camera_health.py`   | `CameraHealth`, `CameraHealthSettings`                                                       |
| `capture_match.py`   | `CaptureStream`, `MissedParts`: the parts of a capture matched by time, for the syncs        |
| `depthai_backend.py` | `oak_backend`, `require_backend`                                                             |
| `geometry/`          | `sensor_modes`, `lens`, `frame_window`, `warp`, `orientation`, `tilt_correction`, `mirror`,  |
|                      | `depth_projection`                                                                           |
| `oak/`               | the OAK-D: `camera`, `clip_camera`, `control`, `pipeline`, `settings`, `sensor`, `device`,   |
|                      | `imu`, `models`                                                                              |
| `oak4/`              | the OAK4: `camera`, `control`, `pipeline`, `settings`, `sensor`, `device`, `imu`,            |
|                      | `messages`, `models`, `keypoint_join`                                                        |
| `vimba/`, `webcam/`  | the Mako over GigE, the PC's own camera                                                      |

The package's `__init__.py` exports what other packages use; a backend takes the geometry from `geometry`'s.
Recording and playback, the replay camera among them, are `modules/recording`, which uses a camera; the cameras'
sync (`CameraSync`) is `modules/inference`'s, beside `InferenceFrameSync`.

## What no settings file runs

Kept, and listed so a reader knows what is live:

| Code                             | Runs when                                            | In a settings file |
|----------------------------------|------------------------------------------------------|-------------|
| `KeypointJoin`, `add_pose_chain` | OAK4 `model_chain` is the detector plus a pose model | none        |
| `add_one_model_chain`            | OAK4 `model_chain` is one pose model                 | none        |
| `vimba`, `webcam`                | smile's image source                                 | smile       |
| `oak.ClipCamera`                 | hd_trio and white_space in -sim, `replay` off        | both        |

