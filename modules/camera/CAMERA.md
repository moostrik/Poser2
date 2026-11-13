# The camera package

What `modules/camera` is for, what is wrong with it now, and the plan to make it clear and concise. Recording and
playback have their own document ([RECORDING.md](../recording/RECORDING.md)), the OAK4 its own
([OAK4.md](oak4/OAK4.md)).

## What it is for

One job: hand an app a camera. That is three things.

| Part         | What                                                                              |
|--------------|-----------------------------------------------------------------------------------|
| the contract | `CameraSource`, and what it hands over: `CameraFrame`, `CameraTracklets`          |
| the devices  | a backend per device: `oak`, `oak4`, `vimba`, `webcam`                            |
| the geometry | what the frame's pixels mean: the sensor modes, the lens, the tilt, the depth's alignment |

Everything else in the package is there because it was made beside a camera, not because it is one.

## The sensor mode and the frame's size

One setting, the sensor mode (`resolution`, `CameraResolution`), sets every width, height, aspect and buffer size
in the pipeline and the render, with the mount. It is a setting and not a constant because everything
frame-relative in a preset is tuned against one frame, and a constant in code with its dependants in a preset is
two sources that can disagree; it also lets a recording play right: a clip shot at 720 rows runs through an 800-row
pipeline with `vfov`, the distance and the panorama all wrong, and a playback preset saying `P720` fixes them at
once. Nothing downstream restates a frame dimension as a literal. It is `CameraResolution`, not `Resolution`,
because `modules/pose` has a `resolution` of its own and the apps import both.

| Mode    | Size        | Sensors                                                |
|---------|-------------|--------------------------------------------------------|
| `P720`  | 1280 x 720  | both                                                   |
| `P800`  | 1280 x 800  | both; the OV9282 W's full readout                      |
| `P1080` | 1920 x 1072 | color only; 1080 cropped to the warp's alignment of 16 |

Both sensors have `THE_720_P` and `THE_800_P` in depthai 2.30, which lets one label span them; mono also has 400,
480 and 1200 rows (the last for AR0234-class sensors) and color 4K upward, none offered until something needs
them. A mono camera asked for `P1080` runs `P800` and says so once.

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
metre of distance at 5 m. So the lens is three settings of its own, shared by an installation's cameras:
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
for the pose model, which was trained on pinhole photographs; rows linear in elevation shrink a metre at 40°
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
the angle).

## The tilt correction

A tilt is a rotation about the camera, so where a pixel lands does not depend on its distance: one homography
straightens a frame and the depth aligned to it. It holds for a rectilinear frame only, so an undistorted one. The
picture is scaled and moved so all of it stays in the frame, tilt 0 is the identity, and the third coordinate the
homography gives a pixel is its depth scale: a distance along the tilted axis times it is the distance along the
level one. The tilt must move `TILT_STEP` (0.25°) before the frames are straightened anew: the reading is noisy, and
a picture that followed it would shimmer. How the OAK4 uses it is in [OAK4.md](oak4/OAK4.md#tilt-correction).

## The depth's alignment

Depth comes from a stereo camera a few centimetres beside the video camera, so a point lands in the video frame
shifted by an amount that depends on its distance. The GPU moves each depth pixel there and keeps the nearest where
two land on one pixel; the background a near object uncovers stays empty, because the depth camera never saw it.

## The cameras' health

`CameraHealth` turns the cameras' readings into verdicts pinned on the panel, because the readings themselves sit
on each camera's settings, deep in the panel, where a readout goes to be ignored; which camera and value is off is
drawn on that camera's own view (`CameraReadingsLayer`).

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

**The detector.** The wide blobs are built on tools.luxonis.com from the stock ultralytics yolov8n weights,
OpenVINO 2022.1, at 512 x 448 for the tall cylindrical frame (1280 x 1152 at P800 is 1.111; 512 x 448 is 1.143; the
P720 frame of 1280 x 960 is 1.333, a 17 % mismatch a 512 x 384 blob would close). They have the pixel count of the
640 x 352 they replaced (229 k against 225 k), so the same inference cost, confirmed on the rig with `tracker_fps` at
the input rate on all four cameras. The 640 x 352 files and the 5-shave builds stay in `data/models` for reference;
nothing selects them. The frame is stretched into the blob (`setKeepAspectRatio(False)`), never cropped or
letterboxed: nothing at the frame's edges goes undetected, and the detections' normalized coordinates map 1:1 onto
the full frame, so an aspect mismatch costs detection quality, never geometry, and the tracker's azimuths are right
whatever blob is in use. The input size is read off the blob, so a file name and a size cannot disagree.

## What is wrong

Counted on 2026-11-12, tests and the vendored `vimbapython` left out.

| Fault                                    | Where                                                                                     |
|------------------------------------------|-------------------------------------------------------------------------------------------|
| Things that are not a camera             | a tracker (`replay/tracker.py`), a trigger for the apps' tracker and poses (`sync/frame_trigger.py`), the host's poses (`FramePoses`, `PoseArrays`), a recorder and a player, two command-line scripts (`oak4/probe.py`, `oak4/models.py`) |
| Files that hold several subjects         | `geometry/geometry.py`: 756 lines, 40 functions, 6 subjects under comment banners; `oak/definitions.py` (215) and `oak4/definitions.py` (366): modes, ranges, devices, models, IMU; `settings.py`: 6 classes of 4 owners |
| Names that do not say what the thing does | `CameraCheck`, `PoseJoin`, `definitions`, `Simulator` beside `replay.Camera`, three `...Readings` |
| A settings class away from its class     | `CameraCheckSettings`, `TiltCorrectionSettings` in `settings.py`                          |
| A public boundary that exports everything | `__init__.py` exports 94 names: 26 are used by nothing outside the package's core, 20 only by its own backends |
| Docstrings and comments as essays        | `geometry.py` is 44% docstring and comment (330 of 756 lines), with history in it; `CameraFrame`'s docstring is 20 lines |
| The same names twice                     | `oak` and `oak4` each define `get_device_list`, `read_lens_calibration`, `imu_rotation_to_camera`, `log_connected_sensors`, `SensorControl`, and 14 constants, with unlike bodies |
| Code no preset runs, unmarked            | `PoseJoin` and the two pose chains on the OAK4 (deep_flow runs the detector alone), `vimba` and `webcam` (smile only) |

What each of the unclear names is, from the code:

| Name             | What it does                                                                                     |
|------------------|--------------------------------------------------------------------------------------------------|
| `CameraTracklet` | what the camera's own tracker says of one player in one frame: id, status, box, age, keypoints   |
| `PoseJoin`       | OAK4, detector-plus-pose chain only: puts the pose model's keypoints on the tracklets of the same frame |
| `CameraCheck`    | turns the cameras' readings into the panel's verdicts: mounted level, at its frame rate, no stutter |
| `FrameTrigger`   | fires once per round of the cameras' frames, to start hd_trio's and white_space's tracker and poses |
| `Simulator`      | oak: a recording sent through the device's own pipeline, so the device's detector finds the players |
| `replay.Camera`  | a recording handed on as a camera without a device                                               |

## The plan

Form only: no behavior changes, nothing with no caller is removed. In this order, a commit each, the guard and
import tests after each, the full suite once at the end.

### 1. Recording leaves the camera

Recording is not a camera: it uses one. `modules/recording` becomes its own package, with the replay camera in it,
and depends on `modules/camera`, never the other way.

```
modules/recording/
    RECORDING.md
    container/      chunk.py, recording_folder.py
    recorder/       recorder.py, recording_sync.py
    player/         player.py, play_clock.py, chunk_readers.py, frame_hand_over.py
    replay/         camera.py        the camera of a recording
```

- `oak.Simulator` takes its clips by a protocol in the contract (`ClipSource`: `add_frame_callback`, its callback
  typing the metadata as `object`, as it does now), and its two options become its own settings:
  `PlayerSettings.sim_passthrough` and `apply_warp` move to `OakCameraSettings`, in every preset that has them
  (deep_flow ×2, hd_trio ×2, white_space ×2). So `modules/camera` does not import `modules/recording`.
- `capture_match.py` (`CaptureStream`, `MissedParts`) stays in the camera, at its root: it matches a camera's parts
  by capture time, for the recorder and for `modules/inference`.
- In the same commit: `replay` and the recording names in the three apps' imports; `FrameMetadata`, `FramePoses`
  and `PoseArrays` in `modules/inference/recorded_poses.py`; the links to RECORDING.md; and OAK4.md's section
  "Recording and replay", cut to what is the OAK4's and a link to RECORDING.md.
- The guard test's list of camera backends loses `replay`. `CLAUDE.md` names no camera file, so it stays as it is.
- RECORDING.md's tables of the player's parts and of the layout follow the move.

### 2. What is not a camera goes where it belongs

| What                         | From                              | To                                                    |
|------------------------------|-----------------------------------|-------------------------------------------------------|
| What                       | From                                | To                                                |
|----------------------------|-------------------------------------|---------------------------------------------------|
| the software tracker       | `camera/replay/tracker.py`          | `modules/tracker`, as `SortTracker`               |
| `FrameTrigger`             | `camera/sync/frame_trigger.py`      | `modules/inference`, as `CameraSync`              |
| the OAK4's two scripts     | `camera/oak4/probe.py`, `models.py` | `scripts/oak4_probe.py`, `scripts/oak4_models.py` |
| `FramePoses`, `PoseArrays` | the recorder's files                | `modules/recording`, with step 1                  |

- `SortTracker` is a Kalman filter and overlap assignment, SORT's shape. The replay camera's `BoxTracker` protocol
  keeps its name.
- `CameraSync` decides when the cameras are in sync: every camera's newest frame (or tracklet message) as close
  together as they come, and fires with that moment. It asks of the cameras what `CaptureSync` asks of one
  capture's parts, and both streams it watches pass the input stage, so it sits beside `CaptureSync`. Its two
  instances keep their stream as qualifier, `frame_sync` and `tracklet_sync`, in hd_trio's and white_space's
  settings and presets; its callback is `add_sync_callback`.
- OAK4.md's commands follow the scripts.

The `sync` folder is gone with it.

### 3. One subject per file

| File now                   | Becomes                                                                                         |
|----------------------------|-------------------------------------------------------------------------------------------------|
| `geometry/geometry.py`     | `sensor_modes.py` (resolutions, frame sizes), `lens.py` (the lens, the warp mesh), `coverage.py` (the frame's window and coverage, for the tracker and the render), `orientation.py` (gravity to tilt and roll), `straighten.py` (the tilt's homography, boxes and keypoints through it, `TiltCorrection`), `mirror.py`, `depth_projection.py` |
| `oak/definitions.py`, `oak4/definitions.py` | in each backend: `sensor.py` (modes, control ranges), `device.py` (finding it, its link, its calibration), `imu.py`, `models.py` (the model chain, the tracker's constants) |
| `settings.py`              | `CameraSettings`, `RigCameraSettings` and their readings stay; `TiltCorrectionSettings` goes beside `TiltCorrection`, the health's settings beside the health |

### 4. Names

| Now                                   | Becomes                                   | Why                                                    |
|---------------------------------------|-------------------------------------------|--------------------------------------------------------|
| `CameraCheck`, `CameraCheckSettings`, the `camera_check` group | `CameraHealth`, `CameraHealthSettings`, `camera_health` | it reports whether the cameras are healthy; "check" names an act, not a thing |
| `PoseJoin` (`pose_join.py`)           | `KeypointJoin` (`keypoint_join.py`)       | it joins keypoints to tracklets; a pose is the host's  |
| `oak.Simulator`                       | `oak.ClipCamera`                          | a camera fed clips; "simulator" says nothing of what is simulated |
| `FrameReadings`                       | stays                                     | per frame; the panel's are `CameraReadings`            |

The `camera_check` group is renamed in the five presets that have it (deep_flow ×2, hd_trio ×2, white_space ×1),
with deep_flow's `_CameraCheckSettings` and `modules/render`'s `CameraReadingsLayer`. `Simulator` is renamed in
hd_trio's and white_space's imports.

### 5. The public boundary

`modules/camera/__init__.py` exports what another package imports, counted after steps 1 to 3. A backend imports
from the camera package's `__init__.py` and from its subpackages' (`from ..geometry import ...`), never from their
files, so `geometry/__init__.py` exports what the backends use.

### 6. Docstrings and comments

A docstring says what the thing is. The reasons move here, into
this document's design sections (the lens model, the warp, the frame's contract), and the history is dropped: it is
in the commits. `geometry.py`'s 330 lines of prose and `CameraFrame`'s 20 are the first to go.

### 7. What no preset runs

Kept. Listed here so a reader knows what is live:

| Code                                         | Runs when                                           | In a preset           |
|----------------------------------------------|-----------------------------------------------------|-----------------------|
| `KeypointJoin`, `add_pose_chain`             | OAK4 `model_chain` is the detector plus a pose model | none                 |
| `add_one_model_chain`                        | OAK4 `model_chain` is one pose model                | none                  |
| `vimba`, `webcam`                            | smile's image source                                | smile                 |
| `oak.ClipCamera`                             | hd_trio and white_space in -sim, `replay` off       | both                  |

## The package after it

```
modules/camera/
    CAMERA.md
    __init__.py
    source.py            CameraSource, ClipSource
    frame.py             CameraFrame, FrameType, FrameReadings, DepthAlignment
    tracklet.py          CameraTracklet, CameraTracklets
    settings.py          CameraSettings, RigCameraSettings and their readings
    camera_health.py     CameraHealth, CameraHealthSettings
    capture_match.py     CaptureStream, MissedParts
    depthai_backend.py   oak_backend
    geometry/            sensor_modes, lens, coverage, orientation, straighten, mirror, depth_projection
    oak/                 camera, clip_camera, control, pipeline, settings, sensor, device, imu, models
    oak4/                camera, control, pipeline, settings, sensor, device, imu, models, keypoint_join
    vimba/  webcam/
```

## The rules it touches

- A package is one thing, and its name is that thing: steps 1 and 2
- Its `__init__.py` holds what other packages use: step 5
- Its files each hold one class or one subject, named after it: step 3
- Its tests sit in its `tests/` folder: the tests move with their files
- Command-line scripts sit in the repo's `scripts/` folder: step 2
- Similar things share the noun and differ by a qualifier: step 2, `SortTracker` and `CameraSync`- A docstring says what the thing is; reasons sit in the package's document, history in the commit: step 6
- Import from a package's `__init__.py`, never from its internal files: step 5
- Keep presets and settings schemas in sync with code changes: steps 1 and 4
- `modules/camera/` is the camera contract: the guard test's backends lose `replay`
- A plan step that is checked by a run or a test is one commit: each step, checked by the import and guard tests
