# The camera package

What `modules/camera` is for, what is wrong with it now, and the plan to make it clear and concise. Recording and
playback have their own document ([RECORDING.md](recording/RECORDING.md)), the OAK4 its own
([OAK4.md](oak4/OAK4.md)).

## What it is for

One job: hand an app a camera. That is three things.

| Part         | What                                                                              |
|--------------|-----------------------------------------------------------------------------------|
| the contract | `CameraSource`, and what it hands over: `CameraFrame`, `CameraTracklets`          |
| the devices  | a backend per device: `oak`, `oak4`, `vimba`, `webcam`                            |
| the geometry | what the frame's pixels mean: the sensor modes, the lens, the tilt, the depth's alignment |

Everything else in the package is there because it was made beside a camera, not because it is one.

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
- The guard test's list of camera backends loses `replay`; `CLAUDE.md`'s project-shape and test lines follow, on the
  user's word.

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
- `CameraSync` decides when a round is complete: one frame (or tracklet message) of every camera, the closest
  together they come. It asks of the cameras what `CaptureSync` asks of one capture's parts, and both streams it
  watches pass the input stage, so it sits beside `CaptureSync`. Its two instances keep their stream as qualifier,
  `frame_sync` and `tracklet_sync`, in hd_trio's and white_space's settings and presets; its callback is
  `add_round_callback`; "round" goes into the glossary, on the user's word.
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
- Similar things share the noun and differ by a qualifier: step 2, `SortTracker` and `CameraSync`
- Use the terms in `.claude/rules/glossary.md`: step 2 adds "round"
- A docstring says what the thing is; reasons sit in the package's document, history in the commit: step 6
- Import from a package's `__init__.py`, never from its internal files: step 5
- Keep presets and settings schemas in sync with code changes: steps 1 and 4
- `modules/camera/` is the camera contract: its backends lose `replay` (project shape, on the user's word)
