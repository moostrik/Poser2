# The camera's plan

The camera package held up to the rules (2026-11-14, the apps' plan F; again 2026-11-21). Who uses what, from the
code: the contract (`CameraSource`, `CameraFrame`, `CameraTracklet(s)`, the settings, `CameraHealth`, the capture
match) is read by every app and by inference, recording, tracker and render. The backends are each one kind of
camera, used now by the apps that have it: `oak` by white_space and hd_trio, `oak4` by deep_flow, low_flow and
smile, `vimba` and `webcam` by smile. That is general code with one user, which stays.

F is what is left of the scan of 2026-11-21, H the proposals of the scan of 2026-11-22 (after 5928db7d..69ecc10d;
its tests pass in both venvs, pyright clean in both). **(seen)** is read in the code, **(deduction)** inferred.

## F. With an OAK-D at the PC

- F1. The live check of the OAK-D's capture stamp, which is built (`oak_device/capture_clock.py`, shared with the OAK4): an
  OAK-D, white_space live, its frames' and tracklets' latency readings against oak4's. Waits for an OAK-D at the PC: none
  was attached on 2026-11-22 **(seen: depthai 2 found no device)**. RECORDING_PLAN.md B's OAK-D capture time is this
  step, and so is the live check of each OAK-D's own pipeline (built 2026-11-22): every camera, and every
  `ClipCamera` in -sim, opens on its own pipeline, its geometry logged.

## H. The scan of 2026-11-22

- H3. Stale text **(seen)**. Why: "A docstring says what the thing is"; a document matches the code.
  - `CameraHealth.update`: "Recompute both verdicts"; it sets three (`mount`, `camera_fps`, `video_rate`).
  - `OakCameraSettings.resolution`: "P1080 is color only"; `P2160` is too (`COLOR_RESOLUTIONS`).
  - OAK4.md, Recording: "Frames are stored mirrored and not straightened"; a recording holds them as delivered with
    their flags (RECORDING.md). The section restates RECORDING.md: it becomes a link to it ("A component is
    described in its own document").
- H4. One word per flag in prose **(seen)**: "mirror"/"mirrored" for `flip_h`, "the turn"/"turned" for `rotate90`,
  "upright" for the oriented image. In `frame.py` (`CameraFrame`'s docstring and field comments, `oriented_corners`),
  `flips.py` (locals `mirrored`, `turned`), `tracklet.py` (`straightened`'s comment), `oak4/camera.py` (`_tracklet`,
  `_homography` and its warning), `oak4/control.py`, `oak4/sensor.py` (`sensor_region`), `vimba/settings.py`
  (`flip_h`: "Mirror the camera image horizontally", against `CameraSettings`' "Flip horizontal"), CAMERA.md (Raw
  clips), OAK4.md (the frame's flags, depth) and the tests' names. The IMU's "turn" on the board is its roll and
  stays. Also "person": `OakCameraSettings.yolo`'s panel text describes the detector, so "Run the YOLO detector and
  tracker on the camera"; a docstring that means a player says "player" (`keypoint_join.py`, `oak4/models.py`,
  `oak4/pipeline.py`, `oak4/camera.py`, `tracklet.py`) **(seen)**. Why: "use that one term everywhere it appears";
  the glossary's "player"; E1 named the helpers after the flags, the prose did not follow.
- H5. Leftovers and form **(seen)**:
  - `num_tracklets` on `oak.Camera` and `oak4.Camera` is read by nobody; `readings.tracklets` holds it.
  - `SensorControl` is exported by the `oak`, `oak4` and `vimba` `__init__.py` and used by no other package
    (vimba's test can import `vimba.control`). Why: "Its `__init__.py` holds what other packages use".
  - `ClipCamera.to_planar` resizes a frame to its own size, a copy for nothing; transpose and flatten stay. Why:
    "Avoid unnecessary copies".
  - `POSE_CONFIDENCE_THRESHOLD`, `POSE_OVERLAP_THRESHOLD` (`oak4/models.py`) set every detector, the plain one
    included (`add_detector`); the OAK-D's are `YOLO_*`. One name in both, after what they are:
    `DETECTOR_CONFIDENCE_THRESHOLD`, `DETECTOR_OVERLAP_THRESHOLD`. Why: "Name a concept after what it is".
  - `CameraFrame`'s flags in the order they are applied (`homography`, `rotate90`, `flip_v`, `flip_h`; `flip_h` sits
    first now); every call passes them by keyword **(seen: at most four positional arguments)**, confirmed by the
    full test run.
  - `import logging` after the other imports in `sensor_modes.py`: PEP 8.
  - `imu_to_camera` passes the vector on unchanged when the matrix is malformed, silently: logged once. Why: "never
    fail silently".
  - A docstring for `oak.Camera`, `oak4.Camera`, `oak.ClipCamera`, `vimba.SensorControl`.
- H7. The OAK4's exposure region is in the frame as the camera delivers it, before the homography and every flag.
  Now `sensor_region` takes it on the picture as shown and undoes the flips and `rotate90`, not the homography: the
  flags undone, the tilt correction not **(seen: `oak4/sensor.py` `sensor_region`, `oak4/control.py`
  `_on_exposure_region`)**. Only the panel sets it, and every settings file that has it holds the full frame
  **(seen: deep_flow's and low_flow's `studio.json`)**, so nothing changes in use. How: `sensor_region` keeps only
  the mapping to sensor pixels (`to_source`); `SensorControl` loses its `rotate90`, `flip_v` and `flip_h`;
  `SensorSettings.exposure_region`'s description says "Part of the delivered frame auto exposure measures, in
  fractions of it"; OAK4.md's `exposure_region` line follows; `test_control.py`'s flipped and turned region tests
  go, and one test holds a region mapped straight through `to_source`. Why: a sensor control lives in the camera's
  own picture; the code is shorter and the region one thing.

H7 is a step of its own, checked by the tests in both venvs, pyright and the guard tests;
H3 to H5 are one commit.

## Open

- The OAK4 is to run standalone, as an OAK App **(site fact)**. Depth alignment then needs a home: `ImageAlign` is
  the only alignment on the camera and fattens near objects ([Depth](oak4/OAK4.md#depth)), and its firmware cannot be
  changed. The options are to keep sending raw depth with its `DepthAlignment` to a PC that runs `DepthAlign`, or to
  run the same reprojection as a host node on the camera's CPU, whose speed for 144,000 points a frame has not been
  measured. A Script node is too slow for it **(deduction)**.
- Flow Motion's A/B of the depth's confidence needs a recording with it and a player moving. No recording with a
  confidence is kept: the two probes of 2026-11-21 were made to measure the encoder and are deleted.

## Rules it touches

- Stamp a frame or tracklet message once, where the backend hands it over, with its moment of capture where the device reports one, else its arrival: F1
- Fix a problem where it is made: H7
- Keep `Field` descriptions to one short line of panel text: what the value is and its unit: H7
- A docstring says what the thing is: H3, H5
- A component is described in its own document; other documents link to it: H3
- Use that one term everywhere it appears: H4, H5
- Use the terms in `.claude/rules/glossary.md`: H4
- Its `__init__.py` holds what other packages use: H5
- Avoid unnecessary copies: H5
- Never fail silently: H5
