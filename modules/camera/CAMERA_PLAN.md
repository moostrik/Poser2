# The camera's plan

The camera package held up to the rules (2026-11-14, the apps' plan F; again 2026-11-21 and 2026-11-22). Who uses
what, from the code: the contract (`CameraSource`, `CameraFrame`, `CameraTracklet(s)`, the settings, `CameraHealth`,
the capture match) is read by every app and by inference, recording, tracker and render. The backends are each one
kind of camera, used now by the apps that have it: `oak` by white_space and hd_trio, `oak4` by deep_flow, low_flow
and smile, `vimba` and `webcam` by smile. That is general code with one user, which stays.

F is what is left of the scans, G the drops the camera does not show, I the scan of 2026-11-22 evening, J recordings
through the camera's own models. **(seen)** is read in the code, **(deduction)** inferred.

## F. With an OAK-D at the PC

- F1. The live check of the OAK-D's capture stamp, which is built (`oak_device/capture_clock.py`, shared with the OAK4): an
  OAK-D, white_space live, its frames' and tracklets' latency readings against oak4's. Waits for an OAK-D at the PC: none
  was attached on 2026-11-22 **(seen: depthai 2 found no device)**. RECORDING_PLAN.md B's OAK-D capture time is this
  step, and so is the live check of each OAK-D's own pipeline (built 2026-11-22): every camera, and the OAK-D camera
  with clips in -sim, opens on its own pipeline, its geometry logged. J1's clip path through the device is checked in
  the same run: a clip tracklet's `timestamp` equals the recorded capture of the frame it was found in, which J1's
  stamp rests on **(deduction)**.

## G. The drops on the panel

- G1. The OAK4's two drops into its readings. Why: "A part that drops an item shows the count to the user"
  (CLAUDE.md, Concurrency). Readings show a count today as a `Field.READ` on the camera's readings, which the panel
  shows under the camera (`tracklets`, Vimba's `dropped_frames`) **(seen)**. The two the OAK4 does not show:
  - `KeypointJoin` drops a frame whose tracklets or keypoints never came within `keep` frames; neither half goes out.
    It counts them in `dropped`, which only `scripts/oak4_probe.py` logs **(seen)**.
  - `_depth_callback` returns without the depth frame until a video frame has given the alignment, uncounted
    **(seen)**: the first frames after every open. The lens it waits for is read only off an NV12 frame, so a grey
    camera with depth never hands a depth frame over and the count rises with every one **(seen)**; deep_flow's and
    low_flow's settings files run color **(seen)**.

  The join also counts some frames twice: a half that comes after its frame was pruned makes a new entry, which is
  pruned and counted again **(seen: `_frame`)**.

  How: `RigCameraReadings` gets `dropped_tracklets` (beside `tracklets`) and `dropped_depth` (beside
  `depth_latency`), each "since the start", as Vimba's `dropped_frames`; `_close` leaves them, as they count across
  reopens. The join remembers the newest number it pruned and turns away a half at or below it, so each frame is
  counted once. It hands its count out under its lock: `put_tracklets` and `put_keypoints` return it beside the joined
  tracklets, and the camera writes it into the readings when it has changed, so the two queue threads cannot write a
  stale count over a newer one **(deduction)**. The camera counts a depth frame without its alignment: a `Field` write
  on the callback path only on a drop. The OAK-D leaves both at 0, as a backend leaves what its device does not
  measure. Files: `modules/camera/settings.py`, `modules/camera/oak4/keypoint_join.py`,
  `modules/camera/oak4/camera.py`. Test: a new `oak4/tests/test_drop_readings.py`, run without a device as
  `test_frame_flags.py` is: a frame the join drops and a depth frame before the first video frame each show in the
  readings; in `oak4/tests/test_keypoint_join.py`, a half after its frame was pruned is counted once. A live look at
  the panel, in the next OAK4 run on Matthias's go: with nobody in view `dropped_tracklets` stays at 0, as it rests on
  `GatherData` sending a frame without players **(deduction)**. Docs: OAK4.md, Reading the panel (the two counts, that
  a few depth frames at each open are expected, and grey with depth) and The backend (`KeypointJoin`'s drop).
- G2. The join starts empty at each open. Why: "Fix a problem where it is made" (CLAUDE.md, Composition and
  wiring), and no state outlives a reopen. The camera makes its `KeypointJoin` once, in `__init__`, and never clears
  it **(seen)**. A new pipeline numbers its frames from 0 again **(deduction)**, so the half-frames left at a close
  are never pruned: `_frame` prunes only numbers below the newest less `keep` **(seen)**. A frame after the reopen
  with the same number joins with a half from before it **(deduction)**. Whether or not the numbers restart, no
  half-frame outlives its pipeline. How: `KeypointJoin.clear()` empties `frames`, forgets the newest pruned number
  (G1) and counts each half-frame it empties in `dropped`, as G1 shows it. `_open` calls it before the pipeline
  starts. Files: `modules/camera/oak4/keypoint_join.py`, `modules/camera/oak4/camera.py`. Test: in
  `oak4/tests/test_keypoint_join.py`, a half-frame put before `clear()` never joins with the frame of the same
  number after it, and it is counted as dropped. Docs: OAK4.md, The backend. Built after G1, as its own
  commit.

## I. The scan of 2026-11-22 evening

- I1. The OAK-D's warp meshes in a file of their own, the keystone written once. Why: "Its files each hold one class
  or one subject" and "Inside a package, code that does the same is written once" (CLAUDE.md, Packages).
  `oak/pipeline.py` holds the pipeline and, under its own banners, the meshes: `find_warp`, `clip_warp_mesh`,
  `find_keystone_warp`, `find_keystone_warp_square`, `_report_lens_reach`, `build_warp_mesh` **(seen)**. The two
  keystone functions are one computation, a corner homography sampled on a grid, for a source rectangle of the whole
  frame or of the centered square; they differ only in that rectangle **(seen)**. Their parameters have no types, and
  they map the grid point by point in a Python loop **(seen)**. The keystone and the lens reach are geometry without
  depthai, as `geometry/warp.py`'s `warp_mesh_points` is, whose points `oak4/pipeline.py` only converts **(seen)**.
  How: one `keystone_mesh_points(source_rect, size, offset, flip_h, flip_v, mesh_w, mesh_h)` in `geometry/warp.py`
  replaces the two, typed, the grid mapped in one numpy product, its points a numpy array; `_report_lens_reach` moves
  to `geometry/mount.py` as `report_lens_reach`, beside `lens_report`, as a report on the lens. `oak/warp_mesh.py` takes `find_warp`, `clip_warp_mesh` and `build_warp_mesh`, the
  meshes as depthai takes them: they only turn the geometry's points into `dai.Point2f`. `pipeline.py` imports
  `build_warp_mesh` and `clip_warp_mesh`. Files: `oak/pipeline.py`, the new `oak/warp_mesh.py`, `geometry/warp.py`,
  `geometry/mount.py`, `geometry/__init__.py`, `oak/tests/test_keystone_warp.py`, a new
  `geometry/tests/test_keystone.py` and `oak/tests/test_camera_pipeline.py` (the import and the patch target). Test:
  the keystone cases move to `geometry/tests/test_keystone.py`; their references and cases stay as they are, their
  calls go to the one function, wide and square, each flip, exact in float32. What stays in
  `oak/tests/test_keystone_warp.py` is `build_warp_mesh`'s selection, so it becomes `oak/tests/test_warp_mesh.py`, its
  docstring saying what it tests now. Docs:
  CAMERA.md, The warp (the keystone in `geometry`) and The package's parts (`oak/` lists `warp_mesh`).
- I2. The color sensor's controls, written once for both OAKs. Why: "Inside a package, code that does the same is
  written once" (CLAUDE.md, Packages). Both OAKs deliver from a color sensor with the same controls: the OAK-D's
  `ColorSensorSettings` and the OAK4's `SensorSettings` hold the same ten fields, by name: exposure, ISO, auto
  exposure, white balance, auto white balance, brightness, contrast, saturation, luma denoise and sharpness, and both `SensorControl`s send them with the same
  `dai.CameraControl` calls and read the same values back off the frames, on depthai 2 and 3 alike **(seen)**. The
  mono sensors are another kind with neither white balance nor image controls: the OAK-D's has exposure, ISO and the
  IR lights (`MonoSensorSettings`), the OAK4's stereo pair runs its own auto exposure, and its IR lights sit in
  `SensorSettings` **(seen)**. Today the OAK-D's one `SensorControl` branches on `color` for its bindings, its
  first send and its readback **(seen)**. How: `oak_device/color_sensor_control.py` holds `ColorSensorControl`: the
  color sensor's fields bound (over a `ColorSensorFields` Protocol of the ten), every value sent once on
  construction, `readback`, `close()` unbinding; the bindings typed. `BaseSettings.bind` takes a `Field` and finds
  it by name **(seen)**, so the Protocol's members are `Field`s (`exposure: Field[int]`): read on the sensor they give
  the value, read on its class (`type(sensor).exposure`) the `Field` to bind; pyright passes both reads and the two
  settings classes against such a Protocol, and fails a Protocol of plain values or of `ClassVar`s **(seen: a
  pyright check of 2026-11-24 in `.venv`)**. The OAK4 keeps its order at open: its `_apply` sends anti-banding,
  compensation and the ISO limit before the shared values, and the IR lights after them, as now **(seen)**. Since the
  shared class sends every value in `__init__`, the OAK4's `SensorControl` sets what its `_apply` uses (the device
  for the IR lights, the exposure region's transformation) before it calls `super().__init__` **(deduction)**. The OAK-D's color camera uses it as it is; its
  mono camera has a `MonoSensorControl` of its own (`oak/control.py`: exposure, ISO, auto exposure and the IR
  lights). The OAK4's `SensorControl` extends `ColorSensorControl` with what only it has: the exposure and balance
  locks, compensation, ISO limit, exposure region, anti-banding, chroma denoise and the IR lights. A subclass and not a
  part, since the OAK4's lock goes out with each auto exposure and auto balance switch the shared class sends
  **(seen: `oak4/control.py`, `_on_auto_exposure`, `_on_auto_balance`)**; `ColorSensorControl` is concrete, as the
  OAK-D's color camera uses it as it is. Files: `oak_device/color_sensor_control.py`, `oak_device/__init__.py`,
  `oak/control.py`, `oak/camera.py` (the control by the camera's sensor), `oak4/control.py`. Test:
  `oak4/tests/test_control.py` passes as it is; a test in `oak_device/tests` that `ColorSensorControl` sends each
  value and reads back exposure, ISO and color temperature, with a fake queue and frame. Docs: CAMERA.md, The
  package's parts (`oak_device/` lists `color_sensor_control`).
- I3. Form only. Each is checked by the tests and pyright:
  - `SYSTEM_RATE_HZ` moves from `oak4/device.py` to `oak4/pipeline.py`, its one user. Why: "Its files each hold one class
    or one subject, named after it"; `device.py` is finding a device **(seen)**.
  - `geometry/__init__.py` drops `MONO_RESOLUTIONS` and `COLOR_RESOLUTIONS`, which only `sensor_modes.py` uses
    **(seen)**. Why: "Its `__init__.py` holds what other packages use".
  - `geometry/tests/test_orientation.py`'s `TestMountCheck`, `TestCameraFpsCheck`, `TestPinnedLatency` and
    `_TiltedCamera` move to `tests/test_camera_health.py`: they test `camera_health.py` **(seen)**. Its
    `TestRollOffsetIsABox` moves to a new `tests/test_settings.py`: it tests `RigCameraReadings` **(seen)**. Why: "Its
    tests sit in its `tests/` folder".
  - `oak4/tests/test_keypoint_join.py`'s docstring says "the player", not "the person" **(seen)**. Why: the
    glossary, a person in docs is "player".
  - CAMERA.md, The lens: `log_connected_sensors` logs the sensor behind each socket at open, not
    `get_device_list(verbose=True)`, which lists the devices **(seen)**.
  - `DepthControl._fields` is typed with its element type, not bare `list` **(seen)**. Why: "Use modern typing".
  - The IR lights have one name in both OAKs: the OAK-D's `ir_grid_light` and `ir_flood_light` (in
    `MonoSensorSettings`, and the relayed `OakCameraSettings.ir_flood_light`) become the OAK4's `dot_projector` and
    `flood_light` **(seen)**. Why: "Use no synonyms for one concept across modules; when a term changes, rename it
    everywhere in the same change". Settings files: `apps/hd_trio/data/settings/setup/studio.json` and `umu.json`,
    `apps/white_space/data/settings/setup/public.json` and `studio.json` **(seen)**. App files:
    `apps/white_space/settings.py` (its `ir_flood_light` and `_cam_share`) and `apps/white_space/docs/CALIBRATION.md`
    **(seen)**.

## J. Recordings through the camera's own models

Feeding a recording's frames into an OAK tests that camera's own detector, tracker and pose chain on a recorded
scene. Our recording format stays, depth is not fed back, and the replay (`modules/recording/replay`) stays as it is
(Matthias's decision of 2026-11-23). Only what an app uses is built and tested: the OAK-D's mono or color frame. No
OAK4 app is fed a recording; the OAK4's pipeline is made ready for it, without a clip input (Matthias, 2026-11-24).
The design leaves room for the rest (the OAK4's clip input, its grey warp, depth) without building it. The OAK-D does
this today: `oak.ClipCamera` sends the player's video frames into the
pipeline where the sensor's would enter, and the device's warp (with `clip_warp`), YOLO and tracker run on them as on
the rig **(seen)**; white_space's `studio.json` runs it (`replay` false) **(seen)**. The OAK4 has no such path
**(seen)**. depthai 3 sends a host frame into a device node with `Node.Input.createInputQueue()`; its `XLinkIn` is
internal and made by depthai **(seen in `.venv-oak4`, depthai 3.11; the Luxonis porting guide)**. `add_detector`,
`add_tracker` and `add_pose_chain` take their frames as one `dai.Node.Output` **(seen)**.

- J1. The OAK-D's camera takes its clips as a part, chosen at build, instead of the `ClipCamera` subclass. Why:
  "Prefer composition over inheritance for runtime assembly" (CLAUDE.md, Composition and wiring). `ClipCamera`
  overrides `start`, `run`, `connected`, `_setup_pipeline`, `_setup_queues` and `_close` **(seen)**. How:
  `oak.Camera(settings, cam_id, clips: ClipSource | None = None)`; with clips the pipeline is built with
  `clips=True` and the camera makes a `ClipInput` (`oak/clip_input.py`: the clip `XLinkIn` queue, a clip frame sent
  as the sensor would send it, the size warning, `to_planar`) in place of the IMU queue and the sensor controls. A fed
  frame is stamped with its recorded capture (`ClipFrame.timestamp`, on the play's clock), not with its send as
  `ClipCamera` does now (`perf_counter()` at send) **(seen)**; a clip frame without one is stamped at its send. The
  device's tracklets then come back through `capture_time` as the recorded capture **(deduction: it rests on the
  tracker keeping its input frame's stamp, and on depthai 2's `dai.Clock` reading `perf_counter`'s counter, measured
  only on depthai 3, OAK4.md)**. A recorded clip is already warped and goes past the device's warp; only `clip_warp`
  sends a raw one through it **(seen: `build_pipeline`, CAMERA.md, Raw clips)**. The clip input takes the clip frames
  of the kind it feeds (`FrameType.VIDEO`) and leaves the others, so another kind (depth) is another input beside it,
  not a change to `ClipSource` **(seen: `ClipFrame.kind`)**. `ClipSource` has only `add_frame_callback` **(seen)**,
  so the camera registers one clip callback with the player at `start`; it hands each clip frame to the open
  `ClipInput`, and to nothing while the device is closed. That callback turns the clip frame into the device's message
  (one copy: planar BGR, or grey) and sends it into the non-blocking queue: the send is the hand-over to the device, and
  a host thread of its own would add a hop and a copy to every frame **(deduction)**. Its queue stays `maxSize=1,
  blocking=False`, a drop no count shows yet (Open, the drops inside depthai).
  `clip_passthrough` goes, with its code path (`ClipCamera._passthrough_callback` and the branches in `start`, `run`
  and `connected`): it hands clip frames on without a device and without tracklets, which `replay.Camera` does, and
  no settings file sets it **(seen)**. Why: "Remove obsolete pathways instead of leaving parallel legacy ones"
  (CLAUDE.md, API evolution); Matthias's decision of 2026-11-23. `clip_warp` stays. Files: `oak/settings.py` (the field),
  `oak/camera.py`, the new `oak/clip_input.py`, `oak/clip_camera.py` goes, `oak/__init__.py` (no `ClipCamera`),
  `apps/white_space/main.py` and `apps/hd_trio/main.py` (`Camera(..., clips=self.player)`, app files). Tests:
  `oak/tests/test_camera_pipeline.py` and `oak/tests/test_frame_stop.py` take the camera with clips, and
  `test_frame_stop.py`'s `test_a_passthrough_clip_camera_runs_from_its_start_until_stop` goes with the mode; a new test in
  `oak/tests` that a clip frame goes into the clip queue planar BGR, gray for a mono camera, stamped with its recorded capture, without a device (a fake
  queue); the clip path through the device is checked live with F1. Docs: CAMERA.md, The clip camera, The package's
  parts and What no settings file runs; RECORDING.md and `modules/recording/player/player.py`'s docstring, which name
  `oak.ClipCamera`.
- J2. The OAK4's pipeline takes its frames from one source, so a clip input can stand in for the sensor later. Why:
  "Prefer composition over inheritance for runtime assembly" (CLAUDE.md, Composition and wiring), and Matthias's
  decision of 2026-11-24: ready, not wired. Today `build_pipeline` makes the `Camera` node inline, and its `frames`
  feed three consumers: the detector's `ImageManip`, `FrameCropper` in `add_pose_chain`, and `PipelineOutputs.video`
  **(seen)**. A host input queue is a node's `Input`, which cannot feed three **(deduction)**. How: `add_camera` makes
  the `Camera` node (color off its NV12 output, grey through the warp) and returns its frames and its control input;
  `build_pipeline` takes the frames source as a parameter and builds the model chains and the video output on that
  one `dai.Node.Output`. With the `Camera` node the graph is the one built now. No clip input, no app wiring, no
  -sim run. Files: `oak4/pipeline.py`, `oak4/camera.py` (the call), `oak4/tests/test_retry.py` (it patches
  `build_pipeline` in `camera.py`) **(seen)**. Test: no OAK4 test builds the pipeline today **(seen)**. A new
  `oak4/tests/test_pipeline.py` builds the graph on a pipeline without a device, before and after the split, for each
  model chain, color and grey, with and without depth, and compares the nodes and their links. That depthai 3 builds a
  pipeline without a device is not tried **(deduction)**; if it cannot, the check is the next OAK4 open on Matthias's
  go, each chain's frames and tracklets coming as before. Docs: OAK4.md, The backend (the frames source).

## Open

- The OAK4 is to run standalone, as an OAK App **(site fact)**. Depth alignment then needs a home: `ImageAlign` is
  the only alignment on the camera and fattens near objects ([Depth](oak4/OAK4.md#depth)), and its firmware cannot be
  changed. The options are to keep sending raw depth with its `DepthAlignment` to a PC that runs `DepthAlign`, or to
  run the same reprojection as a host node on the camera's CPU, whose speed for 144,000 points a frame has not been
  measured. A Script node is too slow for it **(deduction)**.
- Flow Motion's A/B of the depth's confidence needs a recording with it and a player moving. No recording with a
  confidence is kept: the two probes of 2026-11-21 were made to measure the encoder and are deleted.
- How to count the drops inside depthai, of both OAKs. The detector's input is non-blocking
  (`setBlocking(False)` in both `pipeline.py`) and skips the frames it has no time for. The OAK4's depth `Sync`
  (`add_depth`) lets go of a depth or confidence without its partner within `CAPTURE_TOLERANCE` **(deduction)**. The OAK-D's clip input is
  `maxSize=1, blocking=False` **(seen)** and so drops a clip frame the device has not taken yet **(deduction)**.
  depthai reports no count of any of the three **(deduction)**. The panel shows them only as a rate: `tracker_fps`, and
  `video_fps` against `fps` **(seen)**. A gap in a stream's sequence numbers at the hand-over would count them, if
  ImageManip and Warp carry the numbers through, which is not checked. The host's output queues (`maxSize=1`,
  non-blocking) lose nothing: their callback takes every message, and what the queue overflows has already gone to
  the callback **(deduction)**.
- Feeding the OAK4 a recording, when an OAK4 app needs it. One input node, its queue from `createInputQueue()`, as
  J2's frames source in place of the `Camera` node; `oak4.Camera(settings, cam_id, clips: ClipSource | None = None)`
  as J1's camera, with no Neural Depth, no sensor controls and no IMU; the fed frame stamped as in J1. The Player is
  built with `nv12=True` in the OAK4 apps **(seen)**, so a clip frame needs no conversion, only its strides and planes.
  First a probe with the OAK4 attached, on Matthias's go: the crops and their keypoints land in the frame (an
  `ImgTransformation` made from the recorded lens and size, through `FrameCropper` and `remapPointTo`), a stamp set on
  the host frame comes back through `capture_time`, and 720p NV12 sent at 60 fps holds the rate **(deduction, not
  run)**. deep_flow and low_flow have no `replay` switch; their -sim always runs `replay.Camera` **(seen)**.

## Rules it touches

- Stamp a frame or tracklet message once, where the backend hands it over, with its moment of capture where the device reports one, else its arrival: F1, J1
- A part that drops an item shows the count to the user: G1, G2
- Fix a problem where it is made, not where it shows: G1, G2
- A callback only hands the item over and returns: J1
- Its files each hold one class or one subject, named after it: I1, I3
- Inside a package, code that does the same is written once: I1, I2
- Use a `Protocol` for what a consumer needs from a collaborator, an `ABC` for a family of classes that share code: I2
- Its `__init__.py` holds what other packages use: I3
- Its tests sit in its `tests/` folder: I3
- Use the terms in `.claude/rules/glossary.md`: I3
- Use modern typing (`list[T]`, `dict[K, V]`, `X | Y`): I2, I3
- Use no synonyms for one concept across modules; when a term changes, rename it everywhere in the same change: I3
- Keep settings files in sync with code changes: I3
- Prefer composition over inheritance for runtime assembly: J1, J2
- Remove obsolete pathways instead of leaving parallel legacy ones: J1
- Run a test of the app or a GPU harness only on the user's go, and only what the plan names: F1, G1, J2
