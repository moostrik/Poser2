# The camera's plan

The camera package held up to the rules (2026-11-14, the apps' plan F; again 2026-11-21). Who uses what, from the
code: the contract (`CameraSource`, `CameraFrame`, `CameraTracklet(s)`, the settings, `CameraHealth`, the capture
match) is read by every app and by inference, recording, tracker and render. The backends are each one kind of
camera, used now by the apps that have it: `oak` by white_space and hd_trio, `oak4` by deep_flow, low_flow and
smile, `vimba` and `webcam` by smile. That is general code with one user, which stays.

The steps below are proposals from the scan of 2026-11-21; nothing is built. **(seen)** is read in the code,
**(deduction)** inferred.

## A. One callback list

- A1. Each backend's frame and tracklet callbacks become a `utils.Broadcast` (`oak.Camera`, and `ClipCamera` through
  it, `oak4.Camera`, `vimba.Camera`, `webcam.Camera`), with `add_frame_callback` and `add_tracklet_callback` handing
  on to it. Why: "A callback list is a `utils.Broadcast`" and "A failing callback does not stop the others". Today
  four backends each keep a list, a lock and the same add and emit code **(seen)**; in `oak`, `oak4` and `webcam` a
  callback that raises skips the rest and lands in depthai's or the capture thread **(seen: `_emit_frame`,
  `_emit_tracklets`, `_emit`)**; only Vimba catches it **(seen)**. CAMERA.md's contract line on callbacks follows.
  Tests: the camera, vimba and webcam tests; oak4's in `.venv-oak4`.

## B. Written once

- B1. The mount readout, one class in `geometry/orientation.py`: the gravity vector smoothed, its tilt and roll, the
  roll less `roll_offset`, into the readings. Each backend passes its `IMU_BOARD_ROLL`. Why: "Inside a package, code
  that does the same is written once"; `oak.Camera._imu_callback` and `oak4.Camera._imu_callback` are the same code,
  and `IMU_SMOOTHING` and `IMU_RATE_HZ` are set twice **(seen)**. The part that calls depthai
  (`imu_rotation_to_camera`) stays in each backend. Tests: `geometry/tests/test_orientation.py` gains the readout.
- B2. `cam_id` comes from the constructor in `oak.Camera`, `ClipCamera` and `oak4.Camera`, as in `vimba`, `webcam` and
  `replay.Camera`. Why: "Pass collaborators and fixed structure (counts known at startup) through constructors";
  the OAK backends count `Camera._next_cam_id` in creation order, a class-wide count that only holds while one app
  makes its cameras in index order **(seen)**. Call sites: the mains of white_space, hd_trio, deep_flow, low_flow
  and smile, `scripts/oak4_probe.py`.
- B3. `oak.pipeline.WarpConfig` is named `MountConfig`, as `oak4.pipeline.MountConfig`. Why: "Similar things share
  the noun": both are "how a camera is mounted and what its lens is" **(seen)**; the OAK-D's adds tilt and keystone.

## C. Small fixes to the rules

- C1. `webcam.Camera.run` waits after a failed read on a `utils.Waker`, not `time.sleep(0.01)` **(seen)**. Why: "A
  thread that waits for a moment waits on a `utils.Waker`".
- C2. `ClipCamera._clip_callback` stamps a clip frame for the device with `perf_counter`, not `process_time()`, the
  process's CPU time **(seen)**. Why: "Read `time.perf_counter()` for every stamp"; the device's tracker reads this
  stamp **(deduction)**.
- C3. `vimba.SensorControl._read_actual` reports a failed exposure or gain read through `_problem`, as its other
  reads do, not `except Exception: pass` **(seen)**. Why: "Log exceptions with context; never fail silently".
- C4. The Vimba white balance fields take the backends' word, `balance`, in snake case: `WB_red` / `WB_blue` →
  `balance_red` / `balance_blue`, `manual_WB_red` / `manual_WB_blue` → `manual_balance_red` / `manual_balance_blue`,
  `auto_WB` → `auto_balance`, `set_WB` → `measure_balance` **(seen: `vimba/settings.py`)**. Why: PEP 8, and "use
  that one term everywhere": `oak` and `oak4` say `balance`, `auto_balance`. smile's settings files are migrated.
- C5. The fields without a description get one: `OakCameraSettings.square`, `yolo`, `model_path`,
  `Oak4CameraSettings.model_path` **(seen)**. Why: "Keep `Field` descriptions to one short line of panel text".
- C6. `CameraHealth` keeps its settings as `self.settings`, not `self._settings` **(seen)**. Why: the glossary, "A
  component's settings object: `settings` / `self.settings`".
- C7. `KeypointJoin` takes a lock round its frames, and holds a frame's two halves in a small dataclass instead of a
  dict with string keys **(seen)**. Why: its two inputs come from two depthai queues, whose callbacks can run on
  two threads **(deduction)**; "Use `Lock` and `Event` for shared mutable state", "Prefer `IntEnum` over string
  keys". No settings file runs the two-model chain today (CAMERA.md, what no settings file runs).
- C8. "normalised" is spelled "normalized" in `tracklet.py`, `geometry/mirror.py`, `geometry/tilt_correction.py`,
  `oak4/models.py`, `oak4/pipeline.py`, `oak4/keypoint_join.py` **(seen)**. Why: the glossary's American spelling.

## D. The documents

- D1. CAMERA.md names `CameraReadingsDraw`, not `CameraReadingsRenderer` **(seen: `modules/render/__init__.py`)**;
  OAK4.md names `ViewDepth`, not `ViewDepthLayer` **(seen: deep_flow `view_depth.py`)**. Why: the document matches
  the code.

## F. One host-side flip

- F1. `oriented_image` applies `rotate90` and both flips in the frame's own layout, NV12 included (Y as a plain array, UV as (H/2, W/2, 2)); `mirrored_image` goes; `bgr_frame` stays. Why: "Inside a package, code that does the same is written once": `mirrored_image` and `oriented_image` both flip on the host, and only `mirrored_image` takes NV12 **(seen: `frame.py`)**. Tests: `oriented_image` on NV12 against `bgr_frame` then `oriented_image`, for all 8 flag sets.
- F2. The recorder's `mirrored` option, which bakes in `flip_h` alone and skips a frame with `rotate90` **(seen: `recorder.py` `mirrored_capture`)**: becomes `oriented` and applies every flag to the image, the lens (`rotate90` swaps fx/fy and cx/cy), the depth alignment and the tracklets, the info line reading all flags off; tested by a recorder round trip with `oriented` on. A recording by default keeps the frame as delivered with its flags, and replay applies them (smile's -sim of 20261121-110540 came out upright). The recorder is recording's; Smile and Module Refactor agreed this step (2026-11-21). hd_trio's `studio.json` and `umu.json` (`mirrored: true`) get the key renamed.
- F3. The host-side helpers are named after the flag they apply: `upright_image` → `oriented_image`, `upright_corners` → `oriented_corners`; `mirrored_roi` / `mirrored_keypoints` → `flip_h_roi` / `flip_h_keypoints`; `flipped_roi` / `flipped_keypoints` → `flip_v_roi` / `flip_v_keypoints`; `turned_roi` / `turned_keypoints` → `rotate90_roi` / `rotate90_keypoints`; `mirrored_intrinsics`, `mirrored_alignment`, `mirrored_tracklets` → `flip_h_intrinsics`, `flip_h_alignment`, `flip_h_tracklets`; `geometry/mirror.py` → `geometry/flips.py`; the documents say `rotate90`, not "the turn". Why: "use that one term everywhere": the frame's flags are `flip_h`, `flip_v`, `rotate90` **(seen: `frame.py`)**, and the three together are its orientation. Call sites: the recorder, inference's uploader, oak4's `sensor_region`, smile's face finder, render and main. Smile owns `rotate90` and the flips and checks it once built.

## E. With an OAK-D at the PC

- E1. The `oak` backend stamps a frame and its tracklets with their capture, off the message's device timestamp as
  `oak4.messages.capture_time` does, instead of their arrival **(seen: `oak.Camera._emit_frame`,
  `_tracker_callback`)**; depthai 2 messages carry one **(deduction)**. Why: camera.md, "Stamp a frame once ... with
  its moment of capture where the device reports one". RECORDING_PLAN.md 2's OAK-D capture time is this step; it comes after the camera plan, in his order video, camera, recording. Test: an OAK-D, white_space
  live, its latency reading against oak4's.

## Open

- The OAK4 at P2160 (2704 x 2160) in H.264 has not been replayed on the iGPU. Its d3d12va H.264 decode hangs
  silently at 3088 x 2316, while HEVC at that size plays (the video scan, 2026-11-21). `FRAME_WIDTH_ALIGNMENT` in
  `oak4/sensor.py` covers the width only. Test a P2160 H.264 clip on the iGPU, or record P2160 as HEVC? The codec is
  the recording settings'.
- The OAK4 is to run standalone, as an OAK App **(site fact)**. Depth alignment then needs a home: `ImageAlign` is
  the only alignment on the camera and fattens near objects ([Depth](oak4/OAK4.md#depth)), and its firmware cannot be
  changed. The options are to keep sending raw depth with its `DepthAlignment` to a PC that runs `DepthAlign`, or to
  run the same reprojection as a host node on the camera's CPU, whose speed for 144,000 points a frame has not been
  measured. A Script node is too slow for it **(deduction)**.
- Flow Motion's A/B of the depth's confidence needs a recording with it and a player moving. No recording with a
  confidence is kept: the two probes of 2026-11-21 were made to measure the encoder and are deleted.

## Rules it touches

- A callback list is a `utils.Broadcast`; a failing callback does not stop the others: A1
- Inside a package, code that does the same is written once: B1, F1
- Pass collaborators and fixed structure through constructors: B2
- Similar things share the noun: B3
- A thread that waits for a moment waits on a `utils.Waker`: C1
- Read `time.perf_counter()` for every stamp and duration: C2
- Log exceptions with context; never fail silently: C3
- Follow PEP 8; one term everywhere; keep settings files in sync: C4
- Keep `Field` descriptions to one short line of panel text: C5
- The glossary: C6, C8
- Use `Lock` and `Event` for shared mutable state; prefer `IntEnum` over string keys: C7
- A document describes what is built: D1
- Stamp a frame once, with its moment of capture where the device reports one: E1
- Remove obsolete pathways instead of leaving parallel legacy ones: F2
- Use that one term everywhere it appears: F3
