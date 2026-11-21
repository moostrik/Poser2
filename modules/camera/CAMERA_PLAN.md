# The camera's plan

The camera package held up to the rules (2026-11-14, the apps' plan F; again 2026-11-21). Who uses what, from the
code: the contract (`CameraSource`, `CameraFrame`, `CameraTracklet(s)`, the settings, `CameraHealth`, the capture
match) is read by every app and by inference, recording, tracker and render. The backends are each one kind of
camera, used now by the apps that have it: `oak` by white_space and hd_trio, `oak4` by deep_flow, low_flow and
smile, `vimba` and `webcam` by smile. That is general code with one user, which stays.

The steps below are proposals from the scan of 2026-11-21; nothing is built. **(seen)** is read in the code,
**(deduction)** inferred.

## B. Written once

- B4. `rotate90` is a `Field.INIT` field of `CameraSettings` beside the flips, off by default; `oak4`, `vimba` and
  `webcam` read it there and lose their `rotate90` parameter, smile's `_Oak4Settings` drops its copy **(seen:
  `apps/smile/cam/settings.py`)**, and smile's portrait mode decides only the screen. smile's webcam gets its own
  `CameraSettings` group. The replay camera takes the flag from the recording, so its `replay` group holds the key
  unread **(seen: `modules/recording/replay/camera.py`)**. The OAK-D applies no turn (its flips are baked into its warp,
  whose columns are the panoramic tracker's bearings): it logs once that `rotate90` is not applied (his word,
  2026-11-21). Why: "Components take their settings object ... in the constructor"; today Vimba and the webcam take
  portrait mode **(seen: `apps/smile/main.py`)**. Settings files: smile's four turn on only `camera.rotate90`; every
  other app's gain it, off. Call sites: smile's main, oak4's, vimba's and webcam's tests. Tests: those, smile's; smile
  -sim on 20261121-110540 portrait and `-lm`, on his go.

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
  that one term everywhere": `oak` and `oak4` say `balance`, `auto_balance`. smile's `Smile.json`, `Studio.json` and
  `GenerationTest.json` hold the keys and are migrated; `Grintest.json` has none **(seen)**.
- C5. The fields without a description get one: `OakCameraSettings.square`, `yolo`, `model_path`,
  `Oak4CameraSettings.model_path` **(seen)**. Why: "Keep `Field` descriptions to one short line of panel text".
- C6. `CameraHealth` keeps its settings as `self.settings`, not `self._settings` **(seen)**. Why: the glossary, "A
  component's settings object: `settings` / `self.settings`".
- C7. `KeypointJoin` takes a lock round its frames, and holds a frame's two halves in a small dataclass instead of a
  dict with string keys **(seen)**. Why: its two inputs come from the tracklets' and the poses' queues
  **(seen: `oak4/camera.py` `_tracker_callback`, `_poses_callback`)**, whose callbacks can run on two threads
  **(deduction: depthai's docs bring each device output to the host through its own `XLinkInHost`, a threaded host
  node; neither the docs nor `.venv-oak4`'s stubs say which thread runs a callback)**; "Use `Lock` and `Event` for
  shared mutable state", "Prefer `IntEnum` over string keys". No settings file runs the two-model chain today
  (CAMERA.md, what no settings file runs). Check: log `threading.get_ident()` in both callbacks during a two-model
  chain run with the OAK4, on his go.
- C8. "normalised" is spelled "normalized" in `tracklet.py`, `geometry/mirror.py`, `geometry/tilt_correction.py`,
  `oak4/models.py`, `oak4/pipeline.py`, `oak4/keypoint_join.py` **(seen)**. Why: the glossary's American spelling.

## D. The documents

- D1. OAK4.md names `ViewDepth`, not `ViewDepthLayer` **(seen: deep_flow `view_depth.py`)**. Why: the document
  matches the code.

## E. One host-side flip

- E1. The host-side helpers are named after the flag they apply: `upright_image` → `oriented_image`, `upright_corners`
  → `oriented_corners`; `mirrored_roi` / `mirrored_keypoints` → `flip_h_roi` / `flip_h_keypoints`; `flipped_roi` /
  `flipped_keypoints` → `flip_v_roi` / `flip_v_keypoints`; `turned_roi` / `turned_keypoints` → `rotate90_roi` /
  `rotate90_keypoints`; `geometry/mirror.py` → `geometry/flips.py`; CAMERA.md, OAK4.md and RECORDING.md say `rotate90`, not "the turn". Why:
  "use that one term everywhere": the frame's flags are `flip_h`, `flip_v`, `rotate90` **(seen: `frame.py`)**, and the
  three together are its orientation. `mirrored_image`, `mirrored_intrinsics`, `mirrored_alignment` and
  `mirrored_tracklets` are not renamed: the recorder is their only caller outside `modules/camera`, and E3 takes them
  away **(seen)**. Call sites: `tracklet.py`, oak4's `sensor_region`, smile's face finder and main, CAMERA.md, and the
  tests of the camera, oak4, geometry and inference's uploader **(seen)**. Smile owns `rotate90` and the flips and
  checks it once built.
- E2. `oriented_image` applies `rotate90` and both flips in the frame's own layout, NV12 included (Y as a plain array,
  UV as (H/2, W/2, 2)); `bgr_frame` stays. Why: "Inside a package, code that does the same is written once":
  `mirrored_image` and `oriented_image` both flip on the host, and only `mirrored_image` takes NV12 **(seen:
  `frame.py`)**. Every caller of `upright_image` converts with `bgr_frame` first **(seen: smile's face finder and
  main)**, and `mirrored_image` goes in E3. Tests: `oriented_image` on NV12 against `bgr_frame` then `oriented_image`,
  for all 8 flag sets.
- E3. The recorder's `mirrored` option and `mirrored_capture` go: the option acts only on a frame with `flip_h`, and
  the OAK-D, hd_trio's camera, bakes its flips into its warp and sets no flags **(seen: `oak/camera.py`
  `_emit_frame`, `oak/clip_camera.py`, `oak/pipeline.py` `build_warp_mesh`; hd_trio's recordings hold `flip_h: false`,
  so its replay sets none)**. A recording keeps the frame as delivered with its flags, and replay applies them.
  `RecorderSettings.mirrored` goes, and so does what `recorder.py`'s module docstring says of it.
  `mirrored_image`, `mirrored_intrinsics`, `mirrored_alignment` and `mirrored_tracklets` go with it, with their tests:
  `MirroredTest` in `recorder/tests/test_recorder.py`, the `mirrored_image` cases of `tests/test_frame.py` (its
  `bgr_frame`, `oriented_image` and `oriented_corners` cases stay), and `geometry/tests/test_depth_projection.py`'s
  mirrored case **(seen: no other caller)**. RECORDING.md's section on it goes; hd_trio's `studio.json` and `umu.json` and deep_flow's and low_flow's
  `studio.json` drop the key in the same commit. Why: "Remove obsolete pathways instead of leaving parallel legacy
  ones". Tests: the recorder's tests and pyright, and each of the four settings files loads without an unknown-key
  warning; no hd_trio run, as the path never ran there. Module Refactor agreed (2026-11-21).

## F. With an OAK-D at the PC

- F1. The `oak` backend stamps a frame and its tracklets with their capture, off the message's device timestamp as
  `oak4.messages.capture_time` does, instead of their arrival **(seen: `oak.Camera._emit_frame`,
  `_tracker_callback`)**; depthai 2 messages carry one **(seen: `getTimestampDevice` in `.venv`'s depthai stubs)**.
  Why: camera.md, "Stamp a frame once ... with its moment of capture where the device reports one".
  RECORDING_PLAN.md B's OAK-D capture time is this step; it comes after the camera plan, in his order video, camera,
  recording. Test: an OAK-D, white_space live, its latency reading against oak4's.

## Open

- The OAK4 is to run standalone, as an OAK App **(site fact)**. Depth alignment then needs a home: `ImageAlign` is
  the only alignment on the camera and fattens near objects ([Depth](oak4/OAK4.md#depth)), and its firmware cannot be
  changed. The options are to keep sending raw depth with its `DepthAlignment` to a PC that runs `DepthAlign`, or to
  run the same reprojection as a host node on the camera's CPU, whose speed for 144,000 points a frame has not been
  measured. A Script node is too slow for it **(deduction)**.
- Flow Motion's A/B of the depth's confidence needs a recording with it and a player moving. No recording with a
  confidence is kept: the two probes of 2026-11-21 were made to measure the encoder and are deleted.

## Rules it touches

- A callback list is a `utils.Broadcast`; a failing callback does not stop the others: A1
- Inside a package, code that does the same is written once: B1, E2
- Pass collaborators and fixed structure through constructors: B2
- Similar things share the noun: B3
- Components take their settings object in the constructor: B4
- A thread that waits for a moment waits on a `utils.Waker`: C1
- Read `time.perf_counter()` for every stamp and duration: C2
- Log exceptions with context; never fail silently: C3
- Follow PEP 8; one term everywhere; keep settings files in sync: C4
- Keep `Field` descriptions to one short line of panel text: C5
- The glossary: C6, C8
- Use `Lock` and `Event` for shared mutable state; prefer `IntEnum` over string keys: C7
- A document describes what is built: D1
- Use that one term everywhere it appears: E1
- Remove obsolete pathways instead of leaving parallel legacy ones: E3
- Stamp a frame once, with its moment of capture where the device reports one: F1
