# The camera's plan

The camera package held up to the rules (2026-11-14, the apps' plan F; again 2026-11-21). Who uses what, from the
code: the contract (`CameraSource`, `CameraFrame`, `CameraTracklet(s)`, the settings, `CameraHealth`, the capture
match) is read by every app and by inference, recording, tracker and render. The backends are each one kind of
camera, used now by the apps that have it: `oak` by white_space and hd_trio, `oak4` by deep_flow, low_flow and
smile, `vimba` and `webcam` by smile. That is general code with one user, which stays.

The steps below are what is left of the scan of 2026-11-21. **(seen)** is read in the code,
**(deduction)** inferred.

## E. One host-side flip

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

- Remove obsolete pathways instead of leaving parallel legacy ones: E3
- Stamp a frame once, with its moment of capture where the device reports one: F1
