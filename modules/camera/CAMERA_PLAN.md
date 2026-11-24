# The camera's plan

The camera package held up to the rules (2026-11-14, the apps' plan F; again 2026-11-21 and 2026-11-22). Who uses
what, from the code: the contract (`CameraSource`, `CameraFrame`, `CameraTracklet(s)`, the settings, `CameraHealth`,
the capture match) is read by every app and by inference, recording, tracker and render. The backends are each one
kind of camera, used now by the apps that have it: `oak` by white_space and hd_trio, `oak4` by deep_flow, low_flow
and smile, `vimba` and `webcam` by smile. That is general code with one user, which stays.

F is what is left of the scans, J recordings
through the camera's own models. **(seen)** is read in the code, **(deduction)** inferred.

## F. With an OAK-D at the PC

- F1. The live check of the OAK-D's capture stamp, which is built (`oak_device/capture_clock.py`, shared with the OAK4): an
  OAK-D, white_space live, its frames' and tracklets' latency readings against oak4's. Waits for an OAK-D at the PC: none
  was attached on 2026-11-22 **(seen: depthai 2 found no device)**. RECORDING_PLAN.md B's OAK-D capture time is this
  step, and so is the live check of each OAK-D's own pipeline (built 2026-11-22): every camera, and the OAK-D camera
  with clips in -sim, opens on its own pipeline, its geometry logged. The clip path through the device is checked in
  the same run: a clip tracklet's `timestamp` equals the recorded capture of the frame it was found in, which the
  clip camera's stamp rests on ([CAMERA.md, The clip camera](CAMERA.md#the-clip-camera)) **(deduction)**.

## J. Recordings through the camera's own models

Feeding a recording's frames into an OAK tests that camera's own detector, tracker and pose chain on a recorded
scene. Our recording format stays, depth is not fed back, and the replay (`modules/recording/replay`) stays as it is
(Matthias's decision of 2026-11-23). Only what an app uses is built and tested: the OAK-D's mono or color frame. No
OAK4 app is fed a recording; the OAK4's pipeline is made ready for it, without a clip input (Matthias, 2026-11-24).
The design leaves room for the rest (the OAK4's clip input, its grey warp, depth) without building it. The OAK-D does
this: `oak.Camera` with `clips` sends the player's video frames into the pipeline where the sensor's would enter
([CAMERA.md, The clip camera](CAMERA.md#the-clip-camera)); white_space's `studio.json` runs it (`replay` false)
**(seen)**. The OAK4 has no such path
**(seen)**. depthai 3 sends a host frame into a device node with `Node.Input.createInputQueue()`; its `XLinkIn` is
internal and made by depthai **(seen in `.venv-oak4`, depthai 3.11; the Luxonis porting guide)**. `add_detector`,
`add_tracker` and `add_pose_chain` take their frames as one `dai.Node.Output` **(seen)**.

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

- The OAK4's drop counts live, in the next OAK4 run on Matthias's go: with nobody in view `dropped_tracklets` stays at
  0, as it rests on `GatherData` sending a frame without players **(deduction)**.
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
  as the OAK-D's camera with clips, with no Neural Depth, no sensor controls and no IMU; the fed frame stamped with its recorded capture, as the OAK-D's. The Player is
  built with `nv12=True` in the OAK4 apps **(seen)**, so a clip frame needs no conversion, only its strides and planes.
  First a probe with the OAK4 attached, on Matthias's go: the crops and their keypoints land in the frame (an
  `ImgTransformation` made from the recorded lens and size, through `FrameCropper` and `remapPointTo`), a stamp set on
  the host frame comes back through `capture_time`, and 720p NV12 sent at 60 fps holds the rate **(deduction, not
  run)**. deep_flow and low_flow have no `replay` switch; their -sim always runs `replay.Camera` **(seen)**.

## Rules it touches

- Stamp a frame or tracklet message once, where the backend hands it over, with its moment of capture where the device reports one, else its arrival: F1
- Prefer composition over inheritance for runtime assembly: J2
- Run a test of the app or a GPU harness only on the user's go, and only what the plan names: F1, J2
