# The apps' plan

The apps on deep_flow's input, this one plan: what is next, in order, each part's steps and how each is checked. How
the input works is in [INPUT_THREADS.md](../modules/flow/INPUT_THREADS.md); the recorder's plan for every app is in
[RECORDING_PLAN.md](../modules/recording/RECORDING_PLAN.md#2-the-recorder-in-the-other-apps). **(deduction)** marks what is inferred.


Three parts, in this order: C white_space comes back on deep_flow's input; D smile is checked; E hd_trio comes back after the 2D fluid ([the flow's line 5](../modules/flow/FLOW_PLAN.md#5-the-2d-flow-follows-the-3d)). The rules they keep:
a callback stores or enqueues what it receives and returns, a receiver does its work on its own thread, the camera's
thread runs only the camera, and every drop is counted on the panel ([INPUT_THREADS.md](../modules/flow/INPUT_THREADS.md)).

## C. white_space back

white_space (four mono OAK-D cameras at 23 fps, the panoramic tracker, the light) runs on `.venv` (depthai 2, the
`oak` backend), no depth; on this branch it stops at start: it calls the frame source's `process` without the
images the crop needs, which `PoseInference` refuses. Its cameras send no keypoints, so the camera-keypoint path
deep_flow has reaches it only when they do. Every receiver works on its own thread, as in deep_flow
([INPUT_THREADS.md](../modules/flow/INPUT_THREADS.md)), but its input is a different model: deep_flow uploads frame
by frame, color first to start the pose and depth after it, paired by `CaptureSync`; white_space's four cameras'
frames of a moment travel together, synced first and uploaded as one set, cropped as one batch. So white_space gets
an uploader of its own, and the two share the core upload. An upload's cost is not its copy (about 0.1 ms) but its
CUDA launches, which block while the GPU's queue is full, and that grows with the launches, not the pixels
**(deduction: 6–7 ms per color upload for a 0.1 ms copy)**: four frames one by one are four rounds of launches, a
set is one, and both cost one round after the last camera's frame, so the latency is about equal and the upload's
load about a quarter.

The frame source's call already takes a set (`TrackerFrameSource.submit(cam_times, images)`, each player judged on
its own camera's capture time, the poses asked told per camera).

1. **The core upload, shared.** The GPU work on color frames moves out of deep_flow's `Uploader` into one piece both
   uploaders use: the pinned staging, the copy, the conversion (NV12, BGR or mono to RGB), the straightening with
   the camera's grid, the flip, the ready event, for one frame or a stack of frames of one size (one round of
   kernels over the stack, the grids stacked, one flip per mirrored camera). deep_flow's `Uploader` keeps the rest
   as it is: frame by frame, depth and its alignment, the tracklets' straightening, `snapshot()`; it calls the core
   with one frame, so its upload does not change.
2. **white_space's uploader.** Its own class beside `Uploader`, with its own thread: a synced set of the cameras'
   frames in (`submit_set`), queued whole, two sets at most, beyond that the oldest set dropped whole and counted
   (`dropped`), so a frame never leaves its set; the core uploads the set as one stack; out go each image to its
   callbacks (the -sim detector, the board's `set_camera_image`) and the set to a set callback, whose receiver
   calls `TrackerFrameSource.submit` with each camera's capture time; `snapshot()` as deep_flow's. No tracklets and
   no depth.
3. **The sync before the upload.** Each camera hands its frames to the frames' `CameraSync` and the recorder, and
   nothing else. `CameraSync`'s callback hands on a set: `add_set_callback(callback(items))`, `items` by `cam_id`,
   each with its capture time. The frames' sync keeps each camera's newest video frame and hands the set to
   white_space's uploader; the tracklets' sync hands its set of messages to `tracker.update`, which takes the
   newest of their times, as its `(timestamp)` did.
4. **The tracklets, as they come.** Each camera hands its tracklet messages straight to the tracker
   (`submit_cam_tracklets`), the board, the tracker journal, the tracklets' `CameraSync` and the recorder, each a
   store or an enqueue (the journal's `submit_cam_tracklets` checked for it), never through an uploader: the `oak`
   backend's frames carry no homography and no mirror (the device warps and flips), so there is nothing to
   straighten.
5. **The recorder.** Built for every app ([RECORDING_PLAN.md](../modules/recording/RECORDING_PLAN.md#2-the-recorder-in-the-other-apps)):
   white_space's settings and wiring, `expect_parts` per camera, `expect_poses` per camera and image (the frame source's asked
   callback). The OAK-D's capture stamp, as the oak4's `capture_time`, needs an OAK-D to test. Old show recordings
   keep playing; new ones are the new layout.
6. **The calibration layer** (`test_calibration.py`) reads the uploaded image instead of the CPU frame the camera's
   thread puts on the board: it reduces its band of rows to one row on the GPU, queues the row's copy into pinned
   memory with an event, and uses on a later tick the newest row whose event has passed, so nothing waits.
7. **The clip camera** (`oak.ClipCamera`, a recording played through an OAK) hands over as the camera does;
   checked only at the rig, as it needs an OAK.
8. **The drops on the panel**, as deep_flow's: the uploader's sets, the sync's, the frame source's, the pipeline's,
   the runners'; the probe reads them in C's run.
9. **A white_space probe** (`scripts/white_space_probe.py`), beside deep_flow's, which is built on deep_flow's
   classes: the camera's thread per frame, the syncs per second against the cameras' rate, the uploader's time per
   set, the pose rate per camera, the pose latency (RAW, SMOOTH), the tracker's update rate, the drops.

What goes from white_space's wiring: `PoseInference(crop, pose)` without the uploader, `set_camera_frame`,
`add_gpu_image_callback`, `add_camera_images_callback`, `_store_video_frame` and the board's video images, and the
frame sync's callback to `process`. `CameraSync`'s `add_sync_callback` goes for `add_set_callback`; hd_trio's
`main.py` uses the old one and follows in E (it stops at start on this branch already).

Each thread (white_space's uploader, the two `CameraSync`s, the frame source, the pipeline) is started by `main.start` after
the wiring, before the cameras, and stopped by `main.stop` after the cameras and the uploader, before the pose
model. If a sync's wait (`Wait camera sync …`) comes out in milliseconds: the switch interval is lowered
(`sys.setswitchinterval`, 0.5 ms to start), and a sender that keeps running Python after it queues is changed to
give up the GIL; if that still does not hold the latencies, the sync stays on its sender's thread as a stated
exception to the rule, chosen by the measurement.

Assumed, to confirm while building: the four cameras deliver one frame size (the stack needs it; frames of another
size go up as their own stack).

Checked by: `CameraSync`'s tests (a set per sync, a camera's newest frame), the core's (a stack uploaded at once
gives each image as one by one does), deep_flow's uploader tests unchanged, white_space's uploader's (a set in, its
images and the set out; the oldest whole set dropped), the
frame source's (the asked callback per camera; a player judged on its camera's time), the calibration layer's (its
row reduced on the GPU matches the CPU row of a recorded frame), white_space's pipeline spec; the recorder's wiring
by a short -sim recording on the same recording, played back;
a -sim run with the probe on `20250514-130512_4_wide_mono`, an old-layout recording without recorded tracklets, so
in replay with the host detector (`simulator.replay`, `yolo`), not `ClipCamera`, which needs an OAK: the uploader's
time per set, the pose latency and the tracker's update rate.
Done when white_space starts on this branch, its camera threads only hand over (under 1 ms per frame at p95), every
sync runs the pose step, the pose latency and the light's tick are no worse than on `main`, live and back to back,
with you at the rig, and its tests pass.

## D. smile checked

smile imports `modules.camera`, `gl`, `face`, `video`, `inout`, `settings`, `panel` and `utils`, not `inference`
or `board`, and was committed today (its document, `SMILE.md`). The check: whether it runs on this branch, which of
its threads run a receiver's work in a callback (its camera's frame callback into the render and the face finder),
and what moving it onto the rules would take; reported, nothing changed.

## E. hd_trio back, after the 2D fluid

hd_trio comes back once [the flow's line 5](../modules/flow/FLOW_PLAN.md#5-the-2d-flow-follows-the-3d) has rebuilt the 2D fluid, on C's wiring. Known
now: its crops (the pose model's at its input size, RVM's at 768 × 1024 from the same region, both by
`CropExtractor`), its recordings as before (mp4 chunks any player plays, stored mirrored, a pose file per chunk,
RECORDING_PLAN.md), and its pose-following center layers, which stay. Planned when it starts.

Its last step: once every app starts its threads, the inline path goes from `RecordingSync`, `CaptureSync`,
`TrackerFrameSource`, `PosePipeline` and `CameraSync` (a part matched on the thread it comes on while the thread
does not run), and a submit before `start` is an error.

## Details

- The box predictor's `lead`, a fixed 1/30 s past the tracklet's stamp, against the box predicted to the image's own
  capture time: open.

- One line per stream, average and 1% low: the panel's rates as frame-time analysis gives them (RTSS,
  CapFrameX), one pinned line per stream, Render, Video, Captures and Motion,
  `<name>  <average> fps  low <1% low>`, both from one record of each stream's frame times. The average says how
  many frames came (a count for the cadence: 50 when none is lost); the 1% low, the rate over the slowest 1% of
  frames, says how steady, catching hitches without being set for a whole window by one, as the minimum (1 / the
  worst gap) is. It replaces the minimums (`Video min`, `Captures min`, `Motion min`) and gives the render the
  same two numbers, in deep_flow and white_space alike; a frame-time line beside it only if asked.

## Rules it touches

- A callback stores or enqueues what it receives and returns: C's wiring.
- A receiver does its work on its own thread: white_space starts the `CameraSync`s' threads; a sync stays inline
  only if its wait says so, as a stated exception.
- The camera's thread runs only the camera: C3, C4.
- A consumer that needs every item gets a callback; one that needs the latest reads the board: C3, C4.
- Log exceptions with context, never fail silently: drops counted on the panel, C8.
- Remove obsolete pathways: the core upload written once, shared by both uploaders, C1; white_space's old wiring.
- Treat latency as a first-order concern: C's run compares with `main`.
- Update all call sites, remove obsolete pathways: white_space's video images go.
- Plans and commits: one commit per checked step, built on your go.

