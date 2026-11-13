# The apps' plan

The apps on deep_flow's input, this one plan: what is next, in order, each part's steps and how each is checked. How
the input works is in [INPUT_THREADS.md](../modules/flow/INPUT_THREADS.md); the recorder's plan for every app is in
[RECORDING_PLAN.md](../modules/recording/RECORDING_PLAN.md#2-the-recorder-in-the-other-apps). **(deduction)** marks what is inferred.


Three parts, in this order: C white_space comes back on deep_flow's input; D smile is checked; E hd_trio comes back after the 2D fluid ([the flow's line 5](../modules/flow/FLOW_PLAN.md#5-the-2d-flow-follows-the-3d)). The rules they keep:
a callback stores or enqueues what it receives and returns, a receiver does its work on its own thread, the camera's
thread runs only the camera, and every drop is counted on the panel ([INPUT_THREADS.md](../modules/flow/INPUT_THREADS.md)).

## C. white_space back

white_space (four mono OAK-D cameras at 23 fps, the panoramic tracker, the light) runs on `.venv` (depthai 2, the
`oak` backend), no depth. Its cameras send no keypoints, so the camera-keypoint path
deep_flow has reaches it only when they do. Every receiver works on its own thread, as in deep_flow
([INPUT_THREADS.md](../modules/flow/INPUT_THREADS.md)), but its input is a different model: deep_flow uploads frame
by frame, color first to start the pose and depth after it, paired by `CaptureSync`; white_space's four cameras'
frames of a moment travel together, synced first and uploaded as one set, cropped as one batch. So the uploader is
two, named by how they upload: deep_flow's `SerialUploader`, each frame on its own as it comes, and white_space's
`BatchUploader`, a moment's frames as one batch, which hd_trio takes too in E (its input is the same model). They
share the core upload, `ImageUpload`. Both hand on what they uploaded, each camera's image with its previous one
(the optical flow's previous crop, in deep_flow and hd_trio) and the ready event, as one `ImageSnapshot`, through
a set callback, so a receiver gets exactly the images it is for. An upload's cost is not its copy (about 0.1 ms)
but its
CUDA launches, which block while the GPU's queue is full, and that grows with the launches, not the pixels
**(deduction: 6–7 ms per color upload for a 0.1 ms copy)**: four frames one by one are four rounds of launches, a
set is one, and both cost one round after the last camera's frame, so the latency is about equal and the upload's
load about a quarter.

The frame source's call already takes a set (`TrackerFrameSource.submit(cam_times, images)`, each player judged on
its own camera's capture time, the poses asked told per camera).

The core is built (`ImageUpload`, for one frame or a stack of one size), deep_flow's `SerialUploader` uses it and
hands each color image on as an `ImageSnapshot` of one (`add_snapshot_callback`), and `PoseInference` takes no
uploader. white_space's `BatchUploader` is built beside it: a set in (`submit_set`), queued whole, two at most, the
oldest set dropped whole and counted; uploaded as one stack per frame size; each image to its image callbacks and
the set as its `ImageSnapshot` to its snapshot callbacks. `CameraSync` hands on a set
(`add_set_callback(callback(items))`, each camera's newest message by `cam_id`, a camera more than a frame behind
left out). white_space is wired on them: the frames' sync hands its set to the `BatchUploader`, whose snapshot
callback calls `TrackerFrameSource.submit`; the tracklets' sync hands its set to `tracker.update` at the newest of
their times; the tracklets go as they come to the tracker, the board, the journal (which queues them raw) and the
tracklets' sync; the threads start after the wiring, before the cameras. The recorder takes the frames and
tracklets from each camera as they come, `expect_parts` per camera, `expect_poses` per camera and set (the frame
source's asked callback) and the RAW stage's poses. The calibration layer reads the uploaded image (`CameraColumns`:
its band reduced to one row on the GPU, read back a tick later without waiting); the board's video images are gone.
The drops are pinned on the panel (`cadence`: the two syncs', the uploader's sets, the pipeline's; the frame
source's on its own panel). The clip camera (`oak.ClipCamera`) needs no change: its frames come out of the device
as the camera's do, and through the same callbacks without one (`clip_passthrough`). The probe is
`scripts/white_space_probe.py`.

The -sim runs are done (white_space on `20250514-130512_4_wide_mono` 23–31 s in replay with its detector engine,
and a recording of it played back; deep_flow on `measure` 9–19 s). Open from them:

- The syncs hand on 16.6 sets per second from four cameras at 23 fps, 46 of 101 gaps over 50 ms (measured); the
  rule that picks the camera completing a set is unchanged by C, and how it skips sets is not yet traced.
- deep_flow's color upload went from 6.1 to 9.0 ms mean, its latencies 2–4 ms with it, one run each (measured);
  the cause, the shared `ImageUpload`'s stack per single frame or the snapshot callback inside the timed span, is
  inferred.
- The recording over 6.8 s of playhead holds 127 frames per camera, about 1.3 s fewer, none counted missed or
  dropped (measured); the recorder's start is not yet traced.
- white_space's preset leaves `person_detector.model_type` at ONNX; its engine is the app's
  `yolov8n_coco_512x448.trt` (TRT).

The OAK-D's capture stamp, as the oak4's `capture_time`, needs an OAK-D to test; old show recordings keep playing,
new ones are the new layout. hd_trio's `main.py` still uses `CameraSync`'s `add_sync_callback` and follows in E (it
stops at start on this branch already).

Checked at the rig: the clip camera, which needs an OAK.
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

