# The apps' plan

The apps on deep_flow's input, this one plan: what is next, in order, each part's steps and how each is checked. How
the input works is in [INPUT_THREADS.md](../modules/flow/INPUT_THREADS.md); the recorder's plan for every app is in
[RECORDING_PLAN.md](../modules/recording/RECORDING_PLAN.md#2-the-recorder-in-the-other-apps). **(deduction)** marks what is inferred.


Four parts, in this order: A deep_flow's callbacks become hand-overs; B white_space comes back on that input; C smile
is checked; D hd_trio comes back after the 2D fluid ([the flow's line 5](../modules/flow/FLOW_PLAN.md#5-the-2d-flow-follows-the-3d)). The rules they keep:
a callback stores or enqueues what it receives and returns, a receiver does its work on its own thread, the camera's
thread runs only the camera, and every drop is counted on the panel ([INPUT_THREADS.md](../modules/flow/INPUT_THREADS.md)).

## A. deep_flow's callbacks as hand-overs

Two places run a receiver's work on another's thread (read on `deep-flow` at 5c74ad1b):

| Thread                        | Work it runs that is not its own                                                                         |
|-------------------------------|----------------------------------------------------------------------------------------------------------|
| the uploader's                | `_on_image`: `PosesFromTracklets.process`, the box steps, `PoseInference.submit`, `expect_poses`         |
| the pose runner's callback    | the RAW emit, CLEAN, SMOOTH, `CaptureSync.submit_poses`, LERP's crossing (-sim: `RecordedPoses`')        |

A hand-over costs its receiver's wake-up, not more Python work: one thread runs Python at a time today too. The
receiver runs once it has the GIL, which the sender gives up at once when it blocks or enters a C call that
releases it (a queue wait, CUDA, OpenGL, numpy, TensorRT), and only at the switch interval (5 ms by default) when it
or another thread keeps running Python **(deduction)**; the camera's thread was seen waiting up to 5 ms for a busy
render thread ([INPUT_THREADS.md](../modules/flow/INPUT_THREADS.md)). So the hand-overs are measured before any is
added.

The steps left, each one commit, checked by its tests and, on your go, by one run, which shows its new queue's
wait (a `Wait …` timer in the board's time series) beside `CAPTURE_LATENCY`, `POSE_LATENCY_RAW` and
`POSE_LATENCY_SMOOTH`:

3. `PosesFromTracklets` gets its own thread: `submit(timestamp)` enqueues the sync with the images it is for (2
   waiting, the oldest dropped and counted); its thread runs `process`, so the RAW stage's box steps, the crops
   and the model submit run there, on the sync's own images, not the uploader's newest by then. `_on_image` keeps
   the submit and the camera interval; the recorder hears the poses asked from the new thread, by a callback named
   with the step (`add_asked_callback`), with the sync's time and the poses asked per camera.
4. The pose pipeline runs the steps after an async step on a thread of its own: the async step's results are
   enqueued (4 waiting, the oldest dropped and counted, `PosePipeline.dropped`); its thread runs the RAW emit,
   CLEAN, SMOOTH and the crossing. `RecordedPoses`' thread then only enqueues.
5. The six runners (pose, optical flow, segmentation; ONNX and TRT) lose their callback thread, which only came
   because the pipeline ran on it: they call their callbacks from the inference thread, which now only enqueue. A
   batch replaced before it ran is counted (`dropped` on the model's settings) instead of lost silently when the
   runner's `Queue(maxsize=2)` is full. After step 4, not before.
6. `CameraSync` (the frames' and the tracklets' sync of white_space and hd_trio) gets its own thread, as the other
   syncs: its submits enqueue (16 per camera, the oldest dropped and counted), its thread decides the sync and
   calls its callbacks outside its lock. Checked by its tests; its run is B's.

The callback lists these steps touch each get their own lock, copied under it and called outside it: the
uploader's (`add_gpu_image_callback`, `add_gpu_depth_callback`, `add_tracklet_callback`, which register without a
lock today), `CameraSync`'s (called under its lock today), the optical flow predictor's
(called under its lock today) and the runners'.

Each new thread is started by `main.start` after the wiring, before the cameras, and stopped by `main.stop` after
the cameras and the uploader, before the pose model, in the order the data flows; a stop wakes the thread, which
drops what still waits and logs the count. A step adds its thread to both in its commit.

If a wait comes out in milliseconds: the switch interval is lowered (`sys.setswitchinterval`, 0.5 ms
to start), and a sender that keeps running Python after it queues is changed to give up the GIL. If that still
does not hold the latency series where they were, a cheap matcher (`CameraSync`) stays on its
sender's thread as a stated exception to the rule, chosen by the measurement.

## B. white_space back

white_space (four mono OAK-D cameras at 23 fps, the panoramic tracker, the light) runs on `.venv` (depthai 2, the
`oak` backend), no depth; on this branch it stops at start. It moves onto A's result, and its step 4 waits for the
recording thread's recorder for every app and the oak backend's capture time
([RECORDING_PLAN.md](../modules/recording/RECORDING_PLAN.md#2-the-recorder-in-the-other-apps)):

1. Each camera hands its frames and tracklets to the uploader (`Uploader(settings)`, `submit_frame`,
   `submit_tracklets`) and the recorder only.
2. The uploader's images go to the -sim detector, the board's `set_camera_image` (the previews) and the frames'
   `CameraSync`, whose sync is one uploaded image per camera, fired on the uploader's thread once every camera's
   image is on the GPU; its callback is `PosesFromTracklets.submit` (A3), with every camera's image of the sync,
   which it crops (`PoseInference(crop, pose, uploader)`, its crops at the model's size already); the crops go to
   the board.
3. The uploader's tracklets go to the tracker (`submit_cam_tracklets`), the board, the tracker journal and the
   tracklets' `CameraSync`, each a store or an enqueue (the journal's `submit_cam_tracklets` checked for it); the
   sync's callback is `tracker.update`, which sets the tracker's event.
4. The recorder in white_space, as [RECORDING_PLAN.md](../modules/recording/RECORDING_PLAN.md#2-the-recorder-in-the-other-apps)
   plans it, and the oak backend's capture time stamped as the oak4's is: white_space's old show recordings keep
   playing, new ones are the new layout.
5. The light's calibration layer (`test_calibration.py`) reads the uploaded image instead of the CPU frame the
   camera's thread puts on the board (`_store_video_frame` and the board's video images go): it reduces its band of
   rows to one row on the GPU, queues the row's copy into pinned memory with an event, and uses on a later tick the
   newest row whose event has passed, so nothing waits.
6. The clip camera (`oak.ClipCamera`, a recording played through an OAK) hands over as the camera does.
7. The drops on the panel as deep_flow's: the uploader's, the pose step's, the pipeline's, the runners'.
8. The probe (`scripts/cadence_probe.py`) takes the app as an argument: the camera's thread per frame, the syncs
   per second against the cameras' rate, the pose rate per camera, the pose latency (RAW, SMOOTH), the drops.

Done when white_space starts on this branch, its camera threads only hand over (under 1 ms per frame at p95),
every sync runs the pose step, the pose latency and the light's tick are no worse than on `main` (live, back to
back), and its tests pass. One uploader thread serves the four cameras (92 frames a second); the probe says
whether it keeps up, and one per camera follows only if not.

## C. smile checked

smile imports `modules.camera`, `gl`, `face`, `video`, `inout`, `settings`, `panel` and `utils`, not `inference`
or `board`, and was committed today (its document, `SMILE.md`). The check: whether it runs on this branch, which of
its threads run a receiver's work in a callback (its camera's frame callback into the render and the face finder),
and what moving it onto the rules would take; reported, nothing changed.

## D. hd_trio back, after the 2D fluid

hd_trio comes back once [the flow's line 5](../modules/flow/FLOW_PLAN.md#5-the-2d-flow-follows-the-3d) has rebuilt the 2D fluid, on B's wiring. Known
now: its crops (the pose model's at its input size, RVM's at 768 × 1024 from the same region, both by
`CropExtractor`), its recordings as before (mp4 chunks any player plays, stored mirrored, a pose file per chunk,
RECORDING_PLAN.md), and its pose-following center layers, which stay. Planned when it starts.

## Details

- One line per stream, average and 1% low: the panel's rates as frame-time analysis gives them (RTSS,
  CapFrameX), one pinned line per stream, Render, Video, Captures and Motion,
  `<name>  <average> fps  low <1% low>`, both from one record of each stream's frame times. The average says how
  many frames came (a count for the cadence: 50 when none is lost); the 1% low, the rate over the slowest 1% of
  frames, says how steady, catching hitches without being set for a whole window by one, as the minimum (1 / the
  worst gap) is. It replaces the minimums (`Video min`, `Captures min`, `Motion min`) and gives the render the
  same two numbers, in deep_flow and white_space alike; a frame-time line beside it only if asked.

## Rules it touches

- A callback stores or enqueues what it receives and returns: A's two places and B's wiring.
- A receiver does its work on its own thread: the pose step, pipeline and `CameraSync` get one; a matcher stays
  inline only if its wait says so, as a stated exception.
- A callback list has its own lock, copied under it and called outside it: the uploader's, `CameraSync`'s, the
  optical flow predictor's and the runners'.
- The camera's thread runs only the camera: B1.
- A consumer that needs every item gets a callback; one that needs the latest reads the board: B2, B3.
- `submit` enqueues deferred work, `process` transforms and emits synchronously: `CameraSync`'s `submit_*`,
  `PosesFromTracklets.submit`.
- Log exceptions with context, never fail silently: drops counted on the panel.
- Treat latency as a first-order concern: each step's run compares the latency series.
- Update all call sites, remove obsolete pathways: the runners' dispatch thread goes, white_space's video images go.
- Plans and commits: one commit per checked step, built on your go.

