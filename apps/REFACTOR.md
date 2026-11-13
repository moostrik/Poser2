# The app refactor

The apps on deep_flow's input, this one plan: what is next, in order, each part's steps and how each is checked. How
the input works is in [INPUT_THREADS.md](../modules/flow/INPUT_THREADS.md); the recorder's plan for every app is in
[RECORDING.md](../modules/recording/RECORDING.md#open). **(deduction)** marks what is inferred.


Four parts, in this order: A deep_flow's callbacks become hand-overs; B white_space comes back on that input; C smile
is checked; D hd_trio comes back after the 2D fluid ([the flow's line 5](../modules/flow/FLOW_PLAN.md#5-the-2d-flow-follows-the-3d)). The rules they keep:
a callback stores or enqueues what it receives and returns, a receiver does its work on its own thread, the camera's
thread runs only the camera, and every drop is counted on the panel ([INPUT_THREADS.md](../modules/flow/INPUT_THREADS.md)).

## A. deep_flow's callbacks as hand-overs

Four places run a receiver's work on another's thread (read on `deep-flow` at 5c74ad1b):

| Thread                        | Work it runs that is not its own                                                                         |
|-------------------------------|----------------------------------------------------------------------------------------------------------|
| the camera's                  | `RecordingSync` matching under its locks, from `Recorder.submit_frame` and `submit_tracklets`            |
| the uploader's                | `CaptureSync.submit_image` and `submit_depth`: the matching and the capture's emit                       |
| the uploader's                | `_on_image`: `PosesFromTracklets.process`, the box steps, `PoseInference.submit`, `expect_poses`         |
| the pose runner's callback    | the RAW emit, CLEAN, SMOOTH, `CaptureSync.submit_poses`, LERP's crossing (-sim: `RecordedPoses`')        |

Five steps, each one commit, checked by its tests and, on your go, by one run:

1. `RecordingSync` gets its own thread: its submits append to one inbox and return; its thread matches and hands
   each capture to the recorder. The inbox is not bounded: a recording needs every part and the matching is cheap;
   `HELD` after it stays.
2. `CaptureSync` matches on its own thread (it has one, for the deadlines): its submits append to an inbox, its
   thread drains it, waits for the next deadline in the same wait and emits; `_emit_lock` goes.
3. `PosesFromTracklets` gets its own thread: `submit(timestamp)` enqueues the sync (2 waiting, the oldest dropped
   and counted); its thread runs `process`, so the RAW stage's box steps, the crops and the model submit run there.
   `_on_image` keeps the submit and the camera interval; `expect_poses` comes from the new thread, by a callback
   with the sync's time and the poses asked per camera (one camera per sync in deep_flow, several in white_space).
4. The pose pipeline runs the steps after an async step on a thread of its own: the async step's results are
   enqueued (4 waiting, the oldest dropped and counted, `PosePipeline.dropped`); its thread runs the RAW emit,
   CLEAN, SMOOTH and the crossing. `RecordedPoses`' thread then only enqueues.
5. The six runners (pose, optical flow, segmentation; ONNX and TRT) lose their callback thread, which only came
   because the pipeline ran on it: they call their callbacks from the inference thread, which now only enqueue. A
   batch replaced before it ran is counted (`dropped` on the model's settings) instead of lost silently when the
   runner's `Queue(maxsize=2)` is full. The optical flow predictor, which calls its callbacks under its lock, copies
   the list under it and calls outside it. After step 4, not before.

Latency **(deduction)**: a hand-over is a notify and a wake, tens of µs on an idle core, up to the GIL's 5 ms
switch when the handing thread keeps running Python. A capture with its poses gets two more (steps 2 and 3, the
model's to SMOOTH +1 −1); against that the camera's and the uploader's threads no longer match or run the pose
step. Each step's run compares `CAPTURE_LATENCY`, `POSE_LATENCY_RAW`, `POSE_LATENCY_SMOOTH` and the drops, back
to back, on a new recording or live.

Decision: `CameraSync` (the frames' and the tracklets' sync, used by white_space and hd_trio) decides a sync in its
submit, on the caller's thread, and calls its callbacks under its lock. Recommended: it gets its own thread, as the
other syncs in steps 1 and 2, its submits only enqueue, and it calls its callbacks outside its lock; one more
hand-over before white_space's pose step. The other way: it stays inline as a cheap decision over a few dozen
timestamps, with only the lock fixed, and the rule then reads that a sync's decision is part of its store.

## B. white_space back

white_space (four mono OAK-D cameras at 23 fps, the panoramic tracker, the light) runs on `.venv` (depthai 2, the
`oak` backend), no depth; on this branch it stops at start. It moves onto A's result:

1. Each camera hands its frames and tracklets to the uploader (`Uploader(settings)`, `submit_frame`,
   `submit_tracklets`) and the recorder only.
2. The uploader's images go to the -sim detector, the board's `set_camera_image` (the previews) and the frames'
   `CameraSync`, whose sync is one uploaded image per camera, fired on the uploader's thread once every camera's
   image is on the GPU; its callback is `PosesFromTracklets.submit` (A3), which crops every camera's newest image
   (`PoseInference(crop, pose, uploader)`, its crops at the model's size already); the crops go to the board.
3. The uploader's tracklets go to the tracker (`submit_cam_tracklets`), the board, the tracker journal and the
   tracklets' `CameraSync`, each a store or an enqueue (the journal's `submit_cam_tracklets` checked for it); the
   sync's callback is `tracker.update`, which sets the tracker's event.
4. The recorder as [RECORDING.md](../modules/recording/RECORDING.md#open) plans it for every app, and the oak backend's
   capture time stamped as the oak4's is (RECORDING.md Open 2): white_space's old show recordings keep playing,
   new ones are the new layout.
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
RECORDING.md's plan), and its pose-following center layers, which stay. Planned when it starts.

## Details

- One line per stream, average and 1% low: the panel's rates as frame-time analysis gives them (RTSS,
  CapFrameX), one pinned line per stream, Render, Video, Captures and Motion,
  `<name>  <average> fps  low <1% low>`, both from one record of each stream's frame times. The average says how
  many frames came (a count for the cadence: 50 when none is lost); the 1% low, the rate over the slowest 1% of
  frames, says how steady, catching hitches without being set for a whole window by one, as the minimum (1 / the
  worst gap) is. It replaces the minimums (`Video min`, `Captures min`, `Motion min`) and gives the render the
  same two numbers; a frame-time line beside it only if asked.
- The color in one kernel: the color's conversion, straightening and mirroring as one CUDA kernel: from the
  camera's NV12 23 GPU ops become 2, about 0.4 ms less GPU a frame **(deduction)**. After the live run of
  [the flow's line 7](../modules/flow/FLOW_PLAN.md#7-recording-and--sim), which measures it on the camera.
- The previous crop: the optical flow on the crops (`enable_prev_crop`, off in deep_flow, so it runs nothing)
  does not wait for the crops' event, and its pairing of the current and the previous image is to be checked
  when it is used.
- No captures: while no capture comes (the camera stalls, -sim is paused) the bridge keeps its last output and
  the fluid keeps taking it, where the motion holds its last velocity. What the flow should do then is not
  decided.

## Rules it touches

- A callback stores or enqueues what it receives and returns: A's four places and B's wiring.
- A receiver does its work on its own thread: the recording sync, capture sync, pose step and pipeline get one.
- A callback list has its own lock, copied under it and called outside it: every list A touches.
- The camera's thread runs only the camera: A1, B1.
- A consumer that needs every item gets a callback; one that needs the latest reads the board: B2, B3.
- `submit` enqueues deferred work, `process` transforms and emits synchronously: both syncs' `submit_*`,
  `PosesFromTracklets.submit`.
- Log exceptions with context, never fail silently: drops counted on the panel.
- Treat latency as a first-order concern: each step's run compares the latency series.
- Update all call sites, remove obsolete pathways: the runners' dispatch thread goes, white_space's video images go.
- Plans and commits: one commit per checked step, built on your go.

