# The apps' plan

The apps on deep_flow's input, this one plan: what is next, in order, each part's steps and how each is checked. How
the input works is in [INPUT_THREADS.md](../modules/flow/INPUT_THREADS.md); the recorder's plan for every app is in
[RECORDING_PLAN.md](../modules/recording/RECORDING_PLAN.md#2-the-recorder-in-the-other-apps). **(deduction)** marks what is inferred.


Four parts, in this order: B the frame source and its images; C white_space comes back on deep_flow's input; D smile is
checked; E hd_trio comes back after the 2D fluid ([the flow's line 5](../modules/flow/FLOW_PLAN.md#5-the-2d-flow-follows-the-3d)). The rules they keep:
a callback stores or enqueues what it receives and returns, a receiver does its work on its own thread, the camera's
thread runs only the camera, and every drop is counted on the panel ([INPUT_THREADS.md](../modules/flow/INPUT_THREADS.md)).

## B. The frame source and its images

The pipeline's one source of poses is the `FrameSource` (`modules/pose/pipeline/spec.py`): on each image it
pushes a pose per tracked player, from the player's newest tracklet (its box, and the camera's keypoints, azimuth
and distance where it has them), never waiting for tracklets. A pose may hold only its box, so the RAW stage
smooths and predicts boxes with the pipeline's own filters before the async step crops the image and adds the
keypoints. That source is `TrackerFrameSource` (`modules/tracker`); it pushes its poses with the images they are
for, and the crop cuts exactly those (INPUT_THREADS.md). white_space and hd_trio call its `process` without images,
which `PoseInference` refuses, so they stop at start until C and E wire their syncs to `submit`.

4. **The camera's keypoints come out.** A tracklet from an OAK4 pose chain carries the camera's keypoints, and the
   source puts them on the pose (`Points2D` in the crop's window, `keypoint_window`); then the async step drops
   them: with the host pose model on, its points replace them, and with it off `pose.Predictor.submit` returns
   without emitting, so no pose reaches RAW. The check of `pose.enabled` moves from `pose.Predictor.submit` to
   `PoseInference`: with the host model off it still cuts the crops (the optical flow and the board read them) and
   hands on the poses that carry the camera's points as they are, as `RecordedPoses` hands on recorded ones; with
   it on, the model's points replace them, as now. deep_flow's `posing` (what the recorder hears of the poses
   asked) counts the poses carrying the camera's points, so a recording waits for them. deep_flow has the path
   now; white_space has it when its cameras send keypoints (the `oak` backend, depthai 2, sends none).

Checked by: the pose step's (with the host model off a pose with the camera's points comes out with them, one
without none), the recorded poses' tests updated; then one run on `measure` 9–19 s against A6's numbers (capture
latency, pose RAW and SMOOTH latency, poses out of RAW). One commit per step.

## C. white_space back

white_space (four mono OAK-D cameras at 23 fps, the panoramic tracker, the light) runs on `.venv` (depthai 2, the
`oak` backend), no depth; on this branch it stops at start. It moves onto deep_flow's input, every receiver on its
own thread ([INPUT_THREADS.md](../modules/flow/INPUT_THREADS.md)), and its step 4 waits for the
recording thread's recorder for every app and the oak backend's capture time
([RECORDING_PLAN.md](../modules/recording/RECORDING_PLAN.md#2-the-recorder-in-the-other-apps)):

1. Each camera hands its frames and tracklets to the uploader (`Uploader(settings)`, `submit_frame`,
   `submit_tracklets`) and the recorder only.
2. The uploader's images go to the -sim detector, the board's `set_camera_image` (the previews) and the frames'
   `CameraSync`, whose sync is one uploaded image per camera, fired on its own thread once every camera's image is
   on the GPU; its callback is `TrackerFrameSource.submit`, with every camera's image of the sync, which the
   async step crops at the model's size; the crops go to the board.
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

Each thread (the uploader, the two `CameraSync`s, the pose step, the pipeline) is started by `main.start` after the
wiring, before the cameras, and stopped by `main.stop` after the cameras and the uploader, before the pose model.
If a sync's wait (`Wait camera sync …`) comes out in milliseconds: the switch interval is lowered
(`sys.setswitchinterval`, 0.5 ms to start), and a sender that keeps running Python after it queues is changed to
give up the GIL; if that still does not hold the latencies, the sync stays on its sender's thread as a stated
exception to the rule, chosen by the measurement.

Done when white_space starts on this branch, its camera threads only hand over (under 1 ms per frame at p95),
every sync runs the pose step, the pose latency and the light's tick are no worse than on `main` (live, back to
back), and its tests pass. One uploader thread serves the four cameras (92 frames a second); the probe says
whether it keeps up, and one per camera follows only if not.

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
  capture time: open, not part of B.

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
- The camera's thread runs only the camera: C1.
- A consumer that needs every item gets a callback; one that needs the latest reads the board: C2, C3.
- Log exceptions with context, never fail silently: drops counted on the panel, C7.
- Treat latency as a first-order concern: B's run compares with A6's numbers, C's with `main`.
- Update all call sites, remove obsolete pathways: white_space's video images go.
- Plans and commits: one commit per checked step, built on your go.

