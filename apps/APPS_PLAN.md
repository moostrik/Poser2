# The apps' plan

The apps on deep_flow's input, this one plan: what is next, in order, each part's steps and how each is checked. How
the input works is in [INPUT_THREADS.md](../modules/flow/INPUT_THREADS.md); the recorder's plan for every app is in
[RECORDING_PLAN.md](../modules/recording/RECORDING_PLAN.md#2-the-recorder-in-the-other-apps). **(deduction)** marks what is inferred.


In this order: C white_space comes back on deep_flow's input; G hd_trio comes back on it, with today's 2D fluid;
F every module is held up to the rules, once every app runs on this branch. The rules they keep:
a callback only hands the item over and returns, the part that works on it does all its work on its own thread, a
queue that can't keep up drops its oldest and counts it, the camera's thread runs only the camera, and every drop is
counted on the panel ([INPUT_THREADS.md](../modules/flow/INPUT_THREADS.md)).

## C. white_space back

white_space (four mono OAK-D cameras at 23 fps, the panoramic tracker, the light) runs on `.venv` (depthai 2, the
`oak` backend), no depth. Its cameras send no keypoints, so the camera-keypoint path
deep_flow has reaches it only when they do. Every receiver works on its own thread, as in deep_flow
([INPUT_THREADS.md](../modules/flow/INPUT_THREADS.md)), but its input is a different model: deep_flow uploads frame
by frame, color first to start the pose and depth after it, paired by `CaptureSync`; white_space's four cameras'
frames of a moment travel together, synced first and uploaded as one set, cropped as one batch. So the uploader is
two, named by how they upload: deep_flow's `SerialUploader`, each frame on its own as it comes, and white_space's
`BatchUploader`, a moment's frames as one batch, which hd_trio takes too in G (its input is the same model). They
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
and a recording of it played back; deep_flow on `measure` 9–19 s); nothing is open from them.

The OAK-D's capture stamp, as the oak4's `capture_time`, needs an OAK-D to test; old show recordings keep playing,
new ones are the new layout. hd_trio's `main.py` still uses `CameraSync`'s `add_sync_callback` and follows in G (it
stops at start on this branch already).

Checked at the rig: the clip camera, which needs an OAK; and the playhead's smoothing (`playhead.tracking`,
`speed_smoothing`) against the motor's phase on `perf_counter`, whose error is 0.26° at p95 at 32 rpm against 4.3°
on `monotonic` (simulated with the motor's own code, 2026-11-13), so the smoothing may be stronger than it needs.
Done when white_space starts on this branch, its camera threads only hand over (under 1 ms per frame at p95), every
sync runs the pose step, the pose latency and the light's tick are no worse than on `main`, live and back to back,
with you at the rig, and its tests pass.

## F. The modules held up to the rules

Every package in `modules/`, one at a time in alphabetical order (`board`, `camera`, `flow`, `gl`, `inference`,
`inout`, `oak`, `panel`, `pose`, `recording`, `render`, `session`, `settings`, `tracker`, `utils`, `video`, and
`log_config.py`), against `CLAUDE.md` and `.claude/rules`: reported per package, a plan
for what does not meet them.

## G. hd_trio back

hd_trio (three color OAK-Ds, one player per camera, RVM's masks, the 2D fluid per screen) stops at its start on
this branch: its `main.py` still calls what C removed (`PoseInference.set_camera_frame`,
`add_gpu_image_callback`, `add_camera_images_callback`, `CameraSync.add_sync_callback`,
`TrackerFrameSource.process`). Its input is white_space's model, three cameras' frames of a moment as one set, so it
takes C's wiring. The 2D fluid stays as it is (`Motion2DLayer`, `Bridge2DLayer`, `Fluid2DLayer`, which the render
already uses); [the flow's line 5](../modules/flow/FLOW_PLAN.md#5-the-2d-flow-follows-the-3d) rebuilds it later.
Its recordings stay as they are (mp4 chunks any player plays, stored mirrored, a pose file per chunk), and so do its
pose-following center layers. One step per commit.

Built: 1, the pose model's crop settings are `pose_crop` in every app; 2, `PoseInference` cuts the
segmentation's crops beside the pose model's, given their settings (`add_segmentation_crop_callback`); 3, the
recorder's `mirrored` stores a frame still to be mirrored as shown, its tracklets with it, `flip_h` off; 4, hd_trio
runs on C's wiring (its -sim on `intro_1` at the clips' 23 sets a second, two poses a set, a mask a player a set,
no drop; a recording of it plays back the same, its poses on the players).

Checked at the rig: a live recording with `mirrored` on (a live frame comes with `flip_h` on, which a -sim clip
does not, so the run stored the clips as they came; the recorder's test covers the mirroring); the fluid's look.

Reported: 5, hd_trio held up to the rules; its findings are 7 to 9. Built: 6, the previous crop is off by default
(`crop.Settings.enable_prev_crop`): only the optical-flow model reads it, and no app turns it on; 7, the show's
sequencer takes the panel's start, stop, skip and playlist changes as requests carried out at the next tick, its
callbacks under their own lock and logged with their traceback, and the intro player plays on the tick's time;
both, and `main`, unbind on stop what they bind; 8, the render's constructor writes no settings (its `data.b`
values are the presets').

9. **Its document and names.** `HD_TRIO.md` holds the app's reasons, now in its comments and the settings'
   docstring, which say what each thing is; the render's names say what they hold (`lut`, `cmt`, `seq`); its
   imports in order. Checked by: hd_trio's tests and the guard tests.
10. **The inline paths go.** Once every app starts its threads, `CameraSync`, `CaptureSync`, `RecordingSync`,
   `TrackerFrameSource` and `PosePipeline` lose the path that matches a part on the caller's thread while their own
   does not run, and a submit before `start` is an error. Checked by: the apps' tests and the inference, tracker,
   pose and recording tests, and each app's start for 10 s: deep_flow, white_space and hd_trio in -sim, smile on
   its image input.

Parked for [the flow's line 5](../modules/flow/FLOW_PLAN.md#5-the-2d-flow-follows-the-3d), which rebuilds the 2D
fluid and its input: hd_trio's render writes `fluid.density.fade_time` every tick from the player's motion, and its
show stages write `similarity_scale` (and the center geometry's `stage`), settings used to carry values while
running; they reach the fluid and the color mask through their inputs once those are rebuilt, instead of changing
`Fluid2D`, which that line replaces, now.

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

- A callback only hands the item over and returns: C's wiring, the journal's raw tracklets.
- The part that works on the item has its own thread and does all its work there: the `BatchUploader`, the
  `CameraSync`s, the frame source, the pipeline, each started by white_space.
- When that part can't keep up, its queue drops the oldest item and counts it: the uploader's sets, the syncs.
- A part that only needs the latest value reads it at its own tick: the calibration layer's rows, the board's images.
- The camera's thread runs only the camera: C3, C4.
- Log exceptions with context, never fail silently: drops counted on the panel, C8.
- Remove obsolete pathways: the core upload written once, shared by both uploaders, C1; white_space's old wiring.
- Treat latency as a first-order concern: C's run compares with `main`.
- Update all call sites, remove obsolete pathways: white_space's video images go.
- Plans and commits: one commit per checked step, built on your go.
- A callback only hands the item over; the part that works on it has its own thread: G4, G6.
- Remove obsolete pathways: hd_trio's calls into what C removed, G4; the inline paths, G6.
- Name a concept after what it is; rename it everywhere in the same change: `segmentation_crop`, G4.

