# The input threads

A frame goes from the camera to the screen through several threads. Each thread does its own work and hands
on; a hand-off returns at once, so no thread waits for another's work, and a frame is dropped only where a
stage cannot keep up, counted, never because a part of it is late. Nothing but the camera runs on a camera's
thread, and no thread waits for the GPU while the GPU has room. deep_flow runs this way; the other apps follow
([REFACTOR_PLAN.md](../../REFACTOR_PLAN.md)).

"Code" is what the files show; **(deduction)** marks what is inferred from them, **(site fact)** what the
operator saw.

## deep_flow's threads

Per thread, the callbacks in the order they are registered (`apps/deep_flow/main.py`):

| Thread                                            | Runs                                                           | Waits for                                  |
|---------------------------------------------------|----------------------------------------------------------------|--------------------------------------------|
| the camera's (depthai; -sim: the player's reader) | queues each frame for the recorder (live) and the uploader     | nothing                                    |
| the uploader's (`SerialUploader`)                 | uploads each frame and hands it on ([below](#the-uploader))    | the GPU, in a copy while its queue is full |
| the recording sync's (`RecordingSync`)            | matches a capture's parts, hands the capture to the recorder   | nothing                                    |
| the tracklets' (depthai)                          | queues them for the tracker, the board; `tracker.update`       | nothing                                    |
| the tracker's                                     | associates the tracklets, to the frame source and the board    | nothing                                    |
| the frame source's (`TrackerFrameSource`)         | a pose per player, the RAW box steps, the crops, the submit    | nothing                                    |
| the pose model's (`pose.PoseEstimator`)           | RTMPose on the crops, queues the poses for the pipeline        | its own model                              |
| the pipeline's (`PosePipeline raw`)               | the RAW stage after the model, CLEAN, SMOOTH, their users      | nothing                                    |
| the detector's (-sim)                             | YOLOv8n on the uploaded image, boxes to the replay camera      | its own model                              |
| the inference frame sync's (`InferenceFrameSync`) | matches the image, depth and poses, hands the frame on         | the next part, or `pose_wait` (0 here)     |
| the render                                        | every capture since the last tick, then the fluid              | the GPU, in a copy while its queue is full |

### The uploader

The camera's one hand-off, the `SerialUploader`: each frame onto the GPU on its own, the moment it comes, two
frames of a kind per camera waiting at most (the oldest dropped, `dropped`). Its own thread makes the corrected
camera frame, the color and the depth aligned onto it, as [INFERENCE.md](../inference/INFERENCE.md#the-upload)
describes.

The line between the uploader and the flow: what every user of the camera needs (the color's conversion and
straightening, the depth's alignment by the camera's calibration) is the uploader's and reads camera settings
only; what only the flow needs (the view's crop and size, the depth's read into it at its
`tolerance`, the mask) is the flow's view stage ([VIEW.md](VIEW.md)).

When an image is on the GPU, the uploader hands it to four others, each of which only takes note and returns:

- the inference frame sync, which queues the image for its own thread;
- the detector (-sim), which keeps the newest image for its own thread;
- the camera-interval time sample;
- through the snapshot callback, the frame source, which queues the frame bang with exactly that image and its
  previous one, two waiting at most (the oldest dropped, `dropped` on the panel). On its own thread `TrackerFrameSource.process`
  makes a pose per player from the tracker's newest boxes and pushes them with those images; the RAW stage's two
  box filters smooth them, and the pipeline hands the images to the crop step, `CropExtractor.submit(frames, images)`,
  which cuts the crops from exactly those, this very image and not a newer one, at the pose model's input size, queued on the GPU and not waited for; it records a "crops ready" event and hands crops, poses and event to
  the pose model, whose thread has RTMPose wait for that event on its own stream; with the pose model off
  (`pose.enabled`), the poses carrying the camera's own keypoints (an OAK4 pose chain) go on as they are. Then the
  recorder hears whether poses come for the image. The uploader knows nothing of poses.


### No camera sync

With one camera the frames' and the tracklets' `CameraSync` only throttle: they skip a sync within 0.75 of a frame
of the last, an uncounted drop. deep_flow runs the pose step on every uploaded image and `tracker.update` on every
tracklet message; `CameraSync` is for the apps with several cameras.

### The inference frame sync

`InferenceFrameSync` ([INFERENCE.md](../inference/INFERENCE.md#the-inference-frame-sync)) matches an image, its depth and the
poses of one moment. With `pose_wait` 0 a frame carries the poses of the frame before, which with the view's crop
fixed only puts the pose's points drawn over the view a frame behind.

### The hand-overs' waits

How long an item waits in a queue for its receiver, the `Wait …` timers in the time series (`measure` 9–19 s,
-sim, the recorded tracklets, the pose model in the app, `scripts/cadence_probe.py`):

| Queue                            | Mean     | p95      | Max      |
|----------------------------------|----------|----------|----------|
| the uploader's frames            | 4.2 ms   | 17.8 ms  | 23.7 ms  |
| the uploader's tracklets         | 3.2 ms   | 15.6 ms  | 23.3 ms  |
| the pose step's bangs            | 0.16 ms  | 0.47 ms  | 1.4 ms   |
| the inference frame sync's parts | 0.30 ms  | 0.80 ms  | 2.6 ms   |
| the pipeline's poses             | 0.09 ms  | 0.17 ms  | 1.0 ms   |

The uploader's wait is its own work on the frame before it, not the hand-over **(deduction)**: a color frame takes
it 6.4 ms on average to upload.

### The render: every capture

The board keeps each camera's inference frames until the render takes them (`InferenceFrameStoreMixin.take_inference_frames`, at most
4; beyond that the oldest is dropped and counted). Each tick the render runs every capture since the last, in
order (`_run_inference_frame` in `apps/deep_flow/render/deep_flow_render.py`): its image and depth into GL, the view stage (its color grade
the last of its parts), the motion, and the bridge stepping over the time since the frame before. A tick without a frame runs
none of them, so no stage works on a frame twice; the fluid runs every tick on the tick's `dt`.

With two inference frames in one tick the 3D bridge hands the fluid, per pixel, the density and the temperature of the
frame with the larger amount (`FrameMax`: a density's amount, with its own color, a temperature's size,
with its own warmth), and the newest frame's velocity and z position: a lost frame would show as a gap in the density,
hardly in the velocity. The fluid takes nothing from a bridge that has not stepped yet.

The render takes a CUDA tensor into GL by a copy into a PBO queued on its own stream (`Tensor._update_with_pbo`,
every app). That copy still waits, up to about 4 ms at p95, while the GPU's queue is full; a ring of PBOs filled by
the uploader would move the wait to the uploader, not shrink it.

### Counted

Every drop is counted where it happens and shown on the panel beside the cadence (`CadenceSettings`): the
uploader's queue (`Upload drops`) and the board's inference frames (`Inference frame drops`), with the frames the sync hands on
and the frames the motion measures, each as an average and a minimum (1 / the worst gap). At start the uploader
drops about ten frames of each kind in the first 0.3 s, while the panel is built **(site fact)**; none after.

## The cadence

Measured in -sim on the P800 recording (20261111-123241 ([gone](FLOW.md#recordings-gone)), 992 × 800 at 50 fps, the player in view), per loop of the
recording, by `scripts/cadence_probe.py`; two loops each, the two runs back to back:

| Depth alignment            | Motion measures | Ticks   | Tick p95 | Worst tick | Capture latency p95 | Camera thread p95 |
|----------------------------|-----------------|---------|----------|------------|---------------------|-------------------|
| CUDA kernels (the default) | 731 of 731      | 77–80/s | 25–26 ms | 53 ms      | 18–19 ms            | 0.02 ms           |
| torch ops as a CUDA graph  | 731 of 731      | 66–72/s | 31–35 ms | 60–61 ms   | 19.6 ms             | 0.02 ms           |

The tick's p95 swings from run to run, 26 to 40 ms on the same code, and whole batches of runs drift together:
a change is judged by runs back to back.

The GPU's time per step, the GPU otherwise idle, on that recording's frame (`scripts/gpu_input_timing.py`):

| Step                                                              | GPU     | GPU ops |
|-------------------------------------------------------------------|---------|---------|
| color from the recording (BGR): copy, convert, straighten, mirror | 0.57 ms | 8       |
| color from the camera (NV12): the same                            | 0.85 ms | 23      |
| depth: copy and the two kernels                                   | 0.07 ms | 3       |
| a player's crop and the pose model's input                        | 0.11 ms | 6       |

Why nothing waits for the GPU: a thread that waits for it, behind the fluid's GL work and the pose model, hands
its frames on bunched, and the render loses those that land in one tick with the next. Live, P800 at 50 fps,
NEURAL_480, USB, 45 s each, the camera's thread converting on the GPU itself:

| Stream                             | Waiting on the GPU      | Not waiting               |
|------------------------------------|-------------------------|---------------------------|
| the camera's thread per frame      | mean 5.6, p95 16 ms     | mean 1.6, p95 2.6 ms      |
| captures arriving, p95 / max       | 34.6 / 92 ms            | 22.2 / 30.5 ms            |
| captures the motion measures       | 1816 of 2202 (82%)      | 2235 of 2249 (99.4%)      |
| the dye still for over 50 ms       | 197 times, up to 104 ms | 0, at most 45.6 ms        |
| render ticks                       | 80/s                    | 118/s                     |

With a player in view the pose model's GPU work costs frames where any thread waits behind it. In -sim (`blocky` (gone),
50 fps, the host detector on the uploaded image), the pose step on the camera's thread and the render measuring
only the newest capture per tick:

| Run                                | Captures the motion measures | Render ticks | Upload p95 |
|------------------------------------|------------------------------|--------------|------------|
| all on, `pose_wait` 10 ms          | 1699 of 1982 (86%)           | 82/s         | 13.5 ms    |
| the detector, no poses             | 97%                          | 107/s        | 11 ms      |
| without the poses' crops and model | 1989 of 1989 (100%)          | 114/s        | 0.17 ms    |
| all on, `pose_wait` 0              | 1769 of 1987 (89%)           | 82/s         | 13 ms      |

Checked without the app: the uploader's queue and its depth (`test_uploader`), the inference frame sync (`test_inference_frame_sync`),
the crops cut without `synchronize` (`test_crop_extractor`), the alignment's two paths on known geometry
(`test_depth_align`), the view's geometry and the depth's read (`test_view_window`), the board's captures
(`test_inference_frames`); in GL, by the scripts above and `scripts/frame_max_check.py`.
