# The input threads

A frame goes from the camera to the screen through several threads. Each thread does its own work and hands
on; a hand-off returns at once, so no thread waits for another's work, and a frame is dropped only where a
stage cannot keep up, counted, never because a part of it is late. Nothing but the camera runs on a camera's
thread, and no thread waits for the GPU while the GPU has room. deep_flow runs this way; the other apps follow
([PLAN.md](PLAN.md#4-the-other-apps-on-deep_flows-input)).

"Code" is what the files show; **(deduction)** marks what is inferred from them, **(site fact)** what the
operator saw.

## deep_flow's threads

Per thread, the callbacks in the order they are registered (`apps/deep_flow/main.py`):

| Thread                                            | Runs                                                           | Waits for                                  |
|---------------------------------------------------|----------------------------------------------------------------|--------------------------------------------|
| the camera's (depthai; -sim: the player's reader) | queues each frame for the recorder (live) and the uploader     | nothing                                    |
| the uploader's (`Uploader`)                       | uploads each frame and hands it on ([below](#the-uploader))    | the GPU, in a copy while its queue is full |
| the tracklets' (depthai)                          | queues them for the tracker, the board; `tracker.update`       | nothing                                    |
| the tracker's                                     | associates the tracklets, to `PosesFromTracklets` and the board | nothing                                   |
| the pose model's (`pose.Predictor`)               | RTMPose on the crops, the CLEAN and SMOOTH stages, their users | its own model                              |
| the detector's (-sim)                             | YOLOv8n on the uploaded image, boxes to the replay camera      | its own model                              |
| the capture sync's                                | a capture whose wait for its poses ran out                     | `pose_wait`, 0 in deep_flow                |
| the render                                        | every capture since the last tick, then the fluid              | the GPU, in a copy while its queue is full |

### The uploader

The camera's one hand-off (`modules/inference/uploader.py`). `submit_frame` queues the frame and returns: two
frames of a kind per camera wait, and a third drops the oldest, counted (`dropped`). Its own thread makes the
corrected camera frame, which every user of the camera needs:

- the color: copied from a pinned staging buffer (`STAGING_COUNT` 3 per camera and kind, reused once the copy
  queued from it has run), converted (BGR or NV12) to RGB half, straightened by the tilt's homography and
  mirrored, handed on as a `GpuImage` with the `ready` event its readers wait on;
- the depth: copied the same way, turned into metres and moved onto the color's view at a grid as fine as the
  depth itself (`aligned_depth_size`), the nearest kept where two land on one cell and 0 where nothing lands,
  straightened and mirrored as the color, handed on as a `GpuDepth` (`modules/inference/depth_align.py`).

The depth's alignment runs as two CUDA kernels compiled at runtime by the NVRTC torch ships
(`modules/inference/cuda_kernels.py`, through ctypes, no extra package), the GL `DepthAlign` shader's two stages.
`camera.uploader.depth_kernels` off, or kernels that cannot be compiled or loaded (logged once), run the same in
torch ops captured as one CUDA graph per lens, homography and size. The torch ops are about a hundred small
kernels; launched one by one they held the uploader 12 ms a frame while the GPU's queue was full, as each launch
waits there. `scripts/depth_align_check.py` holds both to the shader on recorded frames.

| Depth alignment, P800 recording | GPU a frame         | Pixels off the GL shader by over 1 mm |
|---------------------------------|---------------------|---------------------------------------|
| CUDA kernels (the default)      | 0.07 ms, with copy  | about 1 a frame, at borders           |
| GL `DepthAlign`                 | 0.13 ms             | the reference                         |
| torch ops                       | 0.6 to 1.3 ms       | under 2 a frame, at borders           |

The line between the uploader and the flow: what every user of the camera needs (the color's conversion and
straightening, the depth's alignment by the camera's calibration) is the uploader's and reads camera settings
only; what only the flow needs (the view's window and size, the depth's read into it by the motion's
`z.tolerance`, the mask) is the flow's view stage ([VIEW.md](VIEW.md)).

When an image is on the GPU, the uploader hands it to four others, each of which only takes note and returns:

- the capture sync, which pairs the image and the depth into a capture;
- the detector (-sim), which keeps the newest image for its own thread;
- the pose step: `PosesFromTracklets.process` makes a pose per player from the tracker's newest boxes, the RAW
  stage's two box filters smooth them, and `PoseInference.submit` cuts the crops from this very image
  (`CropExtractor`), at the pose model's input size, queued on the GPU and not waited for; it records a "crops
  ready" event and hands crops, poses and event to the pose model, whose thread has RTMPose wait for that event on
  its own stream. The `Uploader` knows nothing of poses;
- the camera-interval time sample.

The pose model stays FP32: it loses precision in FP16.

### No frame trigger

With one camera the frame and tracklet `FrameTrigger` only throttle: they skip a trigger within 0.75 of a frame of the last,
an uncounted drop. deep_flow runs the pose step on every uploaded image and `tracker.update` on every tracklet
message; `FrameTrigger` is for the apps with several cameras.

### The capture sync

`CaptureSync` (`modules/inference/capture_sync.py`) matches an image, its depth and the poses of one moment by
capture time. With `pose_wait` 0 a capture never waits for its poses: it carries the poses of the frame before,
which with the view's window fixed only puts the pose's points drawn over the view a frame behind. Every complete
capture goes out, in the order of capture, under one lock, to the board.

### The render: every capture

The board keeps each camera's captures until the render takes them (`CaptureStoreMixin.take_captures`, at most
4; beyond that the oldest is dropped and counted). Each tick the render runs every capture since the last, in
order (`_run_capture` in `apps/deep_flow/render.py`): its image and depth into GL, the view stage, the foreground's
look, the motion, and the bridge stepping over the time since the capture before. A tick without a capture runs
none of them, so no stage works on a frame twice; the fluid runs every tick on the tick's `dt`.

With two captures in one tick the 3D bridge hands the fluid, per pixel, the density and the temperature of the
capture with the larger amount (`CaptureMax`: a density's alpha, a temperature's size, with its own color or
warmth), and the newest capture's velocity and z position: a lost frame would show as a gap in the density,
hardly in the velocity. The fluid takes nothing from a bridge that has not stepped yet.

The render takes a CUDA tensor into GL by a copy into a PBO queued on its own stream (`Tensor._update_with_pbo`,
every app). That copy still waits, up to about 4 ms at p95, while the GPU's queue is full; a ring of PBOs filled by
the uploader would move the wait to the uploader, not shrink it.

### Counted

Every drop is counted where it happens and shown on the panel beside the cadence (`CadenceSettings`): the
uploader's queue (`Upload drops`) and the board's captures (`Capture drops`), with the captures the sync hands on
and the frames the motion measures, each as an average and a minimum (1 / the worst gap). At start the uploader
drops about ten frames of each kind in the first 0.3 s, while the panel is built **(site fact)**; none after.

## The cadence

Measured in -sim on the P800 recording (20261111-123241, 992 × 800 at 50 fps, the player in view), per loop of the
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

With a player in view the pose model's GPU work costs frames where any thread waits behind it. In -sim (`blocky`,
50 fps, the host detector on the uploaded image), the pose step on the camera's thread and the render measuring
only the newest capture per tick:

| Run                                | Captures the motion measures | Render ticks | Upload p95 |
|------------------------------------|------------------------------|--------------|------------|
| all on, `pose_wait` 10 ms          | 1699 of 1982 (86%)           | 82/s         | 13.5 ms    |
| the detector, no poses             | 97%                          | 107/s        | 11 ms      |
| without the poses' crops and model | 1989 of 1989 (100%)          | 114/s        | 0.17 ms    |
| all on, `pose_wait` 0              | 1769 of 1987 (89%)           | 82/s         | 13 ms      |

Checked without the app: the uploader's queue and its depth (`test_uploader`), the capture sync (`test_capture_sync`),
the crops cut without `synchronize` (`test_crop_extractor`), the alignment's two paths on known geometry
(`test_depth_align`), the view's geometry and the depth's read (`test_view_window`), the board's captures
(`test_captures`); in GL, by the scripts above and `scripts/capture_max_check.py`.
