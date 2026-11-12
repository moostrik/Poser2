# The input threads

A frame goes from the camera to the screen through several threads. Each thread does its own work and hands
on; a hand-off returns at once, so no thread waits for another's work, and a frame is dropped only where a
stage cannot keep up, counted, never because a part of it is late. Nothing but the camera runs on a camera's
thread, and no thread waits for the GPU while the GPU has room. deep_flow runs this way; the other apps follow
in [Open](#open).

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
`z.tolerance`, the mask) is the flow's [view stage](#the-view-stage).

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

### No bangs

With one camera the frame and tracklet `Sync` only throttle: they skip a bang within 0.75 of a frame of the last,
an uncounted drop. deep_flow runs the pose step on every uploaded image and `tracker.update` on every tracklet
message; `Sync` is for the apps with several cameras.

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

### The view stage

The flow's view of a capture, the stage before the motion (`ViewLayer`, settings `view`, in
`modules/render/layers/flow`; its geometry and the depth's read in `modules/flow/view`): the image and the
aligned depth cut to the view's fixed window at the motion's size; the depth read across a surface (interpolated
within the motion's `z.tolerance`, the nearest across an edge, a 0 never blended); the depth mask and the mask
([RESOLUTION.md](RESOLUTION.md#the-mask-and-the-depth-mask)); the foreground, the image where the mask is; and the
pose's points in the window, for the pose drawn over the view. Its outputs are the image, the foreground, the depth
as a `DepthView` ([MOTION.md](MOTION.md#when-it-measures)), the mask and the depth mask; the panel picks the
motion's picture and the density's among them (`inputs`). The color is resampled twice, straightened in the
uploader and cut to the view here, a little softer than one resample **(deduction)**, for keeping the camera's
work and the flow's apart. `scripts/view_check.py` checks it on a recording: the depth read matches its numpy
reference (`edge_aware_depth`) exactly, and the mask overlaps the depth mask 98.8%, with its edge on the image's.

The density's color is the foreground's look (`ForegroundLook`: the blend over time, the cel shade, the hue shift,
the sharpening) on the view's foreground (`foreground_look`), once per capture.

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

## Open

### 1. The other apps

white_space (four mono OAK-D cameras at 23 fps, the panoramic tracker, the light) and hd_trio (three color OAK-D
cameras, P720 square at 23 fps, RVM's masks, the 2D flow) still run their pose step on the camera's thread
through the frame bang; on this branch they stop at start with a message saying why, and run on `main`. Both run in `.venv` (depthai 2, the `oak` backend);
neither camera delivers depth. The refactor moves them to deep_flow's input, by the same rules:

- the camera's thread only hands over: to the uploader and the recorder;
- the uploader makes the corrected camera frame on its own thread and hands it on; nothing there waits for the GPU;
- the pose step runs off the camera's thread, its crops cut at the pose model's size and not waited for;
- every frame a stage measures over time is measured once, in order, and a frame is dropped only where a stage
  cannot keep up, counted and shown on the panel;
- the render takes GPU images in without waiting, and a stage runs once per frame, not once per tick.

#### Shared parts (`modules/`)

1. `Sync` takes the uploaded images (`submit_image(GpuImage)`, on the uploader's thread) as well as the camera's
   frames: a round is one uploaded image per camera, so the bang fires on the uploader's thread, after every
   camera's image of the round is on the GPU. The tracklet `Sync` is unchanged.
2. The uploader hands a frame as it came to CPU readers on its own thread (`add_frame_callback`), for a reader that
   needs the CPU image (white_space's light calibration); the camera's thread keeps only the hand-off.
3. `CropExtractor` cuts each pose's region at every size asked for, one region per pose and one antialiased resize
   per size: the pose model's input size for RTMPose, and a larger one for RVM (hd_trio's 768 × 1024).
4. `CaptureMax` moves from the 3D bridge to the flow's shared parts, and the 2D bridge takes `BridgeInput.same_tick`
   with it: its density is an amount (red) and its temperature ± (red), both kept by their size.
5. The probe (`scripts/cadence_probe.py`) takes the app as an argument, with each app's measures: the camera's
   thread per frame, the rounds per second, the pose step's rate per camera, the frames the motion measures
   (hd_trio), the render's tick, the drops.

#### white_space

1. Each camera hands its frames to the uploader (`Uploader(settings)`) and the video recorder only.
2. The uploader's images go to the -sim detector, the board (`set_camera_image`, the previews), and the frame
   `Sync`, whose bang runs `PosesFromTracklets.process` and `PoseInference.submit` on every camera's newest image
   (`PoseInference(crop, pose, uploader)`, its crops already at the model's size); the crops go to the board.
3. The light's calibration layer (`test_calibration.py`) reads the CPU frame from the uploader's frame callback
   instead of the camera's (`_store_video_frame`).
4. The device simulator (`Simulator`, the recordings played through an OAK) hands over the same way as the camera.
5. Measured live with four cameras and in -sim against `main`: the camera's thread per frame (deep_flow's went from
   5.9 to 0.02 ms at the mean), the rounds per second against the cameras' rate, the pose rate per camera, the
   uploader's drops; the light's own tick unchanged.

#### hd_trio

1. The cameras, the uploader and the frame `Sync` as white_space's (1, 2, 4 above).
2. The crops: the pose model's at its input size (an antialiased cut where today it gets 768 × 1024 shrunk 4× by
   plain bilinear, an aliased input), RVM's at 768 × 1024 from the same region (shared part 3).
3. Every RVM result is measured. A result (`segmentation.Predictor`'s callback) carries the poses, the masks and
   the foregrounds of one moment, cut from one uploaded image, so it is hd_trio's capture: complete when RVM hands
   it on, with nothing to wait for. The board keeps the results until the render takes them, at most a few, the
   oldest beyond that dropped and counted, as deep_flow's captures. The render runs each result, in order, through
   the center layers (placed by the result's own poses), the 2D motion on the centered mask and the 2D bridge
   stepping over the time since the result before; the fluids run every tick, each taking every camera's bridge,
   and two results of one camera in one tick give the larger amount per pixel (shared part 4).
4. The center layers' temporal blends (the mask's, the camera image's, the foreground's) then run once per result
   instead of once per tick: with their factors as they are, a blend takes the render's rate over the cameras' as
   long (about 2.6× at 60 ticks/s and 23 fps), so the factors are set again, on the look.
5. Measured in -sim and live against `main`: the RVM results the motion measures, the render's tick, the latency
   from the camera's stamp to the result on the board, the camera's thread.

#### Decisions

- hd_trio's capture is RVM's result, not `CaptureSync`'s image and depth: its flow reads only what RVM makes
  (the mask the motion measures, the foreground the bridge colors with), and the poses those were cut for.
- RVM's crop comes from `CropExtractor` at its own size, not from RVM resizing the pose model's crop: a 192 × 256
  crop enlarged to RVM's input would lose the mask's detail.
- hd_trio keeps its center layers, following the pose, fed per result. A view stage for the 2D flow beside
  deep_flow's (`ViewLayer`), placed by the pose and masked by RVM, is the next step after this, not part of it.
- One uploader thread for all of an app's cameras, which at 23 fps take 69 to 92 frames a second together against
  deep_flow's 50: the probe's uploader times say whether it keeps up, and one uploader per camera follows only if
  not.

#### Order and done

white_space first (the uploader's API, the rounds across cameras, no flow), then hd_trio (RVM's results, the 2D
flow per result). Each is done when it starts on this branch without the start-up error, its camera threads only
hand over (under 1 ms per frame at p95), every round runs the pose step and (hd_trio) every RVM result is measured,
the render's tick and the latency are no worse than on `main` by runs back to back, and its tests pass.

### 2. deep_flow live

The cadence with the camera and a player in view; the view's outputs on the monitors against the center layers'
on the same recording; the depth's edges by the resolution probe on stillslowfast (z's step inside the body, z's
noise at the edge and inside); the dye at drag 0.

### 3. The color in one kernel

The color's conversion, straightening and mirroring as one CUDA kernel: from the camera's NV12 23 GPU ops become
2, about 0.4 ms less GPU a frame **(deduction)**. After the live run, which measures it on the camera.

### 4. The previous crop

The optical flow on the crops (`enable_prev_crop`, off in deep_flow, so it runs nothing) does not wait for the
crops' event, and its pairing of the current and the previous image is to be checked when it is used.

### 5. No captures

While no capture comes (the camera stalls, -sim is paused) the bridge keeps its last output and the fluid keeps
taking it, where the motion holds its last velocity. What the flow should do then is not decided.

The tracklets recorded with the frames, for -sim, are planned in
[RECORDING.md](../camera/RECORDING.md#1-plan-b-the-recorded-tracklets).
