# The input threads

A frame goes from the camera to the screen through several threads. Each thread does its own work and hands
on; a hand-off returns at once, so no thread waits for another's work, and a frame is dropped only where a
stage cannot keep up, counted, never because a part of it is late. Nothing but the camera runs on a camera's
thread. deep_flow is described first; the other apps follow it in [Open](#open).

"Code" is what the files show; **(deduction)** marks what is inferred from them.

## deep_flow's threads

How deep_flow runs now, per thread, the callbacks in the order they are registered (`apps/deep_flow/main.py`):

| Thread                                   | Runs                                                                                     | Waits for                                  |
|------------------------------------------|------------------------------------------------------------------------------------------|--------------------------------------------|
| the camera's (depthai; -sim: the player's reader) | `Recorder.submit_frame` (live), `Uploader.submit_frame`: each only queues             | nothing                                    |
| the uploader's (`Uploader`)              | a color frame: the pinned copy, the GPU's conversion and straightening queued, then `PersonDetector.submit` (-sim), `CaptureSync.submit_image`, `board.set_camera_image` and the pose step (`PosesFromTracklets.process` → the RAW stage → `PoseInference.submit`: the crops queued, a "crops ready" event); a depth frame: the pinned copy and the alignment queued as two CUDA kernels (`DepthAligner.align_staged`), then `CaptureSync.submit_depth`, which can hand a capture on to the board | the GPU, in the upload's copy from pinned memory while its queue is full |
| the tracklets' (depthai)                 | `Tracker.submit_cam_tracklets` (a queue put), `board.set_camera_tracklets`, `tracker.update` (sets the tracker's event) | nothing                                    |
| the tracker's                            | associates the tracklets, hands them to `PosesFromTracklets.set_tracklets` and the board   | nothing                                    |
| the pose model's (`pose.Predictor`)      | RTMPose on the crops after their event, then the CLEAN and SMOOTH stages, the capture sync, the board, OSC | its own model                              |
| the detector's (-sim)                    | YOLOv8n on the uploaded image (`PersonDetector`), boxes to the replay camera              | its own model                              |
| the capture sync's                       | hands on a capture whose wait for its poses ran out (`pose_wait`, 0 in deep_flow: none)   | the wait                                   |
| the render                               | each tick the captures since the last (`board.take_captures`), each through its image and depth into GL, the view stage, the foreground's look, the motion and the bridge; then the fluid | the GPU, where its copies into the PBOs wait while the queue is full |

The board keeps each camera's captures until the render takes them (`CaptureStoreMixin`, at most 4, the oldest
beyond that dropped and counted).

## The cadence

Every camera frame is to reach the motion at the camera's cadence, the motion measuring one capture per
frame. `Uploader.set_image` copies the frame into the camera's next pinned staging buffer (`STAGING_COUNT` 3)
and queues the copy to the GPU, the NV12 conversion and the straightening on the upload stream, with the
`ready` event the readers wait on; it does not wait for the GPU while the GPU has room. A copy from the frame's
own pageable memory, or an op that sizes its output on the GPU (`repeat_interleave`), makes the thread wait for
the GPU, behind the fluid's GL work: the sync then hands its captures on bunched, and the render loses those
that land in one tick with the next.

Live, P800 at 50 fps, NEURAL_480, USB, 45 s each, the moments of each capture, motion step and render tick:

| Stream                             | Waiting on the GPU      | Not waiting               |
|------------------------------------|-------------------------|---------------------------|
| the camera's thread in `set_image` | mean 5.6, p95 16 ms     | mean 1.6, p95 2.6 ms      |
| captures arriving, p95 / max       | 34.6 / 92 ms            | 22.2 / 30.5 ms            |
| captures the motion measures       | 1816 of 2202 (82%)      | 2235 of 2249 (99.4%)      |
| the dye still for over 50 ms       | 197 times, up to 104 ms | 0, at most 45.6 ms        |
| render ticks                       | 80/s                    | 118/s                     |

The panel pins the cadence beside the camera's video rates (`CadenceSettings` in deep_flow): the captures the
sync hands on and the frames the motion measures, each as an average and a minimum (1 / the worst gap), so a
frame lost after the camera shows where it is lost.

With a player in view the pose model's GPU work costs frames. In -sim (`blocky`, 50 fps, the host detector on
the uploaded image, FP16; 40 s each, cam 0):

| Run                                | Captures the motion measures | Render ticks | Upload p95 |
|------------------------------------|------------------------------|--------------|------------|
| all on, `pose_wait` 10 ms          | 1699 of 1982 (86%)           | 82/s         | 13.5 ms    |
| the detector, no poses             | 97%                          | 107/s        | 11 ms      |
| without the poses' crops and model | 1989 of 1989 (100%)          | 114/s        | 0.17 ms    |
| all on, `pose_wait` 0              | 1769 of 1987 (89%)           | 82/s         | 13 ms      |

Frames are lost in two places, both "newest wins": the capture sync passes over a complete capture when a newer
one completes while it waits for its poses (about 90 per run at `pose_wait` 10 ms, none at 0), and the render
measures only the newest capture per tick, so of two captures in one tick the first is lost (the rest,
uncounted). The camera's thread waits in the pose step (mean 2.3 ms, 59 times over 10 ms in 40 s) and the
render thread in its copy, both behind the pose model's work.

## Open

### 1. deep_flow: the input refactor

One refactor of deep_flow's input, agreed; no code yet. It fixes the lost frames and puts every piece of work on
the thread it belongs to: the camera only hands over, the uploader makes the corrected camera frame, the flow's
view stage makes the flow's view from it, the render takes them in and draws. The line between the uploader and
the flow: what every user of the camera needs (the color's conversion and straightening, the depth's alignment by
the camera's calibration) is the uploader's and reads camera settings only; what only the flow needs (the view's
window and size, the depth's upscale by the motion's `z.tolerance`, the mask) is the flow's.

**The camera** hands its frames to two places: the uploader (color and depth) and the recorder (the CPU frames as
delivered, for recording).

**The uploader** is the hand-off: its entry only queues (two frames of a kind; beyond that the oldest is dropped
and counted) and returns. Its own thread makes the corrected camera frame:

- the color: the pinned copy, the conversion and the straightening, in full, for the flow, the pose crops and the
  preview;
- the depth: the pinned copy, metres, and the alignment's scatter onto a grid as fine as the depth itself in the
  color frame (`aligned_depth_size`, the nearest kept, 0 where nothing lands), straightened and mirrored as the
  color. In torch, moved from the GL `DepthAlign` and the CPU's `depth_metres`; it replaces the shader only when it
  matches it pixel for pixel on recorded frames (border pixels at most, from fused multiply-adds) and is not
  noticeably slower. There is no full-frame depth picture any more (`cam_depth` goes).

When an image is on the GPU, the uploader tells four others, each of which only takes note and returns:

- the capture sync, which pairs the image and the depth into a capture and puts it on the board;
- the detector (-sim), which keeps the newest image for its own thread;
- the pose step: `PosesFromTracklets.process` makes a pose per player from the tracker's newest boxes, the RAW
  stage's two box filters smooth them, and `PoseInference.submit` cuts the crops from this very image
  (`CropExtractor`), queued on the GPU and not waited for, records a "crops ready" event and hands crops, poses and
  event to the pose model, whose thread has RTMPose wait for that event on its own stream. The `Uploader` knows
  nothing of poses: the crop is `PoseInference`'s, run on the uploader's thread;
- the camera-interval time sample.

None of it waits for the GPU: the crops' `synchronize` and `snapshot()` go.

**No bangs in deep_flow**: with one camera `Sync` only throttles (it skips a bang within 0.75 of a frame of the
last, an uncounted drop). The tracklets go to `tracker.update` directly; `Sync` stays for the apps with several
cameras.

**The capture sync**: `pose_wait` 0; every complete capture goes out in order, oldest first, under one lock. A
capture then carries the poses of the frame before, which with the view's window fixed only puts the pose points
drawn on it a frame behind.

**The flow's view stage**, new, a stage before the motion (GL): its components in `modules/flow`, its layer
(`ViewLayer`, with `ViewInput` and `ViewOutput`, the settings group `view`) beside the other stages' in
`modules/render/layers/flow`. It makes the flow's view from the capture, its textures pushed in its input like
the other stages':

- the view: the image and the aligned depth cut to the view's fixed window at the view's size (the motion's);
- the depth's upscale into the view with the edge-aware read (RESOLUTION.md's plan step 1): each view pixel reads
  the four depth pixels around it, interpolates them bilinearly when they lie within the motion's `z.tolerance`
  (m) of each other, and otherwise takes the nearest, so an edge never reads as a depth in between; a 0 (no
  depth) is never blended, and with one of the four 0 the nearest is taken. The `DepthView` keeps the real
  samples' spacing (`depth_step`, from 1 over the aligned grid's size);
- the depth mask: the view's depth cut by near and far, as it comes, which gates the motion's z;
- the mask: the depth mask after the guided filter that moves its edge to the image's (`guided_radius`,
  `edge_contrast`), which gates the velocity and the density's color. The mask's temporal blend and blur are
  hd_trio's and stay out;
- the foreground: the image where the mask is.

Its output ports are the image, the foreground, the depth as a `DepthView`, the mask and the depth mask. The app
wires them to the motion and the bridge; the panel keeps its choice of the picture the density's color comes from
and of the picture the motion measures on (`inputs.bridge_image`, `inputs.motion_image`), now among the view's
outputs instead of layers, and the preset changes with it. The view also places the pose's points in its window,
so the pose can still be drawn over the view. The center layers (`CenterGeometry`, `CenterMaskLayer`,
`CenterCamLayer`, `CenterDepthLayer`, `CenterForegroundLayer`, `CenterPoseLayer`), `CapturePoses`,
`DepthSourceLayer` and `DepthMaskLayer` leave deep_flow; what the screen shows reads the view's outputs. The color
is resampled twice, straightened in the uploader and cut to the view here, a little softer than one resample would
be **(deduction)**, for keeping the camera's work and the flow's apart.

deep_flow keeps the poses (the tracker, the pose model, the pose stages, the board, OSC, their drawing on the
camera image and over the view); what goes is what placed the view by the pose (`follow_pose`) and RVM as a mask
source, which is for cameras without depth; both can come back from hd_trio.

**The render** waits for no CUDA work: `Tensor._update_with_pbo` copies into its PBO with `cudaMemcpyAsync` on
the render's stream instead of the synchronous `cudaMemcpy`, for every layer and app that takes a CUDA tensor into
GL. The upload's own async copy from pinned memory still waits up to 12 ms while the GPU is busy, the only slow
call on its thread, so the driver may hold an async copy on the render thread the same way **(deduction)**; its
waits are timed before and after, and if the render still waits, the uploader writes into a ring of the render's
PBOs on its own thread and the render only takes them in. The GL client-memory audit leaves `Tensor._update_with_cpu` as it is and proposes a change only in
`particle_density.py`, so the two do not overlap.

**Every capture is measured.** The board's capture slice (`HasCaptures`) keeps each camera's captures since the
last tick (a few; beyond that the oldest is dropped and counted). The render runs, per capture and in order, the
view stage, the motion and the bridge, the bridge stepping on that capture's interval; a tick without a new
capture runs none of them, so no stage works on a frame twice. The fluid runs every tick on the tick's `dt`. With
two captures in a tick the bridge layer hands the fluid, for the density and the temperature, per pixel the
output of the capture with the larger amount (its alpha, with its own color), and for the velocity the newest:
a lost frame shows as a gap in the density, hardly in the velocity. Per capture the guided filter costs 0.05 ms
at 448 × 800 (RESOLUTION.md), the motion about 0.2 ms and the bridge's passes well under 1 ms (TIMING.md); the
newest capture is measured in the same tick as before, so nothing is shown later.

**One uploader**, no second way of uploading beside it: hd_trio, white_space and `oak4_test` stop working on this
branch until [Open 2](#2-the-other-apps); they run on `main`. Their tests that need the old upload are marked as
expected to fail with the reason, and the apps fail at start with a message that says why.

**Counted**: every drop is counted where it happens (the uploader's queue, the pose model's waiting poses, the
board's captures) and shown on the panel beside the cadence.

Built in this order, each part measured before the next; by part 4 the lost frames are fixed and measured, parts
5 to 7 are the structure:

1. the `Tensor` copy (every app);
2. the uploader's own thread and queue, the camera only handing over, the old apps switched off;
3. the pose step on the uploader's thread, the crops' event, no bangs;
4. the capture sync (`pose_wait` 0, in order);
5. the depth in torch at its own grid (pinned upload, metres, the alignment), checked in a harness against the
   shader and not wired into the app yet;
6. the view stage (the view, the edge-aware upscale, the masks, the foreground, the pose's points), the uploaded
   depth wired in, the center layers and what goes with them out, the panel's picks and the preset moved;
7. every capture through the view stage, the motion and the bridge, the density's max.

Tests, without the app: the uploader's queue (order, the oldest of a kind dropped and counted, stopping with
frames queued); the capture sync (every complete capture out, oldest first, none skipped, a mask wait ended by a
newer capture); the crops cut without `synchronize` and read right after the event (CUDA); the depth's alignment
against the geometry (`depth_projection`) on known points and against `DepthAlign`'s output on saved frames (a
harness, as the shader needs GL); the edge-aware read's three rules; the board's capture queue (keep, take in
order, drop and count beyond the bound); the view stage's mask against `CenterMaskLayer`'s on saved frames
without the blend and blur, the same with the depth read as nearest and differing only at edges with the
edge-aware read (harness); the density's max (harness); the guard tests.

Measured, after each part, by the probe kept as `scripts/cadence_probe.py`, in -sim on `blocky` with the player
in view: the camera's thread time (p95, max), the hand-over gaps, the captures and the motion's measures, the
render's tick (p95, max), the capture latency (stamp to board), the drops at each counter, and the GPU time per
stage (the uploader by CUDA events, the view stage, the motion and the bridge by `GpuTimer`); against this
document's numbers, and live with a player in view.

Looked at, in the app: the view stage's ports on the monitors against today's center layers on the same
recording; the depth's edges with the resolution probe on stillslowfast (z's step inside the body, z's noise at
the edge and inside); the dye at drag 0; the density's gaps, the probe's count of the dye still over 50 ms.

Done when the camera's thread does nothing but the hand-off (under 1 ms per frame at p95, unless Python's lock,
which a busy render thread holds for up to 5 ms, is what it waits for; the number is reported either way), the
hand-overs are as even as the camera's, every capture arrives and the motion measures every capture without
overload, and the capture latency and the render's tick p95 are no higher than before.

Not in it: the tracklets recorded with the frames (for -sim, planned apart); the pose model stays FP32, as it
loses precision in FP16; the previous crop (`enable_prev_crop`, off in deep_flow, so the optical flow on the crops
runs nothing), whose image pairing is to be checked when it is used.

#### Built

Parts 1 to 7 are built, one commit each (86d67e75, 38136432, 63baf89d, 6d0fa2da, ca3f2ffd, ca373c15, 77cd1f96),
then the depth's kernels (a4617bd1) and the crop at the pose model's size (0703194d).
Measured in -sim on the P800 recording (20261111-123241, 992 × 800, the player in view), per loop of the
recording, two loops each, by the probe (`scripts/cadence_probe.py`; the baseline and the parts by its first
version, which timed more and read the tick about 3 ms lower, unexplained):

| Part                     | Motion measures | Tick p95 (ms) | Capture latency p95 (ms) | Camera thread mean / p95 (ms) | Uploader drops |
|--------------------------|-----------------|---------------|--------------------------|-------------------------------|----------------|
| 4, the capture sync      | 86–87%          | 26.4–27.0     | 15.5                     | 0.08 / 0.10                   | 19             |
| 6, the view stage        | 81–83%          | 26.4–27.1     | 20.8–21.0                | 0.09 / 0.12                   | 27             |
| 7, every capture         | 100% (731/731)  | 25.7–27.1     | 20.0–20.9                | 0.09 / 0.13                   | 23             |
| 7, the probe in the repo | 100%            | 28.8–31.5     | 19.2–20.0                | 0.02 / 0.02, the hand-off     | 25–29          |

The tick's p95 swings from run to run (26 to 40 ms on the same code), and whole batches of runs drift together,
more than the parts differ: a change is judged by runs back to back. So, by the repo's probe, the depth's alignment:

| Depth alignment            | Ticks     | Tick p95 (ms) | Worst tick (ms) | Capture latency p95 (ms) | Uploader drops |
|----------------------------|-----------|---------------|-----------------|--------------------------|----------------|
| torch ops as a CUDA graph  | 66–72/s   | 31–35         | 60–61           | 19.6                     | 29             |
| CUDA kernels (the default) | 77–80/s   | 25–26         | 53              | 18.0–19.0                | 10             |

The uploader's drops all fall in the app's first 0.3 s, while the panel is built **(site fact)**: none in the
measured loops. The capture latency rose with part 6, where the depth goes through the uploader, by about 4 ms
with the kernels.

Where the build differs from the plan above:
- The PBO ring was not built: the copy queued on the render's stream (part 1) took the render's wait from 13 to
  about 4 ms at p95, and a ring would move the wait, not shrink it.
- The depth's alignment runs as two CUDA kernels compiled at runtime by the NVRTC torch ships (`cuda_kernels.py`),
  the GL shader's two stages: 0.07 ms of GPU a frame with the copy, against 0.13 for the shader. The torch ops,
  about a hundred small kernels at 0.6 to 1.3 ms, held the uploader 12 ms a frame launched one by one while the
  GPU's queue was full; they stay as the fallback (`depth_kernels` off, or the kernels failing), as one CUDA
  graph per lens, homography and size. The kernels match `DepthAlign` but for about one border pixel a frame.
- The view stage's parts: `ViewLayer` (`ViewInput`, `ViewOutput`, settings `view`) beside the other stages; its
  geometry and the edge-aware read (`DepthUpscale`, `edge_aware_depth`) in `modules/flow/view`; the pictures for
  the panels `ViewOutputLayer` and `ViewDepthLayer`; the pose over the view `CenterPoseLayer` on the view's points.
  `scripts/view_check.py`: the depth read matches its numpy reference exactly, the mask overlaps the depth mask
  98.8% with its edge on the image's.
- The foreground's look (blend, cel shade, hue shift, sharpen) is kept, as `ForegroundLook`, on the view's
  foreground (`foreground_look`, the density's color), so the color does not change; it now runs once per capture,
  not per tick, which changes its blend's time a little.
- The mask is the view's, with no pick (`inputs.mask` went); the main window's choice is `main_view`, as `view`
  is the stage's group; the show's `cam_depth` is `view_depth`.
- Two captures in one tick: `CaptureMax` in the 3D bridge, by the density's alpha and the temperature's size;
  `scripts/capture_max_check.py` checks it in GL. The z position the densities go in is the newest capture's.
- The optical flow on the crops (RAFT) does not wait for the crops' event; it runs nothing in deep_flow
  (`enable_prev_crop` off) and is to be checked with the previous crop.

Waiting for a live run: the cadence with the camera, the view's ports against the old center layers on the
monitors, the depth's edges by the resolution probe on stillslowfast, the dye at drag 0.

### 2. The other apps

The immediate follow-up, once deep_flow's refactor has shown it is the better system: hd_trio and white_space
(and `oak4_test`) move to it. They run their pose step on the camera's thread through the frame bang, as deep_flow
does now, and white_space's four cameras need the bang's rounds across cameras.

### 3. No captures

While no capture comes (the camera stalls, -sim is paused) the bridge keeps its last output and the fluid keeps
taking it, as now, where the motion holds its last velocity. What the flow should do then is not decided.
