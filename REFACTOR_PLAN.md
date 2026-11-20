# The refactor's plan

What is next for the code as a whole, modules and apps: every package held up to the rules, one at a time, the
fixes found across packages, and the apps' checks at the rig. A package with a plan of its own keeps its work there:
[camera](modules/camera/CAMERA_PLAN.md), [flow](modules/flow/FLOW_PLAN.md),
[inference](modules/inference/INFERENCE_PLAN.md), [pose](modules/pose/POSE_PLAN.md),
[recording](modules/recording/RECORDING_PLAN.md), [video](modules/video/VIDEO_PLAN.md),
[low_flow](apps/low_flow/LOW_FLOW_PLAN.md). How the apps' input works is in
[INPUT_THREADS.md](modules/flow/INPUT_THREADS.md). **(deduction)** marks what is inferred.

## 2DO

What is to do now, across the threads, in order.

| Who                 | What                                                                                     | After            |
|---------------------|------------------------------------------------------------------------------------------|------------------|
| Refactor            | `ArtNetBars` in numpy (B1)                                                               |                  |
| Refactor            | What holds the GIL on a hot path: py-spy of each app (B2)                                |                  |
| Camera              | Depth confidence from the camera to the motion (CAMERA_PLAN.md A)                        |                  |
| GL Window           | The window manager: GLFW on the main thread, one kind of window, frame pacing (GL_PLAN.md)| the camera plan |
| Inference           | hd_trio and white_space at full batch: whether the STANDARD pose model goes (INFERENCE_PLAN.md C) |  |
| Low Flow            | The water's speed of sound in the 2D (LOW_FLOW_PLAN.md G4); what the view's 2.9 ms is    |                  |
| You                 | The checks at the rig (C)                                                                |                  |
| Refactor            | Board held up to the rules, then one package at a time (A)                               |                  |

## Open questions

- The OAK-D's capture stamp, as the oak4's `capture_time`: needs an OAK-D to test (RECORDING_PLAN.md 2).
- The box predictor's `lead`, a fixed 1/30 s past the tracklet's stamp, against the box predicted to the image's
  own capture time.
- The panel's rates as frame-time analysis gives them (RTSS, CapFrameX), one pinned line per stream (Render,
  Video, Inference frames, Motion): `<name>  <average> fps  low <1% low>`, from one record of each stream's frame
  times. The 1% low replaces the minimums (`Video min`, `Inference frames min`, `Motion min`), in deep_flow and
  white_space alike.

## A. Each package held up to the rules

One package at a time, in alphabetical order, against `CLAUDE.md` and `.claude/rules`. Code that only one app uses
or decides comes first: it moves to that app, unless it is general code with one user (the camera backends stay).
Each package gets a report, and a plan for what does not meet the rules, built on your go. Its rules file, where it
has one, is checked line by line: still true of the code, about this package, a rule and not a description, said
once, not clashing with or repeating `CLAUDE.md`, checkable, without an unwanted side effect, and nothing missing.
What is wrong with a rule, and how it is known, goes to the rules thread.

Done: camera (14 Nov), render (its plan built and removed), hd_trio (G5). Held up in their own plans: flow
(FLOW_PLAN.md J), inference, video (VIDEO_PLAN.md A), gl's window manager (GL_PLAN.md).

- A1. `board`
- A2. `gl`, apart from the window manager
- A3. `inout`
- A4. `panel`
- A5. `pose`; its `pipeline` changed with inference C4 (`SourceFramesCallback` and `AsyncInput` carry the capture
  times, `dict[int, float]`). Its hot path, with B2's py-spy numbers: the steps run per player with a `Frame`
  replace each (white_space 37 a pose and 21 on the light tick), and every stage's `WindowTracker` writes ~18
  scalar features per player with a lock each, ~360 small writes a frame **(deduction)**. The fixes in view: one
  window buffer per stage holding all players, and fusing steps that run back to back (one smoother over several
  features). Batching all players through every step is left out: it goes against `pose-processing.md`'s
  per-track state.
- A6. `recording`, with its plan's split test (RECORDING_PLAN.md 4); its plan's line 3 (the pose model's drops) goes with this plan's commit: it
  assumes a pose every frame, and deep_flow poses every other frame
- A7. `session`
- A8. `settings`
- A9. `tracker`; `TrackerFrameSource.submit(cam_timestamps)` changed with inference C4
- A10. `utils`
- A11. `log_config.py`
- A12. The apps: `deep_flow`, `smile`, `white_space`

## B. The timing issue: Windows' 15.6 ms tick

A wait's timeout ends on Windows' 15.6 ms tick (CLAUDE.md, Time). Built: every loop on a deadline waits on a
`utils.Waker`, a high-resolution waitable timer and an event waited on together (the recording player, `ArtNetBars`,
`ArtNetSender`, `OscSender`, the replay camera's tracklets, smile's camera and clip cells, the Vimba camera's
control loop, `InferenceFrameSync`, `RecordedPoses`; white_space's light clock to 1 ms before its deadline, then a
spin, as LAYERS.md's *The light clock* measures). A 1 ms system tick
(`timeBeginPeriod(1)`) was measured against it and not taken: Windows 11 drops it while the window is minimized.
The high-resolution timer (`CREATE_WAITABLE_TIMER_HIGH_RESOLUTION`, Windows 10 1803 on) is not tied to the system
tick; `time.sleep` uses it since Python 3.11, and on Linux a timed `threading.Event.wait` is already this precise
**(deduction, from the documentation)**.

200 waits each in a script, and deep_flow `-sim` on `20261112-125832_measure` (pose model on, recorded tracklets,
30 s measured after 10 s warm-up) **(seen 2026-11-20)**:

| What (ms)                                              | No tick             | 1 ms tick           | Waker               |
|--------------------------------------------------------|---------------------|---------------------|---------------------|
| a 1 ms wait, waited, mean / p95                        | 15.6 / 16.2         | 1.6 / 2.1           | 1.5 / 1.6           |
| the same, the window minimized                         |                     |                     | 1.6 / 1.8           |
| an 18 ms wait, waited, mean / p95                      | 31.1 / 32.0         | 18.5 / 19.0         | 18.4 / 18.7         |
| the same, the window minimized                         |                     |                     | 18.4 / 18.7         |
| a wake to waking, mean / p95                           | 0.02 (notify)       | 0.02 (notify)       | 0.09 / 0.23         |
| `time.sleep(0.001)`, mean / p95                        | 1.6 / 1.7           | 1.5 / 1.7           |                     |
| the player's frame gaps, mean / p95 / max              | 20.0 / 22.2 / 38.6  | 20.0 / 22.1 / 38.4  | 20.0 / 22.2 / 38.2  |
| the same, the window minimized                         |                     |                     | 20.0 / 21.9 / 30.4  |
| the replay tracklets, late, mean / p95 / max           | 7.9 / 14.5 / 16.7   | 0.7 / 1.5 / 2.6     | 0.4 / 0.9 / 3.0     |
| the same, the window minimized, mean / p95             | 7.7 / 14.5          | 7.9 / 14.7          | 0.4 / 1.0           |
| `Event.wait(0.001)` in the app, minimized, mean / p95  | 15.6 / 16.5         | 15.7 / 16.8         | 15.6 / 16.6         |
| the poses (RAW) after their capture, mean / p95        | 41.2 / 49.1         | 39.8 / 44.6         | 39.4 / 43.9         |
| the process's CPU, cores                               | 15.75               | 15.63               | 15.64               |

The Waker's wait to a wake ran in the script; the run's sync deadline was not reached: no frame waited for its
poses. Beside a thread busy in pure Python, a wake comes up to one tick late, the Waker's and `time.sleep`'s alike
(14.6 ms for a 1 ms wait) **(seen)**: the woken thread asks for the GIL with a timed wait **(deduction)**.

Not changed **(seen)**: the session's chunk timer (`chunk_start += chunk_length` keeps it from drifting; a split is
at most one tick late, every recorder together), white_space's motor sim (its falls are interpolated from
`perf_counter`, so a late wait does not move them), smile's start loop (a 20 ms poll for loaded clips, at start-up
only).

- B1. `ArtNetBars._compute_frame` (`modules/inout/artnet_bars.py`) fills the bar's DMX buffer in numpy, in place of
  its Python loop over the pixels, four writes each. Byte-identical output: a test first pins today's buffer for a
  few settings (fill levels and their rounding edges, channel orders, colors and strengths, pixel counts), passing
  on the loop. A timing of `_compute_frame` before and after, in a script, is reported, not committed.

- B2. What holds the GIL on a hot path: one `py-spy record --gil` of each app over 30 s after 10 s warm-up (deep_flow,
  low_flow, hd_trio in `-sim`; white_space and smile as far as they run without the rig). Every function that holds
  the GIL more than 1% of the run becomes a step here (numpy, a shader, or off the hot path), each on Matthias's go. A
  thread busy in pure Python delays every other thread's wake by up to one 15.6 ms tick **(seen with the Waker and
  `time.sleep`, 2026-11-20)**. py-spy confirms or drops each of these. The pose pipeline's and the window trackers'
  numbers go to A5, which holds their fixes.

  Candidates from reading the code (deduction), most GIL time a second first, at 3 players:
  - The pose pipeline's steps (`modules/pose/nodes`), on the pose model's thread at the pose rate and on the output
    tick: per pose and step a `Frame` replace and numpy on 1–34 values, ~10–30 µs each, Python's overhead rather than
    the arithmetic. white_space runs 37 steps a pose and 21 on the light tick, the others 31–38: 50–150 ms a second.
  - The window trackers: every stage's `WindowTracker.process` writes ~18 scalar features per player, each with a
    lock, `nan_to_num` and two row writes: ~360 small writes a frame in white_space and hd_trio.
  - The OSC sound bundle (`modules/inout/osc_sound.py`, white_space's `OscSoundSender`), on its own thread at the
    output tick: `.tolist()` of every feature and python-osc packing each float on its own, a few hundred floats a
    tick: 5–20 ms a second.
  - The pose recorder's chunk (`modules/pose/recorder/frame_io.py`, `write_chunk`), on the recorder's thread at a
    split and at stop: a Python loop over every buffered frame, up to 50,000 (`_MAX_BUFFER_FRAMES`), for each track
    and each feature **(seen in the code; its GIL time: deduction)**.

  No Python loop over pixels, depth points or samples was found on a hot path: that work is numpy, cv2 and shaders.

## C. Checks at the rig

- C1. white_space, live and back to back with `main`, you at the rig: its camera threads only hand over (under 1 ms
  per frame at p95), every sync runs the pose step, the pose latency and the light's tick no worse than on `main`,
  its tests pass.
- C2. white_space's clip camera (`oak.ClipCamera`), which needs an OAK.
- C3. The playhead's smoothing (`playhead.tracking`, `speed_smoothing`) against the motor's phase: on
  `perf_counter` its error is 0.26° at p95 at 32 rpm against 4.3° on `monotonic` (simulated with the motor's own
  code, 2026-11-13), so the smoothing may be stronger than it needs.
- C4. hd_trio: a live recording with `mirrored` on (a live frame comes with `flip_h` on, which a -sim clip does
  not); the fluid's look.

## Parked

- hd_trio's render writes `fluid.density.fade_time` every tick from the player's motion, and its show stages write
  `similarity_scale` (and the center geometry's `stage`): settings used to carry values while running. They reach
  the fluid and the color mask through their inputs once [the flow's line I](modules/flow/FLOW_PLAN.md#i-the-2d-flow-follows-the-3d)
  rebuilds the 2D fluid, instead of changing `Fluid2D` now.

## The rules it touches

- `CLAUDE.md` Project shape and Packages, what a package holds: A
- `CLAUDE.md` Concurrency, work that numpy or the GPU does faster is not done in a Python loop; the loop holds the GIL
  and delays every other thread: B1, B2
- `CLAUDE.md` Settings, settings connect the user and a component, not two components: Parked
