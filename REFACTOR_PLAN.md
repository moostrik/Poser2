# The refactor's plan

What is next for the code as a whole, modules and apps: every package held up to the rules, one at a time, the
fixes found across packages, and the apps' checks at the rig. A package with a plan of its own keeps its work there:
[camera](modules/camera/CAMERA_PLAN.md), [flow](modules/flow/FLOW_PLAN.md),
[inference](modules/inference/INFERENCE_PLAN.md), [pose](modules/pose/POSE_PLAN.md),
[recording](modules/recording/RECORDING_PLAN.md), [video](modules/video/VIDEO_PLAN.md),
[low_flow](apps/low_flow/LOW_FLOW_PLAN.md). How the apps' input works is in
[INPUT_THREADS.md](modules/flow/INPUT_THREADS.md). **(deduction)** marks what is inferred.

## 2DO

What is to do now, across the threads, in order. Lines with the same number run side by side.

| # | Who                 | What                                                                                     | After            |
|---|---------------------|------------------------------------------------------------------------------------------|------------------|
| 1 | Refactor            | The timing issue: measure a 1 ms system tick (`timeBeginPeriod(1)`) (B1)               |                  |
| 2 | Refactor            | The timing issue: ArtNetBars' frame tick (B2)                                            | 1                |
| 3 | Inference, Refactor | The timing issue: the loops a new item wakes (INFERENCE_PLAN.md B, B's question)       | 1                |
| 3 | Refactor            | Board held up to the rules, then one package at a time (A)                               |                  |
| 4 | Camera              | Depth confidence from the camera to the motion (CAMERA_PLAN.md A)                        |                  |
| 5 | GL Window           | The window manager: GLFW on the main thread, one kind of window, frame pacing (GL_PLAN.md)| the camera plan |
| – | Inference           | hd_trio and white_space at full batch: whether the STANDARD pose model goes (INFERENCE_PLAN.md C) |  |
| – | Low Flow            | The water's speed of sound in the 2D (LOW_FLOW_PLAN.md G4); what the view's 2.9 ms is    |                  |
| – | You                 | The checks at the rig (C)                                                                |                  |

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
  times, `dict[int, float]`)
- A6. `recording`, with its plan's split test (RECORDING_PLAN.md 4); its plan's line 3 (the pose model's drops) goes with this plan's commit: it
  assumes a pose every frame, and deep_flow poses every other frame
- A7. `session`
- A8. `settings`
- A9. `tracker`; `TrackerFrameSource.submit(cam_timestamps)` changed with inference C4
- A10. `utils`
- A11. `log_config.py`
- A12. The apps: `deep_flow`, `smile`, `white_space`

## B. The timing issue: Windows' 15.6 ms tick

A wait's timeout ends on Windows' 15.6 ms tick (CLAUDE.md, Time), so a loop paced by one comes up to 15.6 ms late
**(seen: 18 ms asked, 31.3 ms waited)**. `wait_until` sleeps its last ~17.5 ms, so a signal in that time wakes it up
to ~17 ms late **(seen)**: it fits a loop that only paces, not one a new item must wake at once. Built: the recording
player (`StreamReader`).

- B1. A 1 ms system tick, `timeBeginPeriod(1)` at the app's start, measured against `wait_until`: one deep_flow
  `-sim` run with and without it, the player's frame gaps (mean, p95), the poses' arrival against their moment, the
  process's CPU load, and whether the tick holds with the window minimized (Windows 11 drops the request of a hidden
  window **(deduction)**).
- B2. `ArtNetBars` (`modules/inout/artnet_bars.py`): its frame tick waits `1/fps` after its work, so its rate drifts
  by the work as well as the tick, and `_update_event.clear()` after the wait drops a settings change that lands in
  between **(seen)**. It waits for a fixed-rate deadline (`next += 1/fps`) and clears the event before its work.
  Checked by its frame intervals logged before and after.

Not changed **(seen)**: the session's chunk timer (`chunk_start += chunk_length` keeps it from drifting; a split is
at most one tick late, every recorder together), white_space's motor sim (its falls are interpolated from
`perf_counter`, so a late wait does not move them), smile's start loop (a 20 ms poll for loaded clips, at start-up
only).

### Question

- After B1: which loops a new item wakes still need `wait_until`'s `woken` check (1 ms steps in its last stretch,
  the lock released between) — the replay camera's tracklets (`modules/recording/replay/camera.py`), smile's camera
  image rate (`cam/camera.py`), and INFERENCE_PLAN.md B's `InferenceFrameSync` and `RecordedPoses` — or none, if the
  1 ms tick fixes them; and whether B2 then waits with `wait_until` or a plain `Event.wait`.

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
- `CLAUDE.md` Time, pace a loop with `utils.wait_until`: B
- `CLAUDE.md` Settings, settings connect the user and a component, not two components: Parked
