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
| 1 | Refactor            | Pyright clean outside the tests, package by package (D2, D3)                             |                  |
| 1 | Screen Recorder     | One ffmpeg system (VIDEO_PLAN.md A)                                                      |                  |
| 1 | Inference           | RAFT out, the log names the model file, RVM files `_fp16` (INFERENCE_PLAN.md A)          |                  |
| 1 | Refactor            | The timing issue: the loops that only pace on `wait_until` (B1–B4)                       |                  |
| 2 | Refactor            | The timing issue: measure a 1 ms system tick (`timeBeginPeriod(1)`) against `wait_until` (B6) |             |
| 2 | Screen Recorder     | The timing issue: `wait_until` wakes at once on a new item                               | B6               |
| 2 | Screen Recorder     | The tick when play stops; the player's decode on CPU or iGPU; the tests' type errors     | 1                |
| 3 | Inference, Refactor | The timing issue: the loops a new item wakes (INFERENCE_PLAN.md B, B5)                   | 2                |
| 3 | Refactor            | Board held up to the rules, then one package at a time (A)                               |                  |
| 4 | Camera              | Depth confidence from the camera to the motion (CAMERA_PLAN.md A)                        | video A5         |
| – | Inference           | The pose model at HIGH in the presets; hd_trio's and white_space's resolution tested with several players | |
| 5 | GL Window           | The window manager: GLFW on the main thread, one kind of window, frame pacing (GL_PLAN.md)| the camera plan |
| – | Low Flow            | The water's speed of sound in the 2D (LOW_FLOW_PLAN.md G4); what the view's 2.9 ms is    |                  |
| – | You                 | The screen recorder's preview by eye; the checks at the rig (C)                          |                  |

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
- A6. `recording`, with its plan's split test (RECORDING_PLAN.md 3)
- A7. `session`
- A8. `settings`
- A9. `tracker`; `TrackerFrameSource.submit(cam_timestamps)` changed with inference C4
- A10. `utils`
- A11. `log_config.py`
- A12. The apps: `deep_flow`, `smile`, `white_space`

## B. Loops paced with `utils.wait_until`

A wait's timeout ends on Windows' 15.6 ms tick (CLAUDE.md, Time), so a loop paced by one comes up to 15.6 ms late
**(seen: 18 ms asked, 31.3 ms waited)**. `wait_until` sleeps its last ~17.5 ms, so a signal in that time wakes it up
to ~17 ms late **(seen)**: it fits a loop that only paces, not one a new item must wake at once. Built: the recording
player (`StreamReader`). In INFERENCE_PLAN.md B: `InferenceFrameSync`, `RecordedPoses`.

- B1. `ArtNetBars` (`modules/inout/artnet_bars.py`): its frame tick; woken only by a settings change.
- B2. The session's chunk timer (`modules/session/session.py`): woken only by stop.
- B3. white_space's motor clock (`light/motor.py`): one tick per `light_rate`; woken only by stop and the simulate
  setting. Checked at the rig with C1.
- B4. smile's player (`player/player.py`): its 20 ms poll between cell starts; woken only by stop.
- B5. The replay camera's tracklets (`modules/recording/replay/camera.py`) and smile's camera image rate
  (`cam/camera.py`): a new item notifies each loop, so `wait_until` as it is would hold it up to 17 ms
  **(deduction)**. They take the `woken` check the inference thread proposes for `wait_until` (1 ms steps in its
  last stretch, the lock released between), or whatever that becomes.

- B6. A 1 ms system tick, `timeBeginPeriod(1)` at the app's start, measured against `wait_until`: one deep_flow
  `-sim` run with and without it, the player's frame gaps (mean, p95), the poses' arrival against their moment and
  the process's CPU load. If it fixes the woken loops on its own, they need no `woken` check.

Checked by each package's tests; B5 also by one deep_flow `-sim` run, the tracklets' arrival against their moment.

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

## D. Pyright over the whole repository

Every file outside the tests type-checks clean, fixed in CLAUDE.md's order (annotation, narrowing, `cast`,
suppression last). Tests are not checked. Clean now: `modules/recording` (f9211f41), `modules/video`.

The count (D1, 20 Nov, **seen**): `.venv/Scripts/python.exe -m pyright --pythonpath .venv/Scripts/python.exe`,
24 s, 565 errors; the same with `.venv-oak4`, 20 s, 607 (the OAK-D backend fails there, depthai 3). 404 of the 565
are in tests, 27 in the untracked `temp/`, so about 134 in the code. `pyproject.toml` leaves out `apps/deep_flow`,
`apps/low_flow` and `modules/camera/oak4`.

- D2. Pyright runs in `.venv-oak4`, the venv deep_flow, low_flow and `oak4` run in: `[tool.pyright]` names it
  (`venvPath = "."`, `venv = ".venv-oak4"`), so Pylance and the command line check the same, and the command needs no
  `--pythonpath`. It leaves out every `tests/` folder and `temp/`, checks deep_flow, low_flow and `oak4`, and leaves
  out `modules/camera/oak`, the depthai 2 backend, which a second config at the root (`pyright.oak.json`, venv
  `.venv`) checks alone. Then the count again per package, both commands, and the time of a run on a handful of
  files. Checked by those runs.
- D3. Each package's errors fixed, one commit per package, checked by pyright and the package's tests: `scripts`
  (with `velocity_peak_check.py`, which imports the removed `modules.flow.bridge_3d.velocity_peak`), smile,
  white_space, deep_flow, low_flow, hd_trio, then the modules in alphabetical order. The threads that own a package
  are asked for feedback after its commit.
- The commit check and its CLAUDE.md line are the rules plan's; it gets D2's commit, the command per venv and the
  time of a run on a handful of files.

## Parked

- hd_trio's render writes `fluid.density.fade_time` every tick from the player's motion, and its show stages write
  `similarity_scale` (and the center geometry's `stage`): settings used to carry values while running. They reach
  the fluid and the color mask through their inputs once [the flow's line I](modules/flow/FLOW_PLAN.md#i-the-2d-flow-follows-the-3d)
  rebuilds the 2D fluid, instead of changing `Fluid2D` now.

## The rules it touches

- `CLAUDE.md` Project shape and Packages, what a package holds: A
- `CLAUDE.md` Time, pace a loop with `utils.wait_until`: B
- `CLAUDE.md` Types, fix a type error in order, suppress only as a last resort: D
- `CLAUDE.md` Settings, settings connect the user and a component, not two components: Parked
