# The refactor's plan

What is next for the code as a whole, modules and apps: every package held up to the rules, one at a time, the
fixes found across packages, the apps' checks at the rig among them. A package with a plan of its own keeps its work there:
[camera](modules/camera/CAMERA_PLAN.md), [flow](modules/flow/FLOW_PLAN.md),
[inference](modules/inference/INFERENCE_PLAN.md), [pose](modules/pose/POSE_PLAN.md),
[recording](modules/recording/RECORDING_PLAN.md), [video](modules/video/VIDEO_PLAN.md),
[low_flow](apps/low_flow/LOW_FLOW_PLAN.md). How the apps' input works is in
[INPUT_THREADS.md](modules/flow/INPUT_THREADS.md). **(deduction)** marks what is inferred.

## 2DO

What is to do now, across the threads, in order.

Nothing now; low_flow's G4 and the view's 2.9 ms wait in LOW_FLOW_PLAN.md for later.

## Open questions

- The OAK-D's capture stamp, as the oak4's `capture_time`: needs an OAK-D to test (RECORDING_PLAN.md 2).
- The box predictor's `lead`, a fixed 1/30 s past the tracklet's stamp, against the box predicted to the image's
  own capture time.
- The panel's rates as frame-time analysis gives them (RTSS, CapFrameX), one pinned line per stream (Render,
  Video, Inference frames, Motion): `<name>  <average> fps  low <1% low>`, from one record of each stream's frame
  times. The 1% low replaces the minimums (`Video min`, `Inference frames min`, `Motion min`), in deep_flow and
  white_space alike.

## A. Each package held up to the rules

One package at a time, in alphabetical order, each with the `package-scan` skill: `CLAUDE.md` held to the code
line by line (where the code belongs, leftovers, tests, pyright spelled out), its rules file both ways, its document
and plan. The scan changes nothing. A decided fix becomes a step in the package's plan (or this one), built on your
go; what is not decided is a question. Rule findings go to the rules thread as "Rules that go" and "New rules".

Done: inference (20 Nov), settings (21 Nov). Held up in their own plans: flow
(FLOW_PLAN.md J), gl's window manager (GL_PLAN.md, built 20 Nov).

- A1. `board`, scanned 21 Nov; its fixes:
  - the inference-frame slice and its test move from deep_flow and low_flow into modules/board, exported from
    `__init__.py` and imported by both apps' `RenderBoard`
  - `time_series.py`'s numpy import goes under `TYPE_CHECKING`; the annotations in `crop_images.py` and the moved
    slice are unquoted
  - the docstrings of `__init__.py`, `WindowStoreMixin` and `HasObservations` say only what each thing is; the
    reasons are cut
  - every slice stores what it is handed and returns it as stored, with no copies

  Its check: board's and both apps' tests, the guard tests, pyright.
- A2. `camera`
- A3. `gl`, apart from the window manager
- A4. `inout`. The OSC sound bundle (`osc_sound.py`, white_space's `OscSoundSender`), on its own thread at the
  output tick: `.tolist()` of every feature and python-osc packing each float on its own, a few hundred floats a
  tick, 5–20 ms of GIL a second **(deduction)**. Held against every app that uses it, and against app code that
  sends or receives the same way on its own, so there is one system.
- A5. `panel`
- A6. `pose`; its `pipeline` changed with inference C4 (`SourceFramesCallback` and `AsyncInput` carry the capture
  times, `dict[int, float]`). It starts with one `py-spy record --gil` of the pose pipeline (deep_flow and
  white_space `-sim`, 30 s after 10 s warm-up), so its fixes are chosen on measured numbers. Its hot path: the steps
  run per player with a `Frame` replace each (white_space 37 a pose and 21 on the light tick), and every stage's
  `WindowTracker` writes ~18 scalar features per player with a lock each, ~360 small writes a frame **(deduction)**. The fixes in view: one
  window buffer per stage holding all players, and fusing steps that run back to back (one smoother over several
  features). Batching all players through every step is left out: it goes against `pose-processing.md`'s
  per-track state. The pose recorder's chunk (`recorder/frame_io.py`, `write_chunk`) loops in Python over every
  buffered frame, up to 50,000, for each track and feature, at a split and at stop **(seen; its GIL time:
  deduction)**.
- A7. `recording`, with its plan's split test (RECORDING_PLAN.md 4); its plan's line 3 (the pose model's drops) goes with this plan's commit: it
  assumes a pose every frame, and deep_flow poses every other frame
- A8. `render`
- A9. `session`
- A10. `tracker`; `TrackerFrameSource.submit(cam_timestamps)` changed with inference C4
- A11. `utils`; `Broadcast` (utils/broadcast.py, since 94c65c84) is the one callback list: copied under its lock, called outside it, a
  failing callback logged; each package's own callback lists move to it in that package's review.
- A12. `video`
- A13. `log_config.py`
- A14. The apps: `deep_flow`, `smile`, `white_space`. white_space's checks at the rig, you there:
  - live and back to back with `main`: its camera threads only hand over (under 1 ms per frame at p95), every sync
    runs the pose step, the pose latency and the light's tick no worse than on `main`, its tests pass
  - its clip camera (`oak.ClipCamera`), which needs an OAK
  - the playhead's smoothing (`playhead.tracking`, `speed_smoothing`) against the motor's phase: on `perf_counter`
    its error is 0.26° at p95 at 32 rpm against 4.3° on `monotonic` (simulated with the motor's own code,
    2026-11-13), so the smoothing may be stronger than it needs
- A15. `hd_trio`, after its rebuild with the flow ([the flow's line I](modules/flow/FLOW_PLAN.md#i-the-2d-flow-follows-the-3d)), with its check at the rig: a live recording with `mirrored` on (a live frame
  comes with `flip_h` on, which a -sim clip does not); the fluid's look
  - its render writes `fluid.density.fade_time` every tick from the player's motion, and its show stages write
    `similarity_scale` (and the center geometry's `stage`): settings used to carry values while running. They become
    inputs of the fluid and the color mask in the rebuild, and hd_trio looks exactly as before

## The rules it touches

- `CLAUDE.md` Project shape and Packages, what a package holds: A
- `CLAUDE.md` Packages, code that does the same is written once: A1
- `.claude/rules/board.md`: A1
- `CLAUDE.md` Concurrency, work that numpy or the GPU does faster is not done in a Python loop; the loop holds the GIL
  and delays every other thread: A4, A6
- `CLAUDE.md` Settings, settings connect the user and a component, not two components: A15
