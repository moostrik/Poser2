# Window plan

What is not built in `modules/window`, from the package scan of 2026-11-21. The steps are proposals; a step is one
commit, built on Matthias's go, checked by pyright and the window tests.

## B. The loop and the apps

The apps' render files are the Render thread's while it builds its plan; these steps touch them only once it is done.

- B1. The loop calls the update callbacks itself, with the frame's stamp, just before `renderer.update()`; the
  renderers' `_notify_update()` calls go, and smile's sequencer then runs after the callbacks instead of before. Why:
  four renderers call `_notify_update()` first thing in `update()` and smile after its sequencer, one call written five
  times; "A producer calls its callbacks itself" (seen: the five apps' `update`). Check: pyright, the window tests, a
  -sim of smile (the sequencer's order) and of deep_flow.
- B2. `RenderBase` forwards `input`, and deep_flow and smile use it instead of reaching into `window_manager`. Why:
  `RenderBase` forwards `run`, `stop`, `window_size` and `add_exit_callback` but not `input`, two ways to one thing
  (seen). Check: pyright.
- B3. The loop gives the frame's time as Cinder's `App` does, forwarded by `RenderBase`: `frame_timestamp` (the
  frame's stamp, perf_counter, s), `frame_interval` (s since the last frame's stamp, 0 at the first) and
  `frame_count` (frames since the loop began), each set before the update. deep_flow and low_flow read
  `frame_timestamp` from it instead of `window_manager`; deep_flow, low_flow and hd_trio drop their `_measure_tick`
  and `_last_tick` and hold `frame_interval` with flow's `hold_tick`, which stays flow's. Why: the time of a frame
  is the window loop's, as Cinder's `getElapsedSeconds` and `getElapsedFrames` are `App`'s; three apps measure the
  interval again from the stamps (seen: the three `_measure_tick`s; `WindowManager._update` already takes it for its
  timer). Check: pyright, the window tests, a deep_flow -sim (the held interval drives the flow; low_flow and hd_trio
  run the same lines).
- B4. `RenderBase` becomes `Render`, its file `render.py`: the class, `window`'s `__init__`, `window_manager.py`,
  `WINDOW.md`, the five app renderers and the name in `.claude/rules/app-lifecycle.md`, renamed in the same commit, the
  rules thread asked for feedback. Why: the glossary names an app's renderer `<App>Render`, and similar things share
  the noun and differ by a qualifier; "Base" names its place in the hierarchy, not what it is (seen: `render_base.py`,
  the five `<App>Render`s). Check: pyright, the guard and window tests.

## Rules this plan touches

- CLAUDE.md: A callback list is a `utils.Broadcast`
- CLAUDE.md: Inside a package, code that does the same is written once
- CLAUDE.md: Log exceptions with context; never fail silently
- CLAUDE.md: Use modern typing (`list[T]`, `dict[K, V]`, `X | Y`)
- CLAUDE.md: A producer calls its callbacks itself, from the thread that made the item. It has no thread for dispatching
- CLAUDE.md: Similar things share the noun and differ by a qualifier (`Frame`, `CameraFrame`)
- glossary.md: An app's renderer: `<App>Render`
- glossary.md: A time stamp: `timestamp`; not `time`, `time_stamp`
- CLAUDE.md: Downstream of the camera, take time from the frame or the tick instead of reading a clock
