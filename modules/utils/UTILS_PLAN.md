# Utils plan

Proposed by the utils scan of 2026-11-22 (deep-flow at 3c7a387d), go 2026-11-24; what is here is not built yet. Each step names
its finding, **(seen)** in the code or a run, **(deduction)** when inferred. A step checked by a test or pyright is one
commit. Steps marked *owner* change another package: a minor fix, with that package's thread asked for feedback.

## A. Broadcast

A5. Broadcast's docstrings say what it is. The reasons (copied under its lock and called outside it; an attribute,
not a base class; `__bool__` lets a producer skip building the item) and its one spelling, `Broadcast[[X]]`, the
only form that holds a callback without arguments, move to a Broadcast section in UTILS.md.
Why: CLAUDE.md Docstrings; both docstrings give reasons **(seen)**.

A6. `tests/test_broadcast.py`: calls in registration order, a second add keeps the first place, a failing callback is
logged and the next one runs, remove, clear, `bool`.
Why: every callback list is a Broadcast and it has no test of its own **(seen)**; five other packages' tests lean on
its logger name.

A7. smile's saved callback passes one `SavedVisitor` dataclass instead of six positional values (id, style, kept,
grin path, portrait path, share path). *Owner: smile.*
Why: three of the six are paths of one type, so a swapped pair passes pyright **(seen)**; CLAUDE.md Settings "`@dataclass`
for plain value objects". clip_cells' image callback (cell, frame, first) and the window's mouse button (window, x, y,
button, pressed) stay: their values differ in type, or are an x, y pair. smile's `save/` has no saved or result
type to reuse: `clip_saver.py` holds only `ClipSaverSettings`, `GrinRate` and `ClipSaver`, and `record.py`'s record is a
JSON dict written into the files **(seen)**. Changes with it: the `SavedCallback` alias and the call in
`clip_saver.py`; the one consumer, `online.py`'s `on_saved` (registered in smile's `main.py`), whose `share` default
of `''` goes, since the alias has none; the two registrations in `save/tests/test_clip_saver.py` **(seen**, PC grep at
b0b50437**)**. Check: smile's save tests.

A8. A producer's `Broadcast` is private, and its typed boundary is `add_<noun>_callback(callback: <Noun>Callback)`
(and `remove_...` where a consumer leaves before the producer stops), with the alias named where the item is made
(`CameraFrameCallback`, `TrackletCallback`). The public lists, all of which but one already have the typed add
method **(seen**, PC grep at b0b50437**)**:
- the OAK-D and OAK4 cameras' `frame_callbacks` and `tracklet_callbacks`, the Vimba and webcam cameras'
  `frame_callbacks`, and the replay camera's two (`recording/replay/camera.py`) become `_frame_callbacks` and
  `_tracklet_callbacks`; the Vimba test that clears the list directly (`vimba/tests/test_camera.py`) clears the private
  one;
- the recording player's `loop_callbacks` and `jump_callbacks` become private;
- `FrameHandOver.frame_callbacks` becomes private with an `add_frame_callback`; the player's `add_frame_callback` and
  `test_frame_hand_over.py` add through it, and the player's `frame_callbacks` property, which hands out the hand
  over's list, goes; its callers, not grepped yet, move to `add_frame_callback` (grepped at the build).
Every other producer's list is private already and has its add method; `pose_pipeline._Stage`'s two lists are fields
of a private class that `PosePipeline`'s add methods fill, so they stay. *Owner: camera, recording.*
Why: the in and outputs are read at the boundary: a typed method with a named alias says what goes in and out without
reading the `Broadcast` declaration, and keeps the list itself as the producer's own detail; asked 2026-11-22.
Check: pyright, each touched package's tests.

## B. Geometry

B1. `points_and_rects.py` and `fit.py` become a `geometry/` folder: `point.py` (`Point2f`), `rect.py` (`Rect`),
`quad.py` (`Quad`) and `fit.py` (`fit`, `fill`). `Rect.aspect_fit` and `Rect.aspect_fill` compute through `fit()` and
`fill()`, and `fit`/`fill` share one body. Imports run one way: `rect.py` uses point, quad and fit. `tests/test_fit.py`
becomes `tests/test_geometry.py`, with Rect cases added. Callers import from `__init__.py` and do not change.
Why: the same centering arithmetic is written four times, twice in `fit.py` and twice in `points_and_rects.py`
**(seen)**; CLAUDE.md Packages "code that does the same is written once", and "Its files each hold one class or one
subject, named after it": `points_and_rects.py` holds three classes in 525 lines; four files make a folder (chosen
2026-11-22). Check: test_geometry.

## C. Color

C1. `color.py`: "colour" becomes "color"; the module docstring says what `Color` is (an RGBA color in 0..1), not
"for the settings system", since gl, render, flow, panel and the apps use it too; the hand-written `__eq__` goes, the
dataclass makes the same one.
Why: glossary (American spelling), Docstrings, written once **(seen)**. Check: settings tests (value types,
serialization).

## D. Smoothing

D1. `smoothing.py` becomes four small filters that are used the same way: `f(value, timestamp)` returns the smoothed
value, and each has `value` and `reset(value)`. The timestamp is required and `freq` goes from every filter: it is
only the time step's fallback for a call without a timestamp, and both callers pass the tick's time **(seen**,
`smoothing.py:143-147`, `:322-326`; CLAUDE.md Time**)**; a filter's first sample is its starting value. `OneEuroFilter`'s `target` and `velocity` go with their one reader,
`OneEuroFilterAngular`.
- `EmaFilter`: set by a time constant in seconds, settable between calls, corrected for the time between samples
  (`1 − exp(−dt / τ)`), where a time constant of 0 passes the value through; an
  optional second time constant for falling values gives attack and release, so `EmaFilterAttackRelease` goes into
  it.
- `OneEuroFilter`: our own One Euro filter (Casiez et al. 2012), set by `min_cutoff`, `beta` and `d_cutoff`; the
  third-party `OneEuroFilter` package goes from `pyproject.toml` and `timing.py`'s version list, and with it the
  subclass that read the package's private state through a name that resolves only because both classes share it.
- `SpringFilter`: as it is but for `freq`, the one filter that overshoots.
- `AngleFilter(filter)`: one wrapper for any of the three, which moves each target to the angle nearest the filter's
  value (the shortest path), runs the filter and wraps its value to ±π. Rise and fall then mean a positive and a
  negative turn, as they mean up and down for a number, so the filters stay agnostic of their input and keep one
  meaning; the sin and cos form would break it, since sin and cos are not the angle and one rises while the other
  falls on the same turn **(deduction)**. Pose's `AngleEmaSmooth` uses the same shortest path **(seen)**. Following a
  speeding-up motion faster is the One Euro filter's job, not a second meaning of rise and fall. It replaces `OneEuroFilterAngular`, `SpringFilterAngular`,
  `EmaFilterAngular` and `EmaFilterAttackReleaseAngular`, and the 15 copies of the angle wrap in them.
The two callers move with it: white_space's `light/playhead.py` (the rpm's `EmaFilter`, whose alpha from the
`speed_smoothing` setting is a rate per second, `1 - (1 - alpha)^dt` **(seen**, `smoothing.py:331`**)**, and converts at
the call, each tick as now, to the time constant −1 / ln(1 − alpha), so it smooths as before) and
`light/layers/projection/test_pose_waves.py` (a `OneEuroFilter` on the player count). *Owner: white_space.*
Why: 8 classes serve 2 users **(seen**, PC grep at b0b50437**)**, built as a subclass per filter and one more per angle
form, with the wrap written 15 times, a mangled `_EmaFilter__alpha` and a third-party base read by its private names
**(seen)**; smoothing is the single-value tool for every package but pose (chosen 2026-11-22), and pose keeps its own
array filters so it can stand on its own (his word 2026-11-22); the wrapper is composition. Nobody outside pose smooths an angle today **(seen)**; the wrapper keeps it
at the cost of one class.
Check: a small `tests/test_smoothing.py` (Ema step reaches 63% after one time constant at any sample rate; rise
against fall; a time constant of 0 passes the value through; the rpm case, the new `EmaFilter` with the converted
time constant against the old `1 − (1 − alpha)^dt` on the same samples; One Euro holds a constant input and follows a
ramp; an angle filtered across ±π takes the short way; an angle Ema with a fall time follows a negative turn at it); white_space's tests.

D2. `interpolation.py` goes, with its four exports from `__init__.py`.
Why: its Hermite interpolators (scalar, vector, scalar angle, vector angle) predict a sampled signal between its
samples, and pose's `nodes/_utils/array_predict.py` (`Predict`, `AnglePredict`, `PointPredict`, used by
`predict_interpolators.py`) does that job for every pose feature **(seen)**; CLAUDE.md API evolution "Remove obsolete
pathways". No commit since the history starts (2026-10-29) uses them outside utils **(seen**, `git log -G`**)**. The
methods differ, a Hermite spline with smoothed velocity against a least-squares fit, so this removes a second way,
not a copy. Check: pyright, utils tests.

## E. Document

E1. UTILS.md's Waker section drops its list of the loops that wait on one, and keeps the measurements, the light
clock's spin and the plain-wait timers.
Why: the list is stale: it misses the webcam camera, the video stream reader and the window manager, and names "the
recording player", which imports no Waker of its own **(seen**, PC grep**)**; CLAUDE.md Time already states the rule.

## F. What utils holds

utils holds small tools with no domain that many packages share, and imports nothing else of the project **(seen**:
its files import only each other, the standard library, numpy and two third-party packages, `watchdog` and
`OneEuroFilter`**)**; after F1 and D1 it imports no third-party package but numpy.

F1. `hot_reload_methods.py` goes, with `HotReloadMethods`, `MethodType` and `MethodInfo` from `__init__.py`.
Why: nothing uses it **(seen**, PC grep**)**, and through `__init__.py` every import of utils loads the third-party
`watchdog` for it **(seen)**; removed as asked 2026-11-22. `watchdog` stays in `pyproject.toml`: gl's shader file
watcher and smile's folder scanner use it **(seen**, PC grep at b0b50437**)**. Check: pyright, utils tests.

F2. `modules/logging` moves into utils as `utils/logging/` (`crash_log.py`, `ring_buffer.py`, `setup.py`, its tests);
`crash_log.ROOT` becomes `parents[3]`, one level deeper; every caller imports it from utils in the same change:
`launcher.py`, the four probe scripts in `scripts/`, and the panel's `server.py`, `process.py`, `page.py`,
`log_drawer.py` and `tests/test_link.py`; `test_ring_buffer.py` imports `ShortNameFormatter` relatively. SMILE.md's two
lines that name `modules/logging` say utils. The root guard tests name no package and do not change **(seen**, PC grep
at b0b50437; no app's main imports it**)**.
Why: it became its own package in 8ce648e7, but log setup, the panel's log buffer and the crash log are shared tools
with no domain, which is what utils is for, and `modules/` keeps one small package less, as asked 2026-11-22.
Check: the logging tests, pyright, one app start that writes its log file.

F3. The PC's `modules/oak` and `modules/tests` folders go: they hold only `__pycache__` folders, no code and nothing
in git **(seen)**. No commit.

F4. `waker.py`, `fps.py`, `performance_timer.py`, `timing.py` and `time_series.py` move into a `time/` folder in
utils; the rest stay at the root beside `geometry/` and `logging/`. Callers import from `__init__.py` and do not change.
Why: the root holds 11 loose files after B1, D2 and F1, five of them about time **(seen)**; CLAUDE.md Packages "A
package with many loose files groups them in folders by job"; in utils, not a `modules/time` package, since these are
shared tools with no domain (chosen 2026-11-24). `smoothing.py` and `lfo.py` alone would make a folder of two, which
the rule prefers not. Check: pyright, utils tests.

## Rules this plan touches

- Concurrency: A callback list is a `utils.Broadcast`
- Naming: Use no synonyms for one concept across modules
- Error handling: Log exceptions with context; never fail silently
- API evolution: Remove obsolete pathways instead of leaving parallel legacy ones
- Docstrings: A docstring says what the thing is; reasons sit in the package's document
- Packages: Inside a package, code that does the same is written once
- Packages: Its files each hold one class or one subject, named after it
- Settings: Use `@dataclass` for plain value objects
- Glossary: American spelling
- Testing: unit-test infrastructure
- Composition and wiring: Prefer composition over inheritance (D1)
- Time: Take time from the frame or the tick downstream instead of reading a clock (D1)
- Imports: Import from a package's `__init__.py`, never from its internal files (F2)
