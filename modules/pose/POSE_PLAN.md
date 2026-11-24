# The pose package's plan

What is not built in `modules/pose`. **(deduction)** marks what is inferred.

## A. Names

- A1. `nodes/_utils/array_moving_average.py` says "ring buffer" in place of "circular buffer" (its docstring
  and two comments). Why: the buffer is one, written at `_buffer_index = (_buffer_index + 1) % _window_size`, and
  "ring buffer" is the repo's one term for it (`LogRingBuffer`, `WindowNode`) **(seen)**. Check: the pose unit tests.

Questions:
- Each keypoint's uncertainty for the One Euro filter (noted 2026-11-18): RTMPose gives per keypoint a distribution
  over its SimCC bins, a narrow peak where the model is sure and a wide or double one where it is not; today only the
  peak's height reaches the pipeline, as the score (`modules/inference/pose/decode.py`). The distribution's width
  could set how strongly the One Euro filter smooths each point: harder where the model is unsure, lighter where it is
  sure. Cheap to compute next to the decode **(deduction)**; no paper tests it for human pose (the pose model's
  research, 2026-11-18, [INFERENCE.md](../inference/INFERENCE.md#the-models-and-their-files)). Built, it would be the width carried as a
  feature beside the score and the filter's cutoff scaled by it; whether it steadies the points more than the
  filter's own settings is to be measured
- `_AsyncSplit.submit` → `process`? (noted 2026-11-20) It stores the group and hands the frames to the async step's
  `AsyncInput.process` on the caller's thread (`pipeline/pose_pipeline.py`) **(seen)**, which the naming rule calls
  `process`; `AsyncInput.submit` became `process` in d277fa2e. It is internal: the stage's segment before the split
  calls it (`segment(functions, split.submit)`) **(seen)**
- Feedback on two changes made from the inference package (2026-11-20): `AsyncInput.process` (d277fa2e), and
  `has_points` moved beside `Frame` in `frame/frame.py` (fd65fde4), with `modules/recording` now importing
  `modules.pose` for its pose packing

## B. Teardown

- B1. The six nodes that unbind their settings only in `__del__` get an explicit close that their owner calls:
  `nodes/filters/ema_smoothers.py`, `euro_smoothers.py`, `moving_average_smoothers.py`, `predictors.py`,
  `rate_limiters.py` and `nodes/interpolators/base_interpolator.py`; `__del__` goes. Why: their settings' callback
  lists (`utils.Broadcast`, a plain list) hold the bound methods, so the object is never collected while its
  settings live and `__del__` never runs (seen for `ArtNetBars` in the inout scan; for these six a **(deduction)**
  from the same mechanism). Check: the pose unit tests.

## C. Smoothing

Questions:
- One rate for the angle, point and box smoothers (noted 2026-11-24)? The Ema smoother node
  (`nodes/filters/ema_smoothers.py`) hands one `EmaSmootherSettings` pair, attack 0.95 and release 0.8 by default, to
  every feature it smooths **(seen)**. The filters keep one meaning, rise against fall (a positive against a negative
  turn for an angle, on the shortest path), and are right as they are **(seen)**. For angle velocity, motion,
  symmetry and similarity, which are levels, rise and fall mean something; for joint angles, point and box positions
  they are only a direction, so a turn or a move one way follows faster than the other way, and with left and right
  angles mirrored (`angle_utils.py`, `_SIGN`) the left and right elbow would follow at different speeds
  **(deduction**, not measured**)**. Which rates suit a feature is the node's choice: should these features get one
  rate, attack equal to release? A box's width and height are levels, its x and y are not.

- C1. Interpolation smoothness with poses on every other frame: the user saw it less smooth than before in deep_flow
  -sim on `20261124-102304_Test`, and live not very smooth, the points sitting well **(site fact, 2026-11-24)**;
  never tested with the one-frame skip. Check live and in -sim.

The array filters in `nodes/_utils` stay in pose, so pose can stand on its own; utils keeps its own single-value
filters for the other packages (his word 2026-11-22).

## D. The recorder's name

- D1. The package's `Recorder` (HDF5 per stage, `recorder/recorder.py`) becomes `PoseRecorder`, and its
  `RecorderSettings` becomes `PoseRecorderSettings`; `recorder/recorder.py` becomes `recorder/pose_recorder.py`.
  `recorder/__init__.py`, the package's `__init__.py` and `tests/test_frame_io.py` follow **(seen)**. hd_trio's
  `main.py` drops its aliases `Recorder as PoseRecorder`, `RecorderSettings as PoseRecorderSettings` and
  `Recorder as VideoRecorder` for `modules/recording`'s `Recorder`, and white_space's `main.py` drops
  `Recorder as VideoRecorder` **(seen)**. hd_trio's `settings.py` drops its alias too; its subclass
  `_PoseRecorderSettings` then differs from its base by an underscore only and takes a name of its own
  **(seen)**. The log text already says `PoseRecorder` **(seen)**. Why: "Similar things share the noun and differ
  by a qualifier"; the user chose this on 2026-11-22. Tests: the pose tests, the recording tests, pyright.

## The rules it touches

- `CLAUDE.md` Naming: use no synonyms for one concept across modules
- `CLAUDE.md` Settings, components that bind() must unbind() the same callbacks on teardown: B1
- `CLAUDE.md` Composition and wiring, clean up in an explicit stop or teardown, never in `__del__`: B1
- `CLAUDE.md` Composition, a value derived from pose data is the pipeline's, and so are its settings: the width, and
  how much the filter follows it
- `CLAUDE.md` Naming, similar things share the noun and differ by a qualifier: D1
- `glossary.md`, a settings class is `<ClassName>Settings`, after the class it configures: D1
