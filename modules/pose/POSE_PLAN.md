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

## The rules it touches

- `CLAUDE.md` Naming: use no synonyms for one concept across modules
- `CLAUDE.md` Settings, components that bind() must unbind() the same callbacks on teardown: B1
- `CLAUDE.md` Composition and wiring, clean up in an explicit stop or teardown, never in `__del__`: B1
- `CLAUDE.md` Composition, a value derived from pose data is the pipeline's, and so are its settings: the width, and
  how much the filter follows it
