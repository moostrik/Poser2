# The inference package's plan

What is not built in `modules/inference`, apart from the threads the app refactor moves
([REFACTOR_PLAN.md](../../REFACTOR_PLAN.md)). What is built is in [INFERENCE.md](INFERENCE.md), its times and how
the refactor's steps were checked in [TIMING.md](TIMING.md). **(deduction)** marks what is inferred.

## A. Each model's file named for its precision

- A3. The segmentation's files are named for their precision, as the pose's: RVM's ONNX files are FP16, all 159
  weight tensors but two of one value each, their inputs FP16 too, while their names say nothing **(seen: the
  weights' types, 2026-11-20)**. A name says `_fp16` when the file is FP16 and nothing when FP32, so
  `rvm_mobilenetv3_<size>.onnx` and `.trt` become `rvm_mobilenetv3_<size>_fp16`: in LFS, in the apps' model
  folders (the engines, not tracked), in `tools/models/export_rvm_to_onnx.py`, `scripts/makemodels.bat` and the
  segmentation's settings table (`segmentation/settings.py`). Checked by the inference tests and a low_flow `-sim`
  start, which loads the segmentation (deep_flow runs none **(seen)**)

## B. Timed waits that end on the Windows tick (from the module refactor thread, 2026-11-20)

On Windows a timed wait (`Condition.wait(t)`, `Event.wait(t)`) ends on the 15.6 ms system tick, up to 15.6 ms
late; a signalled wait wakes at once, and `time.sleep` is about 1 ms on Python 3.12 **(site fact, the module refactor
thread; measured there in `StreamReader`, a quarter of the frames ~30 ms apart at 50 fps)**. Two loops here pace on
a timed `Condition.wait` **(seen)**, and both are woken by every new item: the sync by each part it pairs, the
player by each frame it plays. `modules.utils.wait_until` sleeps the last `WAKE_STEP` (17.5 ms) before a deadline
blind, the lock released (`wait_until.py`) **(seen)**: a part that comes in that stretch waits for the sleep's end,
up to 17.5 ms, so as it stands it would add latency here, not remove it **(deduction, and the Screen Recorder
thread's measurement)**.

B waits for the module refactor's measurement of `timeBeginPeriod(1)` (its `REFACTOR_PLAN.md` B6, decided
2026-11-20): with every timed wait ending within about 1 ms, B2 and B3 would need only a plain timed wait, and B1
nothing.

- B1. `wait_until` takes a wake test, `woken: Callable[[], bool] | None`: in its last stretch it sleeps in 1 ms steps
  (`time.sleep(0.001)`, about 1 ms on Python 3.12), the lock released between, and returns True as soon as the test
  holds. Without a test it behaves exactly as now: its five tests stay, one for the wake test joins them in
  `modules/utils/tests`; its docstring and `modules/video/VIDEO.md`'s line on it say so (agreed with the Screen
  Recorder thread, which made it, 2026-11-20)
- B2. `InferenceFrameSync.run` waits for the next frame's deadline (`_remaining()`) with
  `wait_until(deadline, self._wake.wait, self._wake, woken=lambda: bool(self._inbox))`: a frame whose results do not
  all come leaves within about 1 ms of its deadline, not up to 15.6 ms after it, and a part still pairs within
  about 1 ms **(deduction)**
- B3. `RecordedPoses._send` waits for each recorded pose's moment (capture + its recorded latency) the same way,
  woken by a new frame that is due sooner: in `-sim` the poses come within about 1 ms of their moment, not up to
  15.6 ms late and uneven **(deduction)**
- Checked by deep_flow `-sim` on `20261112-125832_measure`, 60 s before and after: a timer of each overdue frame's
  lateness past its deadline (B2) and of each played pose's lateness past its moment (B3), both new, the inference
  frame sync's part wait (`Wait inference frame sync`), and `POSE_LATENCY_RAW`'s spread (p95 − p50); expected
  **(deduction)**: lateness and part wait under about 1 ms where they reached 15

## C. The pose model

For one player `STANDARD` does not fill the GPU, so `HIGH` costs as much in the app (TIMING.md, "The pose
models": 20 to 23 ms whichever model) **(seen)**; hd_trio (up to 3 players a set) and white_space (up to 8) batch
several crops, where it may not.

- C1. deep_flow and low_flow run the pose at `HIGH` (decided 2026-11-20): `pose.pose.resolution` from `STANDARD` to
  `HIGH` in every preset that has it, deep_flow's `settings.json`, `close.json`, `mattijs.json`, `slow.json` and
  `very nice.json`, and low_flow's `studio.json` **(seen: all `STANDARD` today)**. Checked by each app's `-sim` start:
  the log's "Pose: HIGH model ready", and no drops
- C2. hd_trio and white_space are measured at `STANDARD` and `HIGH`, FP16, before their resolution is chosen: alone,
  per batch size up to each app's engine's maximum (3 and 8), on their own engines (`scripts/model_benchmark.py
  --app <app> --models pose`, the resolution set); in the app, each app's `-sim` on a recording with several players,
  60 s a resolution: the pose's time, its drops and `POSE_LATENCY_RAW`. Into TIMING.md, "The pose models". They run
  on `.venv` (depthai 2); which of their recordings hold several players is found first, offline from the files

Questions:
- Each keypoint's uncertainty for the One Euro filter: in the pose package's plan ([POSE_PLAN.md](../pose/POSE_PLAN.md))

## The rules it touches

- `CLAUDE.md` Naming, name a concept after what it is: A3
- `CLAUDE.md` Time, pace a loop with `utils.wait_until`: B1, B2, B3
