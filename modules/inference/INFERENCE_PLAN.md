# The inference package's plan

What is not built in `modules/inference`, apart from the threads the app refactor moves
([apps/APPS_PLAN.md](../../apps/APPS_PLAN.md)). How the input works is in
[INPUT_THREADS.md](../flow/INPUT_THREADS.md). **(deduction)** marks what is inferred.

## A. The color in one kernel

- A1. The uploader's color conversion, straightening and mirroring as one CUDA kernel, as the depth's alignment is
  (`depth_align.py`): from the camera's NV12 23 GPU ops become 2, about 0.4 ms less GPU a frame **(deduction)**,
  measured per step by `scripts/gpu_input_timing.py`. After the live run of
  [the recording plan's checks](../recording/RECORDING_PLAN.md#1-checks-waiting-for-the-camera), which measures it on
  the camera.

## B. The previous crop

- B1. The optical flow on the crops (`enable_prev_crop`, off in deep_flow, so it runs nothing) does not wait for the
  crops' event, and its pairing of the current and the previous image is to be checked when it is used.

## The package held up to the rules (the apps' plan F, agreed 2026-11-17)

No code in the package is one app's: the face model is smile's alone, the segmentation hd_trio's, the optical flow,
`RecordedPoses` and `CaptureSync` deep_flow's, but each is general code with one user, which stays (as the camera's
backends do). The package has no rules file; nothing it keeps needs one beyond `CLAUDE.md` **(deduction)**.

## C. One thread per model, runners that only infer

The optical-flow runners are each a `Thread` with its own queue, input lock, callback list and dispatch, written
twice, once per backend: `RunnerONNX` and `RunnerTRT` share 155 of its ~390 lines **(seen)**. The face and person
detectors and the pose and segmentation predictors have the textbook shape: the predictor owns the one thread, the
queue, the drop count and the callbacks; each backend's runner holds `setup` and `infer` alone **(seen)**.

- C3. The optical-flow predictor takes that shape
- The output and the threads' count stay as they are; checked by the package's tests with a fake runner: the
  optical flow has no live user (`enable_prev_crop` off)

## D. The small breaks

- D1. `CropExtractor`'s callback list has no lock **(seen)**: it gets one, registered under it, copied under it,
  called outside it
- D2. The settings classes named `Settings` (`crop`, `detection`, `pose`, `segmentation`, `optical_flow`) take the
  glossary's `<ClassName>Settings`: `CropExtractorSettings`, `PersonDetectorSettings`, `PosePredictorSettings`,
  `SegmentationPredictorSettings`, `OpticalFlowPredictorSettings`; the apps' settings follow; the presets do not
  change (they hold field names, not class names)
- D3. `tools/benchmark.py` is a script in the package: it moves to `scripts/model_benchmark.py`, and its deadline
  reads `perf_counter`, not `monotonic` **(seen)**
- D4. The uploaders' docstrings name their apps (`BatchUploader`: "white_space, hd_trio"; `SerialUploader`: "in
  deep_flow") **(seen)**: they say what the uploader is, not who uses it
- D5. `model_types.Resolution`, `RESOLUTION_DIMS` and `ASPECT_RATIOS` have no user anywhere **(seen)**: they go

## The rules it touches

- Treat latency as a first-order concern; choose a path by its measured latency: A1, measured on the camera
- Fix a problem where it is made: B1, the optical flow waits for the event its crops carry
- `CLAUDE.md` Composition, prefer composition; Types, an ABC or shared code for a family: C
- `CLAUDE.md` Concurrency, a callback list has its own lock: D1
- `glossary.md`, a settings class is `<ClassName>Settings`: D2
- `CLAUDE.md` Packages, a script sits in a `scripts/` folder; Time, `perf_counter` for every stamp: D3
- `CLAUDE.md` Docstrings, a docstring says what the thing is: D4
- `CLAUDE.md` API evolution, remove obsolete pathways: D5
