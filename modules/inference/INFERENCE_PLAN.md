# The inference package's plan

What is not built in `modules/inference`, apart from the threads the app refactor moves
([REFACTOR_PLAN.md](../../REFACTOR_PLAN.md)). What is built is in [INFERENCE.md](INFERENCE.md), its times and how
the refactor's steps were checked in [TIMING.md](TIMING.md). **(deduction)** marks what is inferred. From the module
refactor's package scan of 2026-11-20, its questions decided the same day.

## C. Names, places and threads

- C1. `CropExtractor.submit` → `process`: it cuts and calls its callbacks on the caller's thread, which the naming
  rule calls `process` **(seen)**. It is the pose pipeline's `AsyncInput` (`modules/pose/pipeline/spec.py`), called
  by `PosePipeline` (`pose_pipeline.py`) **(seen)**, so `AsyncInput.submit` becomes `process` there too, its one fake
  in `modules/pose/tests/test_pipeline.py` and `test_crop_step.py` following **(seen)**, and INPUT_THREADS.md's mention; the pose thread is asked for
  feedback. Checked by pyright and the inference, pose and apps' tests
- C2. `has_points` moves to modules/pose, beside the `Frame` it tests (`pose/pose_estimator.py` today, called by the
  pose estimator and four apps' `main.py`) **(seen)**; `pose_arrays` calls it instead of its own copy of the test
  **(seen)**. `frame_poses` and `pose_arrays` move to modules/recording's recorder, beside the `FramePoses` they
  build (`pose/recorded_poses.py` today) **(seen)**, so the recorder packs the poses itself: the `pack_poses`
  argument of `Recorder`, `VideoRecorder` and `RecordingSync` goes, and the apps stop handing `frame_poses` in;
  their tests follow (`test_recorded_poses.py`, recording's `test_pose_packing.py` and the rigs that pass a packer).
  modules/recording then imports modules/pose, which does not import it back **(seen)**. The recording and pose
  threads are asked for feedback. Checked by pyright and the inference, pose, recording and apps' tests
- C3. INFERENCE.md's "What else was weighed (2026-11-18, online sources)" is restated as the constraints it sets,
  without its date or search: what rules out ViTPose, the one-stage models, Sapiens, the video models and RTMPose-x
  (docs.md, a rejected alternative as a constraint)
- C4. The reasons in docstrings and comments move to INFERENCE.md, each docstring saying what the thing is:
  `CropExtractorSettings.padding`, `nv12_to_rgb`, `StagingRing`'s module docstring, `KEPT`, and any others the
  sweep finds in the package. Form only, checked by pyright and the inference tests
- C5. Last: `SerialUploader`, `BatchUploader`, `CameraSync` and `InferenceFrameSync` own their thread instead of
  being one, as `ModelWorker` and `RecordedPoses` do (CLAUDE.md, composition over inheritance; the part that works on
  the item has its own thread): a `_thread`, with `start` and `stop` of their own; callers that use `Thread`'s own
  API (`join`, `is_alive`, `daemon`) follow, the five apps among them (smile has a `SerialUploader`) **(seen)**. Checked by pyright, the inference and apps' tests, and a `-sim` start
  of hd_trio and deep_flow (both uploaders, both syncs)

## The rules it touches

- `CLAUDE.md` Composition and wiring, data-flow naming (`process`): C1
- `CLAUDE.md` Packages, one thing per package; Imports, layering: C2
- `.claude/rules/docs.md`, a rejected alternative as a constraint: C3
- `CLAUDE.md` Docstrings, a docstring says what the thing is: C4
- `CLAUDE.md` Composition, composition over inheritance; Concurrency, the part has its own thread: C5
