# The inference package's plan

What is not built in `modules/inference`, apart from the threads the app refactor moves
([REFACTOR_PLAN.md](../../REFACTOR_PLAN.md)). What is built is in [INFERENCE.md](INFERENCE.md), its times and how
the refactor's steps were checked in [TIMING.md](TIMING.md). **(deduction)** marks what is inferred.

Questions (from the module refactor's package scan of 2026-11-20, with this thread's answer):
- `CropExtractor.submit` → `process`? It cuts and calls its callbacks on the caller's thread: yes, by the naming rule
- `has_points` to modules/pose, `frame_poses` / `pose_arrays` to modules/recording? Yes: one tests a pose `Frame`,
  the others build recording's `FramePoses`
- The reasons in docstrings (`CropExtractorSettings.padding`, `nv12_to_rgb`, `StagingRing`, `KEPT`) to
  INFERENCE.md in one sweep? Yes
- Threads: subclass `Thread` (SerialUploader, BatchUploader, CameraSync, InferenceFrameSync) or own one (ModelWorker,
  RecordedPoses)? Own one, by composition over inheritance; form only, last
- INFERENCE.md's dated "What else was weighed": restate as constraints? Yes
- Each keypoint's uncertainty for the One Euro filter: in the pose package's plan ([POSE_PLAN.md](../pose/POSE_PLAN.md))
