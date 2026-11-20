# The inference package's plan

What is not built in `modules/inference`, apart from the threads the app refactor moves
([REFACTOR_PLAN.md](../../REFACTOR_PLAN.md)). What is built is in [INFERENCE.md](INFERENCE.md), its times and how
the refactor's steps were checked in [TIMING.md](TIMING.md). **(deduction)** marks what is inferred. From the module
refactor's package scan of 2026-11-20 (INFERENCE_SCAN.md, in the project's files).

## A. Model files

- A1. One way to find a model file. Today the pose and the segmentation join `model_path` and a `model` that already
  starts with `data/models/`, so their ONNX file is looked for in `apps/<app>/data/models/data/models/` **(seen)**;
  the person and the face have `model_path` for ONNX, `engine_path` for engines and a `model_file` **(seen)**. The
  ONNX files sit once in the repo's `data/models` (LFS) and the engines in each app's folder, so the two folders
  stay:
  - one base, `ModelWorkerSettings(BaseSettings)` (`engine/model_worker_settings.py`), after the `ModelWorker` it
    configures, replacing the Protocol of that name in `engine/model_worker.py` **(seen)**: `model_type`,
    `onnx_path`, `engine_path`, `verbose` and `model_file(name)`, the whole path, in the shape face's already has
    **(seen)**; pose, segmentation, person and face extend it, each naming its file; the workers open `model_file`;
    `model_path` goes. `BaseSettings` takes a base's fields first, in the panel and the preset, and a field a subclass
    declares again keeps the base's place **(seen: base_settings.py)**: each subclass declares `model_type` again
    with its default (ONNX for face and person, TRT for pose and segmentation) **(seen)**; `dropped`, a monitor,
    stays in each subclass so it stays last (panels.md); `enabled` comes after the model files **(deduction)**
  - the apps' `settings.py`: hd_trio, white_space, deep_flow and low_flow declare `model_path` on their inference
    group and share it into the pose and the segmentation (`_batch_share`, white_space's pose share) **(seen)**;
    they declare and share `onnx_path` and `engine_path`
  - the presets rename the keys, in all four apps and smile
  - the camera's `model_path` (the OAK's blobs, `modules/camera`) is the camera's folder and stays
  - `scripts/model_benchmark.py` follows, and stops reading the pose's `resolution` (gone in 0a7709de) **(seen)**
  - a test of each `model_file`, ONNX and TensorRT
  Checked by pyright, the inference and apps' tests, and the pose and segmentation on ONNX in `model_benchmark.py`
- A2. low_flow loads its own engines: its presets point `engine_path` and the pose's `model_path` at
  `apps/deep_flow/data/models` **(seen)**; an app stands on its own, with its own data (decided 2026-11-18), so they
  name `apps/low_flow/data/models`, which `makemodels.bat`'s low_flow section builds. Checked by a low_flow `-sim`
  start: its engines load from its own folder

## B. Leftovers

- B1. `Resolution` and `RESOLUTION_DIMS` move from `engine/model_types.py` to `segmentation/settings.py`, its only
  user **(seen)**
- B2. Removed, no caller **(seen)**: `ModelType.NONE` (its `""` branches go with A1); `PoseEstimatorSettings.aspect_ratio`;
  `ImageUpload.reset`; `CameraSyncSettings.num_cams`, its shares in hd_trio and white_space and its preset keys
- B3. One pinned staging ring: `SerialUploader._staging_buffer` is a copy of `ImageUpload._staging_buffer`, and
  SerialUploader owns an `ImageUpload`, both used on its one thread **(seen)**; its depth takes its buffer from that
  `ImageUpload`'s ring, and its own copy goes
- B4. Form only: `normalise` → `normalize` (person/decode.py) and the British spellings in comments, "metre(s)"
  among them (CLAUDE.md says meters since afcd0d11); the stale texts
  (`KEPT`'s "bangs" in crop_extractor.py, `shaders/depth_align.py`'s `geometry.depth_projection`, INFERENCE.md's
  RecordedPoses users, its worker count and person sizes); `CropExtractorSettings` into `crop_extractor.py`, the
  apps' imports following; `import logging` in PEP 8 order in camera_sync.py and ort_runtime.py; RecordedPoses'
  docstring says it is a player, whose thread makes each item at its recorded moment
- B5. A test of OnnxEngine padding a batch to `max_batch` and cutting the results back
  B1 to B5 are checked by pyright and the inference and apps' tests, B3 by the uploaders' among them

Questions (the module refactor thread's and this thread's answer):
- `CropExtractor.submit` → `process`? It cuts and calls its callbacks on the caller's thread: yes, by the naming rule
- `has_points` to modules/pose, `frame_poses` / `pose_arrays` to modules/recording? Yes: one tests a pose `Frame`,
  the others build recording's `FramePoses`
- The reasons in docstrings (`CropExtractorSettings.padding`, `nv12_to_rgb`, `_staging_buffer`, `KEPT`) to
  INFERENCE.md in one sweep? Yes
- Threads: subclass `Thread` (SerialUploader, BatchUploader, CameraSync, InferenceFrameSync) or own one (ModelWorker,
  RecordedPoses)? Own one, by composition over inheritance; form only, last
- INFERENCE.md's dated "What else was weighed": restate as constraints? Yes
- Each keypoint's uncertainty for the One Euro filter: in the pose package's plan ([POSE_PLAN.md](../pose/POSE_PLAN.md))

## The rules it touches

- `CLAUDE.md` API evolution, update all call sites; keep presets in sync: A1, B2
- `CLAUDE.md` Naming, one term per concept: A1, B1
- `CLAUDE.md` Glossary, American spelling: B4
- `CLAUDE.md` Packages, one class or subject per file: B4
- `CLAUDE.md` Testing, unit-test infrastructure: A1, B5
- `.claude/rules/inference.md`, `_fp16` in a model file's name: A1
