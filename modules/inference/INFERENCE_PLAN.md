# The inference package's plan

What is not built in `modules/inference`, apart from the threads the app refactor moves
([apps/APPS_PLAN.md](../../apps/APPS_PLAN.md)). What is built is in [INFERENCE.md](INFERENCE.md), its times and how
the refactor's steps were checked in [TIMING.md](TIMING.md). **(deduction)** marks what is inferred.

## A. The TensorRT models side by side

Questions:
- CUDA graphs: capture each model's `execute_async_v3` per batch size and replay it, to cut the launch time on the
  CPU. Worth it, now the models run side by side?
- The pose's stream at a higher priority than the others', as the model on the critical path?

## B. The pose model

Questions:
- The licence of the Body8 training data (AI Challenger and others) is unresolved, for ours as for every candidate
- The workers log the engine's first input type as the model's precision, FP32 for the pose's FP16 models, whose
  inputs and outputs are FP32 **(seen)**: the weights' type, or the file's name, would say what runs
- Which pose resolution the presets run, `STANDARD` or `HIGH`, both FP16: yours to save
- Each keypoint's uncertainty for the One Euro filter: in the pose package's plan ([POSE_PLAN.md](../pose/POSE_PLAN.md))

## C. The results handed on finished (from low_flow's plan E, 2026-11-19)

An item that leaves its package is finished, its GPU work included (`CLAUDE.md` Concurrency, 2026-11-19). Inside
`modules/inference` a CUDA reader takes an item at once and waits for its `ready` on its own stream: the crop step,
the pose, the optical flow, the segmenter and the detectors do **(seen)**. A GL reader cannot wait on the GPU: its
PBO copy's unmap makes the GL thread wait for the CUDA work (`modules/gl/tensor.py`, `_update_with_pbo`) **(seen)**.
Today three producers hand on unfinished items outside the package **(seen)**:

- The segmenter waits for its model, then queues the mask's rescale and the foreground (crop x mask), records `ready`
  and hands on at once (`segmentation/segmenter.py`, `_infer`). low_flow's view waits on the mask
  (`ViewMask.place`): its GPU view takes 2.9 ms against deep_flow's 0.19 **(seen, the low_flow thread)**;
  hd_trio's mask and foreground layers wait the same way **(deduction)**
- The crop step hands its crops at once to hd_trio's and white_space's board (`main.py`, `set_crop_images`), and
  `CropImageLayer` waits for them in GL
- The uploaders (`ImageUpload.upload` records `ready` and returns; `SerialUploader` and `BatchUploader` call their
  callbacks at once) hand the images to the board (`set_camera_image`) and to the apps' callbacks, and the snapshot
  to the app's `_on_snapshot`. That one carries it to `TrackerFrameSource` (`modules/tracker`), which carries it
  unread through the pose pipeline (`modules/pose/pipeline`, `SourceFramesCallback`) to `CropExtractor.submit`: the
  images take the poses' route, though the tracker and the pipeline need only the bang, its capture times

A producer with readers on both sides has two outputs: its callbacks for the CUDA readers in the package are called
first, at once; then its thread waits for its stream (`synchronize()`) and calls its finished callbacks,
`add_finished_<noun>_callback`, only when one is registered. The CUDA readers are queued before the wait, so none of
them waits for it **(deduction)**.

- C1. The segmenter hands on finished masks: its readers are all outside the package (hd_trio's board,
  low_flow's `_on_masks`) **(seen)**, so its one `synchronize()` moves from right after the model to after
  the foreground; `ready` stays, recorded. No second output
- C2. The crop step gets the finished output for the board, `add_finished_image_callback`: after the image and asked
  callbacks, `CropExtractor.submit` waits for the crop stream, only when a due size has one, then calls them.
  hd_trio's and white_space's `set_crop_images` move to it
- C3. The uploaders get the finished output, `add_finished_gpu_image_callback`, for the board's `set_camera_image`,
  the inference frame sync's `submit_image` (deep_flow, low_flow: its `InferenceFrame` goes to the render and the
  flow) and the apps' camera-interval `_on_image` **(seen, `main.py`)**, and `add_finished_gpu_depth_callback` for
  deep_flow's `_on_depth`. The person and face detectors and the crop step keep the outputs as they are. A third
  output, `add_capture_callback`, hands on each set's capture times for the app's bang. In order: the snapshot and
  image callbacks, then the capture callbacks, so a bang never reaches the crop step before its snapshot; then the
  wait and the finished callbacks
- C4. The images go to the crop step straight from the uploader: `CropExtractor.submit_snapshot(snapshot)`, an
  uploader's snapshot callback, keeps the last few snapshots (as many as `TrackerFrameSource`'s `WAITING` + 1) by
  their capture times, under a lock of its own (the uploader's thread writes, the pipeline's reads). The bang carries
  only its capture times: the app's capture callback calls `TrackerFrameSource.submit(cam_timestamps)`, the pipeline
  hands them on (`SourceFramesCallback` and `AsyncInput.submit` take `cam_timestamps: dict[int, float]` in place of
  `images: Any`), and `CropExtractor.submit(frames, cam_timestamps)` cuts from the snapshot of those capture times.
  A bang whose snapshot is no longer kept is logged and cuts nothing, and its asked callbacks still hear it with no
  poses, so the inference frame sync does not wait for it. The four apps' `_on_snapshot` go. The tracker's and the
  pipeline's tests follow. `cam_times` becomes `cam_timestamps` everywhere (`tracker_frame_source.py`,
  `apps/APPS_PLAN.md`)
- C5. The readers outside the package drop their `wait_event` on `ready`, now that what they get is written, and
  keep their `record_stream`, so the memory is not handed out again while the render's stream still reads it **(seen)**:
  in `modules/render/modules`, `camera/camera_image_layer.py`, `inference/crop_image_layer.py`,
  `inference/inference_frame_image_layer.py`, `inference/inference_frame_depth_layer.py`,
  `inference/segmentation_mask_layer.py` and `inference/segmentation_foreground_layer.py`; low_flow's
  `render/layers/view/view_mask.py`; white_space's `light/layers/projection/camera_columns.py`
- `crop.py`'s `Image.ready` and `source.py`'s docstrings, and `INFERENCE.md`'s "From the frame to the model", say
  which output hands on finished items; `tracker_frame_source.py`'s and `spec.py`'s docstrings say the bang carries
  its capture times. The threads of `modules/tracker` and `modules/pose/pipeline` are asked for feedback on C4
- Checked by running each app as it is, then built, on the same recording, 60 s each, the numbers into
  `TIMING.md`:
  - low_flow `-sim` on `20261112-125832_measure`: the GPU view time, `mask_wait` and the capture latency
    (`CAPTURE_LATENCY`, a capture to the board) (C1, C3, C5)
  - deep_flow `-sim` on the same recording, which its flow was measured on (`modules/flow/TIMING.md`) **(seen)**:
    the pose latency (`POSE_LATENCY_RAW`, `POSE_LATENCY_SMOOTH`, a capture to the pose out of the model and out of
    the SMOOTH stage), the capture latency and the GPU view time (C3, C4, C5)
  - a timer on C2's and C3's waits, read in the same runs
  - Expected **(deduction)**: the pose latency stays; the GPU view time falls; the capture latency, which the
    board now gets after the wait, grows by at most the wait's timer
  - the inference, tracker, pose and hd_trio tests; by eye, hd_trio's segmentation and crop layers and white_space's
    crop and camera image
  About 5 minutes of runs

## The rules it touches

- `CLAUDE.md` Performance, prefer the design with the lower latency, measured in the context of the app: A, C
- `CLAUDE.md` Concurrency, an item that leaves its package is finished, its GPU work included: C1, C2, C3, C4, C5
- `CLAUDE.md` Concurrency, the part that works on the item does all its work on its own thread: C1
- `CLAUDE.md` Composition, fix a problem where it is made: C4
- `.claude/rules/glossary.md`, a time stamp is `timestamp`: C4
