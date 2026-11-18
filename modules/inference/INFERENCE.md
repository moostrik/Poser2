# Inference

The camera frames on the GPU and the models that run on them. An app's cameras hand their frames to an uploader,
which puts them on the GPU once; the crop step cuts each player's box from those images, and the model workers run
on the crops or the whole images, each on its own thread and CUDA stream. The upload and the models are one package:
every app uses them together, and the crop step and the detectors read the upload's images. What is not built is in
[INFERENCE_PLAN.md](INFERENCE_PLAN.md); how the threads of an app hand on is in
[INPUT_THREADS.md](../flow/INPUT_THREADS.md). **(deduction)** marks what is inferred.

## The parts

| Part                                 | File                                  | Its thread     | Does                                                               |
|--------------------------------------|---------------------------------------|----------------|--------------------------------------------------------------------|
| `SerialUploader`                     | `serial_uploader.py`                  | its own        | each frame of one camera onto the GPU, the moment it comes         |
| `CameraSync`, `BatchUploader`        | `camera_sync.py`, `batch_uploader.py` | its own, each  | several cameras' frames of one moment, as one set, onto the GPU    |
| `ImageUpload`                        | `image_upload.py`                     | the uploader's | the GPU work both uploaders share                                  |
| `DepthAligner`                       | `depth_align.py`, `cuda_kernels.py`   | the uploader's | the depth moved onto the color's view                              |
| `CropExtractor`                      | `crop_extractor.py`, `crop.py`        | the caller's   | every player's box cut at every size, in one pass                  |
| `PoseEstimator`                      | `pose/`                               | its own        | RTMPose, 17 points a player                                        |
| `Segmenter`                          | `segmentation/`                       | its own        | Robust Video Matting, a mask a player                              |
| `OpticalFlowEstimator`               | `optical_flow/`                       | its own        | RAFT, the flow from a player's previous crop to this one           |
| `PersonDetector`                     | `detection/`                          | its own        | YOLOv8n, the players' boxes on the host (replay)                   |
| `FaceDetector`                       | `face/`                               | its own        | SCRFD and the 106 landmarks, the faces (smile)                     |
| `ModelWorker`                        | `model_worker.py`                     |                | the five workers' base                                             |
| `Engine` (`OnnxEngine`, `TrtEngine`) | `engine/`                             | the worker's   | a model file run on GPU tensors, ONNX Runtime or TensorRT          |
| `CaptureSync`                        | `capture_sync.py`, `capture.py`       | its own        | an image, its depth, poses and masks of one moment, as a `Capture` |
| `RecordedPoses`                      | `recorded_poses.py`                   | the caller's   | the recorded poses, in place of the pose model in -sim             |

## From the frame to the model

No thread waits for the GPU between the upload, the crop and the model. Each step queues its work on its own stream
behind an event of the step before, and hands its results on with an event of its own, which every reader waits on
(`Stream.wait_event`) before it reads; a tensor read on a stream other than the one that made it is marked there
(`record_stream`), so the cache does not hand its memory out again before that stream has run. PyTorch's streams run
in any order unless they are synchronized so
**(site fact: [CUDA streams](https://docs.pytorch.org/docs/stable/notes/cuda.html#cuda-streams))**.

- The uploader hands each image on as a `GpuImage` with its `ready` event, and as an `ImageSnapshot`: each
  camera's image and its previous one, their capture times, the event (`source.py`).
- The crop step takes the poses with the snapshot they are for and cuts from exactly those images, never a newer
  one, at each size it was made with (the pose model's, and where a segmentation runs, its own), from one box per
  player. All sizes share one `crops_ready` event; each size has its own callbacks. It cuts the RAW stage's smoothed
  and predicted box, so it runs after that stage, on the frame source's thread, not in the upload.
- Each model worker waits for the crops' event on its stream and runs there. The pose estimator synchronizes once,
  at the end, to decode its points on the CPU; the segmentation's masks stay on the GPU with an event of their own
  (`segmentation.Image.ready`), as do the flows.

### The model workers

`ModelWorker` is a model on its own thread and stream. A worker queues an item under a key and returns: the newest
item per key waits, and one it replaces is counted in the worker's `settings.dropped`. The pose estimator, the
segmenter and the optical flow estimator keep one key, the batch of players; the detectors one per camera, each
camera's newest image. The thread loads its model on its stream, runs it once at its largest batch so no kernel is
made while running, and then runs every waiting item. Each worker keeps its model's own steps (the pose's ImageNet
normalization and SimCC decode, the segmentation's recurrent states, RAFT's 0 to 255 input) and its callbacks.

The engines run a model's named GPU tensors on the worker's stream. `TrtEngine` gives each worker its own TensorRT
execution context on the worker's thread, which needs no lock between contexts (TensorRT's thread-safety model,
[`engine/trt_runtime.py`](engine/trt_runtime.py)), so the models run side by side. `OnnxEngine` runs ONNX Runtime's CUDA provider on the worker's stream
(`user_compute_stream`), so it waits no more than TensorRT; its outputs come out as torch tensors without a copy
(DLPack). ONNX Runtime plans its kernels per shape, so the ONNX engine pads every batch to the size it was warmed up
at and cuts the outputs back.

| Model               | Worker                 | Input (width x height)  | Precision  | Batch, ONNX / TensorRT |
|---------------------|------------------------|-------------------------|------------|------------------------|
| RTMPose-l           | `PoseEstimator`        | 192x256 or 288x384      | FP32       | players, up to 8       |
| RVM MobileNetV3     | `Segmenter`            | 192x256 to 768x1024     | FP16       | players, up to 4       |
| RAFT Sintel, 12 it. | `OpticalFlowEstimator` | 192x256                 | FP32       | players, up to 8 / 4   |
| YOLOv8n             | `PersonDetector`       | 480x384 to 640x512      | the file's | one image per camera   |
| SCRFD 10G, 2d106det | `FaceDetector`         | 640x640, 192x192        | the file's | an image, its faces    |

The pose model stays FP32: it loses precision in FP16. The times, before and after each step and on both backends,
are in the plan's results ([INFERENCE_PLAN.md](INFERENCE_PLAN.md#the-check)).

## The upload

`SerialUploader` is the camera's one hand-off: each frame onto the GPU on its own, the moment it comes, so the
color goes up at once to start the pose. `submit_frame` queues the frame and returns: two frames of a kind per
camera wait, and a third drops the oldest, counted (`dropped`). Its own thread makes the corrected camera frame,
which every user of the camera needs:

- the color: `ImageUpload` copies it from a pinned staging buffer (`STAGING_COUNT` 3, reused once the copy queued
  from it has run), converts it (BGR, NV12 or mono) to RGB half, straightens it by the tilt's homography and mirrors
  it, in one CUDA kernel;
- the depth: copied the same way, turned into metres and moved onto the color's view at a grid as fine as the depth
  itself (`aligned_depth_size`), the nearest kept where two land on one cell and 0 where nothing lands, straightened
  and mirrored as the color, handed on as a `GpuDepth`.

`CameraSync` groups one moment's frames of several cameras into a set, which `BatchUploader.submit_set` puts on the
GPU as one stack (one copy, one round of kernels, when the frames share a size). It stays in this package: it
groups the frames for the upload, the input's first step, and holds nothing of a camera.

The kernels are CUDA C compiled at runtime by the NVRTC torch ships (`cuda_kernels.py`, through ctypes, no extra
package). `camera.uploader.depth_kernels` off, or kernels that cannot be compiled or loaded (logged once), run the
same in torch ops, the reference the tests and the harness hold the kernels to; the depth's then run as one CUDA
graph per lens, homography and size. The torch ops are about a hundred small kernels; launched one by one they held
the uploader 12 ms a frame while the GPU's queue was full, as each launch waits there. `scripts/depth_align_check.py`
holds both to the GL `DepthAlign` shader on recorded frames.

| Depth alignment, P800 recording | GPU a frame         | Pixels off the GL shader by over 1 mm |
|---------------------------------|---------------------|---------------------------------------|
| CUDA kernels (the default)      | 0.07 ms, with copy  | about 1 a frame, at borders           |
| GL `DepthAlign`                 | 0.13 ms             | the reference                         |
| torch ops                       | 0.6 to 1.3 ms       | under 2 a frame, at borders           |

## The capture sync

`CaptureSync` matches an image, its depth and the poses of one moment by capture time. Its submits queue the parts,
at most 8 of a stream per camera (the oldest dropped, `dropped` on the panel), and its own thread matches them and
hands every complete capture, in the order of capture, on. With `pose_wait` 0 a capture never waits for its poses:
it carries the poses of the frame before.

## Checks

- The package's tests (`tests/` and each model's `tests/`): the queues of the workers with a fake model, the crop
  step, the uploaders, the syncs, the depth alignment, the detectors on stub and real models.
- `scripts/model_benchmark.py`: an app's models without cameras, their times and their outputs against an earlier
  run, on ONNX or TensorRT.
- `scripts/gpu_input_timing.py` and `scripts/depth_align_check.py`: the upload's kernels against torch ops and GL.
