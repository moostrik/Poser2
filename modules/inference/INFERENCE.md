# Inference

The camera frames on the GPU and the models that run on them. An app's cameras hand their frames to an uploader,
which puts them on the GPU once; the crop step cuts each player's box from those images, and the model workers run
on the crops or the whole images, each on its own thread and CUDA stream. The upload and the models are one package:
every app uses them together, and the crop step and the detectors read the upload's images. What is not built is in
[INFERENCE_PLAN.md](INFERENCE_PLAN.md), the models' times in [TIMING.md](TIMING.md); how the threads of an app hand
on is in [INPUT_THREADS.md](../flow/INPUT_THREADS.md). **(deduction)** marks what is inferred.

No code in the package is one app's: the face model is smile's alone, the segmentation hd_trio's and low_flow's,
`RecordedPoses` deep_flow's, but each is general code with one user, as the camera's backends are.

## The parts

`preprocessing/` makes the input ready for the models, the upload and the crop step; `engine/` holds what the
models share; each model has its package; the inference frame and its sync, which come after the models, sit at the root.

| Part                                 | File                                                | Its thread     | Does                                                               |
|--------------------------------------|-----------------------------------------------------|----------------|--------------------------------------------------------------------|
| `SerialUploader`                     | `preprocessing/serial_uploader.py`                  | its own        | each frame of one camera onto the GPU, the moment it comes         |
| `CameraSync`, `BatchUploader`        | `preprocessing/camera_sync.py`, `batch_uploader.py` | its own, each  | several cameras' frames of one moment, as one set, onto the GPU    |
| `ImageUpload`                        | `preprocessing/image_upload.py`                     | the uploader's | the GPU work both uploaders share                                  |
| `DepthAligner`                       | `preprocessing/depth_align.py`, `cuda_kernels.py`   | the uploader's | the depth moved onto the color's view                              |
| `CropExtractor`                      | `preprocessing/crop_extractor.py`, `crop.py`        | the caller's   | every player's box cut at every size, in one pass                  |
| `PoseEstimator`                      | `pose/`                                             | its own        | RTMPose, 17 points a player                                        |
| `RecordedPoses`                      | `pose/recorded_poses.py`                            | the caller's   | the recorded poses, in place of the pose model in -sim             |
| `Segmenter`                          | `segmentation/`                                     | its own        | Robust Video Matting, a mask a player                              |
| `PersonDetector`                     | `person/`                                           | its own        | YOLOv8n, the players' boxes on the host (replay)                   |
| `FaceDetector`                       | `face/`                                             | its own        | SCRFD and the 106 landmarks, the faces (smile)                     |
| `ModelWorker`                        | `engine/model_worker.py`                            |                | the five workers' base                                             |
| `Engine` (`OnnxEngine`, `TrtEngine`) | `engine/`                                           | the worker's   | a model file run on GPU tensors, ONNX Runtime or TensorRT          |
| `InferenceFrameSync`                 | `inference_frame_sync.py`, `inference_frame.py`     | its own        | an image and the results of one moment, as an `InferenceFrame`     |

## From the frame to the model

No thread waits for the GPU between the upload, the crop and the model. Each step queues its work on its own stream
behind an event of the step before, and hands its results on with an event of its own, which every reader in the
package waits on (`Stream.wait_event`) before it reads. A reader outside the package gets its items finished: a GL
reader cannot wait for a CUDA event on the GPU, its PBO copy's unmap would hold the GL thread instead
(`modules/gl/tensor.py`, `_update_with_pbo`). A producer with readers on both sides calls the ones in the package
first, then waits for its stream and calls its finished callbacks, so none in the package waits for it; a tensor read on a stream other than the one that made it is marked there
(`record_stream`), so the cache does not hand its memory out again before that stream has run. PyTorch's streams run
in any order unless they are synchronized so
**(site fact: [CUDA streams](https://docs.pytorch.org/docs/stable/notes/cuda.html#cuda-streams))**.

- The uploader hands each image on as a `GpuImage` with its `ready` event, and as an `ImageSnapshot`: each
  camera's image, their capture times, the event (`preprocessing/source.py`). Then it hands
  the set's capture times to the capture callbacks, and then, for the readers outside the package, it waits for its
  stream and hands the images and the depth on written (`add_finished_gpu_image_callback`,
  `add_finished_gpu_depth_callback`), only when such a callback is registered. The readers in the package are queued
  before that wait, so none of them waits for it.
- The crop step keeps the uploader's last snapshots (`submit_snapshot`), takes the poses with the capture times
  they are for, which the frame source and the pose pipeline carry without the images, and cuts from exactly the
  images of those times, never a newer one, at each size it was made with (the pose model's, and where a segmentation runs, its own), from one box per
  player. All sizes share one `crops_ready` event; each size has its own callbacks, and its finished callbacks
  for readers outside the package, called after the crop stream has run. It cuts the RAW stage's smoothed
  and predicted box, so it runs after that stage, on the frame source's thread, not in the upload. A size is cut
  every `interval`-th set (the pose's every other frame, the segmentation's every frame); after the crops it tells
  each size's asked callbacks per camera which poses were cut, none on a passed set, so the inference frame sync
  and the recorder do not wait for poses that do not come. Each crop carries the box it was cut at
  (`crop.Image.box`), and the segmentation's mask takes it (`segmentation.Image.box`), where the mask lies in the
  camera image.
- Each model worker waits for the crops' event on its stream and runs there. The pose estimator synchronizes once,
  at the end, to decode its points on the CPU; the segmenter synchronizes once at the end too, so its
  masks are written when handed on, for readers outside the package. The decode stays on the CPU: on the GPU it
  gives the same points but takes longer, its kernel launches costing more than copying the model's outputs
  ([TIMING.md](TIMING.md#the-poses-decode)).

### The model workers

`ModelWorker` is a model on its own thread and stream. A worker queues an item under a key and returns: the newest
item per key waits, and one it replaces is counted in the worker's `settings.dropped`. The pose estimator and the
segmenter keep one key, the batch of players; the detectors one per camera, each
camera's newest image. The thread loads its model on its stream, runs it once at its largest batch so no kernel is
made while running, and then runs every waiting item. Each worker keeps its model's own steps (the pose's ImageNet
normalization and SimCC decode, the segmentation's recurrent states) and its callbacks.

The engines run a model's named GPU tensors on the worker's stream. `TrtEngine` gives each worker its own TensorRT
execution context on the worker's thread, which needs no lock between contexts (TensorRT's thread-safety model,
[`engine/trt_runtime.py`](engine/trt_runtime.py)), so the models run side by side. `OnnxEngine` runs ONNX Runtime's CUDA provider on the worker's stream
(`user_compute_stream`), so it waits no more than TensorRT; its outputs come out as torch tensors without a copy
(DLPack). ONNX Runtime plans its kernels per shape, so the ONNX engine pads every batch to the size it was warmed up
at and cuts the outputs back.

| Model               | Worker                 | Input (width x height)  | Precision  | Batch, ONNX / TensorRT |
|---------------------|------------------------|-------------------------|------------|------------------------|
| RTMPose-l           | `PoseEstimator`        | 192x256 or 288x384      | FP16 (1)   | players, up to 8       |
| RVM MobileNetV3     | `Segmenter`            | 192x256 to 768x1024     | FP16       | players, up to 4       |
| YOLOv8n             | `PersonDetector`       | 480x384 to 640x512      | the file's | one image per camera   |
| SCRFD 10G, 2d106det | `FaceDetector`         | 640x640, 192x192        | the file's | an image, its faces    |

(1) The pose runs FP16 in every app, with its head's two ScaleNorms, its input and its output in FP32: half the time
of FP32, 97% of its points within one SimCC bin of FP32's ([TIMING.md](TIMING.md#the-pose-models)).

### The models and their files

The models' ONNX files sit in `data/models`, in LFS; an engine is built from one on each PC by `scripts/makemodels.bat`,
per app at its batch, as an engine runs only on the TensorRT version and the GPU that built it. TensorRT 11 builds
strongly typed: each layer's precision is its ONNX file's, so an FP16 engine is built from an FP16 ONNX file
(`tools/models/export_rtm_onnx_to_trt.py`). A file's name says `_fp16` when it is FP16 and nothing when it is FP32,
its engine's the same, and the workers log the file they load: the pose's and RVM's (`rvm_mobilenetv3_<size>_fp16`)
are FP16. LFS therefore keeps what an engine is built from and the original it was
made from: the pose's FP32 original (`rtmpose-l_384x288.onnx`) and the FP16 file the
pose runs (`rtmpose-l_384x288_fp16.onnx`), made once by
`uv run --with onnxconverter-common python tools/models/convert_onnx_to_fp16.py --fp32-norms --keep-io-types`. Its
library is not one of the project's: it runs only when a model is added. `--fp32-norms` finds each L2 norm by its
structure, as either export writes it (`ReduceL2`, or a square, `ReduceSum` and `Sqrt`), and keeps it FP32: its sum of
squares can overflow FP16's 65504 in other runtimes **(site fact: a LiteRT port of RTMPose)**.

The pose is RTMPose-l trained on Body8, OpenMMLab's own checkpoint (76.7 COCO AP at 256x192, 78.3 at 384x288)
**(seen: its weights against OpenMMLab's file; site fact: the AP)**. Every app runs the one model, 288x384
(`PoseEstimatorSettings`): in the app a full room costs 2 to 3 ms more than at 192x256
([TIMING.md](TIMING.md), "The pose's resolution for several players"), for 1.6 AP. What else was weighed
(2026-11-18, online sources) sets these limits **(site fact)**: ViTPose gains as much for several times the time and
needs a heatmap decode; the one-stage models (RTMO, YOLO-pose) score below RTMPose-l; Sapiens is too heavy and its
weights non-commercial; the video models need future frames, 33 to 67 ms more; RTMPose-x gains 0.5 AP over
RTMPose-l at 384x288 for nearly twice the file and the time.

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

The kernels are CUDA C compiled at runtime by the NVRTC torch ships (`preprocessing/cuda_kernels.py`, through ctypes, no extra
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

## The inference frame sync

`InferenceFrameSync` matches an image and its results of one moment by capture time, as an `InferenceFrame`
(`inference_frame.py`): its identity (`cam_id`, `timestamp`, the image) and its results keyed by their type
(`Depth`, `Poses`, `Masks`), read as `frame[Poses]`, which never raises; a type the frame does not carry reads as
its empty one (no players), as the pose `Frame` reads its features. Each result carries its camera, its image's
capture time and `synced`: this moment's own, or the newest from before. The sync is made with the types its app
runs (deep_flow: `Depth`, `Poses`), so each app's frames carry what it needs; a frame is complete with its image
and, where the app has depth, its depth, and it waits for each model's result by that type's setting (`pose_wait`,
`mask_wait`). Its submits queue the parts, at most 8 of a stream per camera (the oldest dropped, `dropped` on the
panel), and its own thread matches them, waits for a waiting frame's deadline on a `modules.utils.Waker` (to the
millisecond, and at once for a new part), and hands every complete frame, in the order of capture, on. With
`pose_wait` 0 a frame never waits for its poses: it carries the poses of the frame before.

## Checks

- The package's tests (`tests/` and each model's `tests/`): the queues of the workers with a fake model, the crop
  step, the uploaders, the syncs, the depth alignment, the detectors on stub and real models.
- `scripts/model_benchmark.py`: an app's models without cameras, the crop models on seeded crops and the detectors on
  one uploaded photo, their times and their outputs against an earlier run, on ONNX or TensorRT.
- `scripts/gpu_input_timing.py` and `scripts/depth_align_check.py`: the upload's kernels against torch ops and GL.
