# smile's plan

smile's faces on the inference path the other apps use (2026-11-13). Today `modules/face` holds two ONNX models
(SCRFD face detection, the 106-point landmarks) that work on numpy frames with cv2 preprocessing and set up ONNX
Runtime themselves, beside smile's own choice of a subject (the gates, the portrait crop). After this plan the
models are a model kind of `modules/inference`, on TensorRT, fed with uploaded images as the person detector is, for
any app that needs faces, with what the landmarks measure (head pose, eyes); what smile does with the faces (the
gates, the subject, the portrait) is smile's; `modules/face` is gone. One step per commit, each
checked by the tests named; after the last, smile runs once on the image input (20 s, a face in every frame, as
before) and, with ComfyUI started, through one whole visitor (its portrait crop as before).

## 1. The face models in `modules/inference/face`

A `GpuImage` keeps the `CameraFrame` it was uploaded from (`frame`, a reference, set by `ImageUpload`'s users,
the `SerialUploader` and the `BatchUploader`), so the uploaded image and the CPU frame of one moment travel as one.

`inference/face/`, shaped like `inference/detection`: `FaceDetector`, a thread that takes uploaded images
(`submit(GpuImage)`, the newest of a camera replacing one still waiting) and hands each image's faces to its
callbacks (`add_face_callback`) as one `FaceFrame`, as a pose frame carries its images: the image's time, the
`GpuImage` the faces were found in (its CPU frame with it), carried unread, and per face its box, its five points
and score, normalised to the image, its 106 landmarks, and what they measure (`head_pose`, `eye_openness`,
`eye_distance`, from `quality.py`). The
image is fitted into SCRFD's 640 input and the landmarks' 192 crops on the GPU (`grid_sample`, as the person
detector stretches its images), the decode (anchors, NMS) as now. Each model runs on TensorRT or ONNX Runtime
(`ModelType`, a `runner_trt` and a `runner_onnx` as in `detection`; ONNX Runtime set up once,
`ort_runtime.preload_cuda`); the landmarks run the faces of an image as one batch. The engines are built by
`tools/models/export_face_onnx_to_trt.py` (SCRFD at batch 1, the landmarks at a batch of faces: `2d106det.onnx`
runs a batch, `(3, 212)` for 3 faces, but declares its output at batch 1, which the script makes dynamic first), from
`scripts/makemodels.bat smile`, into `apps/smile/data/models/`. Its settings: the model type, the model and engine
paths, the detection threshold; smile's preset sets TRT.
Checked by: its tests: on the test photos of `apps/smile/data/input/`, the boxes, points and landmarks within a
pixel of today's CPU path, on ONNX and on TRT, and the measures equal; a newer image replacing one waiting; a
`FaceFrame` carrying its image and that image's `CameraFrame`; the uploaders' `GpuImage` keeping its frame.

## 2. smile on the face detector

smile's camera source hands each fitted frame, as a `CameraFrame` (camera 0, video, its time), to a
`SerialUploader` (color only), as the other apps' cameras do. smile hands the `FaceDetector` the uploaded images
it wants judged, as it hands the face finder its frames today: in the camera states (CAMERA_SHOW to CAMERA_DETECT)
and new frames only, resetting at CAMERA_SHOW, so nothing is detected while the generator works on the same GPU.
What smile does with the faces moves into the app (`apps/smile/face/`): the gates and the subject (`gate`,
`distance`, the largest qualifying face, the rejection and its label), `best()` (the portrait crop of the last
`FaceFrame` the subject qualified in, cut from that frame's own CPU frame: no frames kept by time), the pitch
calibration, and its settings (the gates' limits, the zoom and offset of the portrait). Its judged faces per frame
(today's `FaceFrame` of `FaceObservation`s, named in the step after what they are) and their users (the render's
marks, the camera layer, the online turn, the sequencer's charge) take their names from there. `modules/face/crop.py`, the command-line portrait crop, moves to smile with it.
`modules/face` goes, its tests with what they test.
Checked by: smile's tests and the moved face tests; the image-input run and the visitor run above.

## Rules it touches

- A package is one thing, and its name is that thing: 1, 2.
- `modules/` must never import app code; an app imports modules: 2.
- Fix a problem where it is made: the models' setup once, in `inference`: 1.
- Remove obsolete pathways: `modules/face` and its CPU preprocessing go: 2.
