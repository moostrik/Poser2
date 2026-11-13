# smile's plan

smile's faces on the inference path the other apps use (2026-11-13). Today `modules/face` holds two ONNX models
(SCRFD face detection, the 106-point landmarks) that work on numpy frames with cv2 preprocessing and set up ONNX
Runtime themselves, beside smile's own choice of a subject (the gates, the portrait crop). After this plan the
models are a model kind of `modules/inference`, on TensorRT, fed with uploaded images as the person detector is, for
any app that needs faces, with what the landmarks measure (head pose, eyes); what smile does with the faces (the
gates, the subject, the portrait) is smile's; `modules/face` is gone. One step per commit, each
checked by the tests named; after the last, smile runs once on the image input (20 s, a face in every frame, as
before) and, with ComfyUI started, through one whole visitor (its portrait crop as before).

The face models are built in `modules/inference/face` (`FaceDetector`, `FaceFrame`, `Face`, `measures`, ONNX and
TensorRT runners, `FaceDetectorSettings` with `pitch_zero`; the engines by `scripts/makemodels.bat smile`), and a
`GpuImage` keeps its `CameraFrame` (`frame`). A `Face` is in the image's pixels, its `FaceFrame` with the image's
size. `modules/face` is still there, for smile.

## 2. smile on the face detector

smile's camera source hands each fitted frame, as a `CameraFrame` (camera 0, video, its time), to a
`SerialUploader` (color only), as the other apps' cameras do. smile hands the `FaceDetector` the uploaded images
it wants judged, as it hands the face finder its frames today: in the camera states (CAMERA_SHOW to CAMERA_DETECT)
and new frames only, resetting at CAMERA_SHOW, so nothing is detected while the generator works on the same GPU.
What smile does with the faces moves into the app (`apps/smile/face/`): the gates and the subject (`gate`,
`distance`, the largest qualifying face, the rejection and its label), `best()` (the portrait crop of the last
`FaceFrame` the subject qualified in, cut from that frame's own CPU frame: no frames kept by time), the pitch
calibration (writing the detector's `pitch_zero`), and its settings (the gates' limits, the zoom and offset of the
portrait, beside the detector's, which its preset sets to TRT with `engine_path` `apps/smile/data/models`). Its judged faces per frame
(today's `FaceFrame` of `FaceObservation`s, named in the step after what they are) and their users (the render's
marks, the camera layer, the online turn, the sequencer's charge) take their names from there. `modules/face/crop.py`, the command-line portrait crop, moves to smile with it.
`modules/face` goes, its tests with what they test.
Checked by: smile's tests and the moved face tests; the image-input run and the visitor run above.

## Rules it touches

- A package is one thing, and its name is that thing: 1, 2.
- `modules/` must never import app code; an app imports modules: 2.
- Fix a problem where it is made: the models' setup once, in `inference`: 1.
- Remove obsolete pathways: `modules/face` and its CPU preprocessing go: 2.
