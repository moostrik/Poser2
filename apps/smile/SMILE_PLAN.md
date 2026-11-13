# smile's plan

smile's faces on the inference path the other apps use (2026-11-13). Both steps are built: the face models are
`modules/inference/face` (`FaceDetector`, `FaceFrame`, `Face`, `measures`, ONNX and TensorRT runners,
`FaceDetectorSettings` with `pitch_zero`; the engines by `scripts/makemodels.bat smile`), smile uploads its frames
through a `SerialUploader` to the `FaceDetector`, and what smile does with the faces (the gates, the subject, the
portrait, the pitch calibration, the command-line crop) is in `apps/smile/face/` and `apps/smile/scripts/crop_faces.py`;
`modules/face` is gone.

## The runs

smile runs once on the image input (20 s, a face in every frame, as before) and, with ComfyUI started, through one
whole visitor (its portrait crop as before).

## Rules it touches

- A package is one thing, and its name is that thing: 1, 2.
- `modules/` must never import app code; an app imports modules: 2.
- Fix a problem where it is made: the models' setup once, in `inference`: 1.
- Remove obsolete pathways: `modules/face` and its CPU preprocessing go: 2.
