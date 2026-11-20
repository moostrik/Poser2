# The inference package's plan

What is not built in `modules/inference`, apart from the threads the app refactor moves
([REFACTOR_PLAN.md](../../REFACTOR_PLAN.md)). What is built is in [INFERENCE.md](INFERENCE.md), its times and how
the refactor's steps were checked in [TIMING.md](TIMING.md). **(deduction)** marks what is inferred.

## A. The pose model

- A1. One pose model, RTMPose-l at 288x384 (`HIGH`), in every app; `STANDARD` goes (decided 2026-11-20). Alone it
  costs 0.3 to 0.4 ms more than `STANDARD` at full batches; in the app a full room costs 2 to 3 ms more, one run each
  (TIMING.md, "The pose's resolution for several players") **(seen)**, most of it the wait for the GPU it shares
  **(deduction)**; for 78.3 against 76.7 COCO AP **(site fact)**. With one model the pose's `resolution` has nothing
  to choose, so it goes:
  - `PoseEstimatorSettings.resolution` and `_MODELS`; `width`, `height` and `model` are the one model's
    (`pose/settings.py`); the apps' crops keep taking the pose's size (their `main.py`); the engine's `Resolution`
    and `RESOLUTION_DIMS` stay, the segmentation chooses by them
  - deep_flow's and low_flow's `main.py` comment on "a change of pose resolution" says what it guards
  - the presets' `pose.pose.resolution`, in all four apps (hd_trio and white_space run `STANDARD` today **(seen)**)
  - in LFS `rtmpose-l_256x192.onnx` and `rtmpose-l_256x192_fp16.onnx`; the `rtmpose-l_256x192*.trt` engines in the
    apps' model folders; their lines in `scripts/makemodels.bat`, which gains low_flow's 384x288 line (it has
    none); the 256x192 examples in `tools/models`'s docstrings (`convert_onnx_to_fp16.py`,
    `diagnose_fp16_weights.py`, `diagnostic_precision.py`, `export_rtm_onnx_to_trt.py`) name the 384x288
  - `test_pose_estimator.py`'s resolution, and its ULTRA case with the "no model" error it tests; INFERENCE.md's
    lines on the two resolutions; TIMING.md keeps its numbers
  Checked by pyright, the inference and four apps' tests, and a `-sim` start of hd_trio and white_space: "Pose
  model ready: 288x384 rtmpose-l_384x288_fp16.trt", and no drops

Questions:
- Each keypoint's uncertainty for the One Euro filter: in the pose package's plan ([POSE_PLAN.md](../pose/POSE_PLAN.md))

## The rules it touches

- `CLAUDE.md` API evolution, remove obsolete pathways; keep presets in sync: A1
