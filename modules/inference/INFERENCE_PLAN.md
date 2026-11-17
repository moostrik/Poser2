# The inference package's plan

What is not built in `modules/inference`, apart from the threads the app refactor moves
([apps/APPS_PLAN.md](../../apps/APPS_PLAN.md)). How the input works is in
[INPUT_THREADS.md](../flow/INPUT_THREADS.md). **(deduction)** marks what is inferred.

## A. The color in one kernel

- A1. The uploader's color conversion, straightening and mirroring as one CUDA kernel, as the depth's alignment is
  (`depth_align.py`), for both of `ImageUpload`'s inputs: the OAK4's NV12 and the depthai 2 cameras' BGR
  (white_space, hd_trio). From NV12 23 GPU ops become 2, about 0.4 ms less GPU a frame **(deduction)**. Measured per
  step by `scripts/gpu_input_timing.py` before and after, on a recording here and then on the camera at the next live
  session.

## B. The crops' event

- B1. Every predictor hands its crops' event to its runner, which waits for it on its stream, as the pose does: the
  segmentation (live in hd_trio; its crops are cut on their own stream and read without a wait **(seen)**) and the
  optical flow (`enable_prev_crop` off, so it runs nothing). Checked by the tests and an hd_trio check in the app.

Question: does the optical flow pair the current and the previous image of the same camera frame? To check when it
is used.

## The package held up to the rules (the apps' plan F, agreed 2026-11-17)

No code in the package is one app's: the face model is smile's alone, the segmentation hd_trio's, the optical flow,
`RecordedPoses` and `CaptureSync` deep_flow's, but each is general code with one user, which stays (as the camera's
backends do). The package has no rules file; nothing it keeps needs one beyond `CLAUDE.md` **(deduction)**.

## The rules it touches

- `CLAUDE.md` Performance, prefer the design with the lower latency, measured in the context of the app: A1
- `CLAUDE.md` Composition, fix a problem where it is made: B1, each predictor at the crops' hand-over
