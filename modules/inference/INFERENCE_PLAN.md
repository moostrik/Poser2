# The inference package's plan

What is not built in `modules/inference`, apart from the threads the app refactor moves
([apps/APPS_PLAN.md](../../apps/APPS_PLAN.md)). How the input works is in
[INPUT_THREADS.md](../flow/INPUT_THREADS.md). **(deduction)** marks what is inferred.

## The check

The models' outputs are checked on a known good video in the app: deep_flow `-sim` on `20261112-125832_measure`
(still, then an arm sideways, slow then fast), 30 s a run, two runs a step, only while you are away and the GPU runs
nothing else.

- The times: deep_flow as it is (its preset: the pose every other frame, the person detector; no segmentation or
  optical flow)
- The outputs: a probe, not committed, adds the segmentation and its crop (`reset_interval` 1, so no run's drops
  carry into its masks), the previous crop and the optical flow, and the pose on every frame (`pose_interval` 1: at
  2, which half of the frames is posed differs per run **(seen)**). It keys every output by the recording's frame
  (a recording without metadata is stamped on the play's clock, other each run **(seen)**), and crops every frame
  with the first baseline run's box (the detector's tracker gives each run slightly other boxes **(seen)**). Two runs
  are compared on the frames both have, a flow only when both paired the same two frames. It stops the app in a
  `finally`, and reads both settings names (before and after the refactor's)
- e754ddee runs from a copy of that commit outside the repository, with the models and recordings linked in, so the
  working folder the other threads use stays as it is

The same engine on the same input gives the same numbers, so any difference is a fault **(deduction)**. A difference
is the largest over the frames compared, against e754ddee's first run.

| Step            | Model        | Mean / p99 as it is (ms) | Frames matched | Frames that differ | Max difference |
|-----------------|--------------|--------------------------|----------------|--------------------|----------------|
| A1 e754ddee ×2  | pose         | 25.07 / 39.23            | 163            | 0                  | 0              |
| A1 e754ddee ×2  | segmentation |                          | 39             | 0                  | 0              |
| A1 e754ddee ×2  | optical flow |                          | 45             | 1                  | 2.05 px        |
| A1 refactor     | pose         | 25.17 / 39.73            | 168            | 0                  | 0              |
| A1 refactor     | segmentation |                          | 43             | 2                  | 1.0            |
| A1 refactor     | optical flow |                          | 51             | 5                  | 7.34 px        |
| B1 crops' event | pose         | 25.17 / 39.27            | 144            | 0                  | 0              |
| B1 crops' event | segmentation |                          | 44             | 2                  | 1.0            |
| B1 crops' event | optical flow |                          | 56             | 0                  | 0              |

A frame that differs is a whole frame, the rest exact. Before B1 the segmentation and the optical flow read their
crops without waiting for them; the pose, which waited, never differed. After B1 the flow is exact, and two runs after
it agree on every frame (26 masks, 35 flows): the segmentation's frames that still differ are e754ddee's own, read
before their crop was written **(deduction)**. The refactor (398ca410 to 4f72f124, built without a go) is judged by
these rows and your look at its diff. The optical flow pairs each frame with the camera's frame just before it: 673
of 674 flows, the other across the recording's loop **(seen)**.

## C. The TensorRT models side by side

Every TensorRT runner (pose, segmentation, optical flow, person detector, face) runs `execute_async_v3` and its
stream's `synchronize()` under one global lock (`trt_runtime.get_exec_lock`), and loads under another
(`get_init_lock`) **(seen)**: the GPU runs one TensorRT model at a time, start to end. TensorRT 11's documentation
(we run 11.4) has each thread run its own context on its own stream, in parallel, with no lock; deserializing and
creating contexts are thread-safe **(site fact:
[Threading](https://docs.nvidia.com/deeplearning/tensorrt/latest/architecture/how-trt-works.html),
[multi-streaming](https://docs.nvidia.com/deeplearning/tensorrt/latest/performance/optimization.html))**. The locks'
reason, "Myelin graph conflicts", is in no NVIDIA page; likely an older TensorRT's **(deduction)**. Without the exec
lock a model starts while another runs: less wait for the pose, each model's own time likely longer **(deduction)**.

- C1. Both locks removed from all five runners. Checked by the package's tests and the check's two runs: the
  outputs at the tolerance, each model's time and lock wait in the table

Questions:
- CUDA graphs: capture each model's `execute_async_v3` per batch size and replay it, to cut the launch time on the
  CPU. Worth it once C1 shows how much of a model's time is launch?
- The pose's stream at a higher priority than the others', as the model on the critical path?

## D. The color in one kernel

- D1. The uploader's color conversion, straightening and mirroring as one CUDA kernel, as the depth's alignment is
  (`depth_align.py`), for both of `ImageUpload`'s inputs: the OAK4's NV12 and the depthai 2 cameras' BGR
  (white_space, hd_trio). From NV12 23 GPU ops become 2, about 0.4 ms less GPU a frame **(deduction)**. The kernel
  rounds otherwise, so its image is held to at most 1/255 per pixel from the old one, one step of the camera's 8 bits,
  not to the models' exactness. Checked by the check's two runs for NV12 (deep_flow's `-sim` decodes to NV12 **(seen)**), and one run of
  `scripts/gpu_input_timing.py` before and after for BGR's time per step and image difference; the camera's time at
  the next live session.

## The package held up to the rules (the apps' plan F, agreed 2026-11-17)

No code in the package is one app's: the face model is smile's alone, the segmentation hd_trio's, the optical flow,
`RecordedPoses` and `CaptureSync` deep_flow's, but each is general code with one user, which stays (as the camera's
backends do). The package has no rules file; nothing it keeps needs one beyond `CLAUDE.md` **(deduction)**.

## The rules it touches

- `CLAUDE.md` Running code, run a GPU harness only on the user's go, and only what the plan names: A1, B1, C1, D1
- `CLAUDE.md` Composition, fix a problem where it is made: B1, each predictor at the crops' hand-over
- `CLAUDE.md` Performance, prefer the design with the lower latency, measured in the context of the app: C1, D1
- `CLAUDE.md` API evolution, remove obsolete pathways: C1, the locks
