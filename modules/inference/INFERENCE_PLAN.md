# The inference package's plan

What is not built in `modules/inference`, apart from the threads the app refactor moves
([apps/APPS_PLAN.md](../../apps/APPS_PLAN.md)). How the input works is in
[INPUT_THREADS.md](../flow/INPUT_THREADS.md). **(deduction)** marks what is inferred.

## The check

The models' outputs are checked on a known good video in the app, with the fewest runs that show a step: deep_flow
`-sim` on `20261112-125832_measure` (still, then an arm sideways, slow then fast), the fluid off. A probe, not
committed, adds what deep_flow lacks for the run alone: the segmentation and its crop, and `enable_prev_crop` on so
the optical flow runs. It saves every model's outputs (points, scores, masks, foregrounds, flows) and the pose's box by
the frame's capture time, and the models' timers, from the one run, and reads both settings names (before and after
the refactor's). `-sim` drops other frames each run, so two runs are compared on the frames both have with the same
box (the boxes come from the detector and its tracker, which follow the frames that ran **(deduction)**), and the
count is given; a flow is compared only when both runs paired the same two frames, by the previous frame's capture
time the probe keeps per camera (the snapshot carries only the current one's **(seen)**). The probe stops the app in a
`finally`. The segmentation's `reset_interval` is 1 in the probe: its recurrent state would carry the run's own
drops into every mask. The same engine on the same input gives the same numbers, so a difference beyond two baseline
runs' own is a fault **(deduction)**. e754ddee runs from a copy of that commit outside the repository, with
`apps/deep_flow/data` linked into it, so the working folder the other threads use stays as it is. Each run lasts the
recording, only while you are away and the GPU runs nothing else: the table's times hold only then.

Measured 2026-11-18, 30 s of the recording a run. Times from deep_flow as it is (its preset: the pose every other
frame, the person detector; no segmentation or optical flow); outputs from the probe's run. A difference is the
largest over the frames both runs have with the same box (and frame pair), against e754ddee's first run.

| Step            | Model        | Mean / p99 as it is (ms) | Frames matched | Frames that differ | Max difference |
|-----------------|--------------|--------------------------|----------------|--------------------|----------------|
| A1 e754ddee ×2  | pose         | 25.07 / 39.23            | 163            | 0                  | 0              |
| A1 e754ddee ×2  | segmentation |                          | 39             | 0                  | 0              |
| A1 e754ddee ×2  | optical flow |                          | 45             | 1                  | 2.05 px        |
| A1 refactor     | pose         | 25.17 / 39.73            | 168            | 0                  | 0              |
| A1 refactor     | segmentation |                          | 43             | 2                  | 1.0            |
| A1 refactor     | optical flow |                          | 51             | 5                  | 7.34 px        |

The frames that differ are whole frames, the rest exact, in the two models that read their crops without waiting
for them (B1); the pose, which waits, never differs **(deduction: the race B1 removes)**.

## A. The baseline

- A1. The refactor confirmed: the probe at e754ddee (before it) twice, which gives the noise and the tolerance (at
  least 1e-4), and on the current code once against it; the numbers in the table. The refactor (398ca410 to
  4f72f124, one commit per model and one for the small breaks) was built without a go: this comparison and your look
  at its diff judge it, and a model that differs is fixed or its commit reverted

## B. The crops' event

- B1. Every predictor hands its crops' event to its runner, which waits for it on its stream, as the pose does: the
  segmentation (live in hd_trio; its crops are cut on their own stream and read without a wait **(seen)**) and the
  optical flow (`enable_prev_crop` off, so it runs nothing). Checked by the tests and one probe run against A1.

Question: does the optical flow pair each frame with the camera's frame just before it? A1's runs show it, by the
capture times the probe keeps.

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

- C1. Both locks removed from all five runners. Checked by the package's tests and one probe run against A1: the
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
  not to A1's noise. Checked by one probe run against A1 for NV12 (deep_flow's `-sim` decodes to NV12 **(seen)**), and one run of
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
