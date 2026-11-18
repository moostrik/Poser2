# The inference package's plan

What is not built in `modules/inference`, apart from the threads the app refactor moves
([apps/APPS_PLAN.md](../../apps/APPS_PLAN.md)). What is built is in [INFERENCE.md](INFERENCE.md), how
the input's threads hand on in [INPUT_THREADS.md](../flow/INPUT_THREADS.md). **(deduction)** marks what is inferred.

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
| C1 no locks     | pose         | 25.00 / 39.34            | 245 (B1's)     | 0                  | 0              |
| C1 no locks     | segmentation |                          | 122 (B1's)     | 0                  | 0              |
| C1 no locks     | optical flow |                          | 142 (B1's)     | 0                  | 0              |
| E1 crop step    | pose         | 20.75 / 34.48            | 521 (B2's)     | 0                  | 0              |
| E1 crop step    | segmentation |                          | 982 equal crops| 1, B2's run's      | 1.0            |
| E1 crop step    | optical flow |                          | 952 equal pairs| 0                  | 0              |
| E2 two engines  | pose         | 20.84 / 34.66            | 547 (E1's)     | 0                  | 0              |
| E2 two engines  | segmentation |                          | 1000 eq. crops | 0                  | 0              |
| E2 two engines  | optical flow |                          | 951 equal pairs| 0                  | 0              |

C1 is compared with B1's run, which has the fix C1 keeps. With the probe's four models running together (pose,
segmentation, optical flow, person detector), the TensorRT models no longer take turns: in 30 s the pose ran 1441
batches against 675, the batches dropped fell from 1604 to 472 (pose), 1604 to 106 (segmentation), 1612 to 584
(flow), and the pose's mean and p99 from 20.9 / 51.2 to 15.7 / 38.4 ms, the segmentation's from 16.4 / 35.2 to
9.8 / 19.7 **(seen)**. deep_flow as it is runs one TensorRT model beside the detector, so its pose stays as it was.

A frame that differs is a whole frame, the rest exact. Before B1 the segmentation and the optical flow read their
crops without waiting for them; the pose, which waited, never differed. After B1 the flow is exact, and two runs after
it agree on every frame (26 masks, 35 flows): the segmentation's frames that still differ are e754ddee's own, read
before their crop was written **(deduction)**. The refactor (398ca410 to 4f72f124, built without a go) is judged by
these rows and your look at its diff. The optical flow pairs each frame with the camera's frame just before it: 673
of 674 flows, the other across the recording's loop **(seen)**.

D1, the color in one kernel, against C1 (the image differs from torch ops' by at most one float16 step):

| Step            | Model        | Mean / p99 as it is (ms) | Frames matched | Frames that differ | Max difference   |
|-----------------|--------------|--------------------------|----------------|--------------------|------------------|
| D1 one kernel   | pose         | 25.19 / 38.46            | 472            | 70                 | 0.003, one 0.648 |
| D1 one kernel   | segmentation |                          | 445            | (2 masks NaN)      | foreground 0.038 |
| D1 one kernel   | optical flow |                          | 414            |                    | 0.028 px         |

`scripts/gpu_input_timing.py` on a 992x800 recording, straightened and mirrored, against torch ops: NV12 23 GPU ops
to 2, 0.922 to 0.571 ms; BGR 8 to 2, 0.353 to 0.240 ms; the image's largest difference 0.000488 for both, an eighth
of 1/255 **(seen)**. In deep_flow as it is the upload's time (`GPU Image Upload`) went from 5.00 / 23.71 to 2.95 /
17.84 ms, its wait (`Wait upload`) from 3.33 / 23.15 to 2.02 / 17.72 **(seen)**. A pose point that differs moves one
SimCC bin (0.003); one frame's point moved 0.648, a keypoint whose two peaks were nearly equal **(deduction)**.

B2, the segmentation's output: D1's run read 2 masks of NaN with whole foregrounds **(seen)**. The runners made their
outputs on the default stream and wrote them on their own, and the segmentation handed on its foreground while its
stream still wrote it **(seen)**. Now each runner makes its outputs on its own stream, and `segmentation.Image`
carries a `ready` event its readers (the render's mask and foreground layers) wait for, marking the tensor used there
(`record_stream`), as the camera's and the crops' images are read; PyTorch: streams run "in any relative order" unless
synchronized; `Event.wait` orders a stream without blocking the CPU; `record_stream` keeps the cache from reusing a
tensor another stream still reads **(site fact:
[CUDA streams](https://docs.pytorch.org/docs/stable/notes/cuda.html#cuda-streams),
[record_stream](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.record_stream.html),
[Event](https://docs.pytorch.org/docs/stable/generated/torch.cuda.Event.html))**. Two 2-minute runs before and after
had no NaN mask, so the runs do not show the fix; it rests on the documentation. Two runs after it, pairing each
crop by the object: no crop changed between its hand-over and the model (0 of 2964), and the masks of the crops
equal in both runs were equal (469 of 469) **(seen)**. Two runs differ only where the probe cropped another box,
looked up by the tracklet's time rather than the image's **(deduction)**.

E1 (`PoseInference` gone; one crop step, every size in one pass, the pose model behind it in the pipeline's async
step; 2026-11-18) is compared with B2's run where both runs cut the same crop (each crop's fingerprint as handed to the
model): the one mask that differs is the odd one in B2's run, unlike its neighbours (0.024 against 0.002 mean
difference), while E1's matches its own (0.001 to 0.004) **(seen)**. deep_flow as it is: the pose 20.75 / 34.48 ms
against D1's 25.19 / 38.46, the upload 1.65 / 17.03 against 2.95 / 17.84, the hop to the frame source ("Wait pose
step") 0.40 / 2.51 against 0.28 / 0.97 **(seen, one run each)**; nothing in E1 explains a faster pose, so it is not
claimed **(deduction)**.

E2 (`OnnxEngine` and `TrtEngine` in the `engine` subpackage, each model's own steps in its worker, the runners gone;
2026-11-18) is compared with E1's run on the frames whose crops were byte-equal in both runs: every output equal
**(seen)**. A frame whose crops differ while its box is the same is one with two players: the probe keys a frame's
crop by its time, so it keeps one player's, not always the same one's **(deduction)**. deep_flow as it is: the pose
20.84 / 34.66 ms against E1's 20.75 / 34.48, the upload 1.59 / 16.83 against 1.65 / 17.03 **(seen, one run each)**.

`scripts/model_benchmark.py`, 300 batches after 50, each model alone on the GPU, before and after E2, its outputs
against the run before (`--reference`):

| Model                          | TensorRT before | TensorRT after | ONNX before   | ONNX after       | Outputs after          |
|--------------------------------|-----------------|----------------|---------------|------------------|------------------------|
| pose (hd_trio, batch 3)        | 3.64 / 3.97     | 3.52 / 3.97    | 3.26 / 3.94   | 3.34 / 4.15 (1)  | exact                  |
| segmentation (EXTREME, b. 3)   | 5.38 / 5.87     | 5.29 / 6.14    | 21.02 / 22.82 | 21.15 / 24.37    | exact (2)              |
| optical flow (deep_flow, b. 1) | 4.98 / 5.42     | 5.07 / 5.57    | 22.36 / 25.53 | 17.69 / 31.64    | exact                  |

(1) 5.05 / 5.85 in two other runs. (2) ONNX's foreground differs by design, its mask is exact.

Mean / p99 in ms **(seen)**. The pose on ONNX ran 3.3 or 5.1 ms from one run to the next on the same code (two runs
each), so a single ONNX row is no measure **(seen)**; cuDNN's exhaustive search choosing another algorithm per load
is the likely cause **(deduction)**. ONNX's segmentation hands on the crops times the mask as its foreground, as
TensorRT's did, not the model's own foreground output: its mask is exact, its foreground differs by design. ONNX
against TensorRT: the pose is as fast; the segmentation and the flow take four times as long on ONNX **(seen)**.

E3 (`ModelWorker`, the one base of `PoseEstimator`, `Segmenter`, `OpticalFlowEstimator`, `PersonDetector` and
`FaceDetector`: the thread, the newest item per key, the drops in the settings, the stream and the stop; 2026-11-18)
against E2's run, on the frames whose inputs were equal: the pose 512, the masks 914, the flows 875, every output
equal **(seen)**. The probe's run, all four models together: the pose 12.32 / 30.93 ms against 12.40 / 31.79, the
segmentation 14.63 / 36.49 against 15.03 / 39.17, the flow 18.87 / 43.53 against 19.14 / 45.63, and the batches
dropped 395, 239 and 540 against 454, 266 and 632 **(seen, one run each)**. The engine stays the worker's own, not the
base's: the face detector runs two.

## C. The TensorRT models side by side

Questions:
- CUDA graphs: capture each model's `execute_async_v3` per batch size and replay it, to cut the launch time on the
  CPU. Worth it, now the models run side by side?
- The pose's stream at a higher priority than the others', as the model on the critical path?

## F. The pose's decode on the GPU

The pose's thread copies RTMPose's whole SimCC output to the CPU, (N, 17, 384) and (N, 17, 512) at 192x256, and
decodes it there with numpy (`pose/decode.py`) **(seen)**. On the GPU the decode is two argmax and two max over those
outputs, and only 17 points and scores a player come back **(deduction)**.

- F1. A proof of concept, not committed: `decode_simcc` as torch ops on the GPU, against the numpy decode on recorded
  outputs (the same points and scores, exact: argmax and max on the same numbers), and the time from the model's end
  to the poses on the CPU, both ways, for 1 to 8 players, on the GPU otherwise idle and in deep_flow as it is
- F2. If F1 shows the same numbers and less time: the pose's decode on its stream before its one `synchronize()`,
  `pose/decode.py` one decode, the numpy one gone. Checked by the package's tests and the check's run (the outputs
  exact, the times)

## The package held up to the rules (the apps' plan F, agreed 2026-11-17)

No code in the package is one app's: the face model is smile's alone, the segmentation hd_trio's, the optical flow,
`RecordedPoses` and `CaptureSync` deep_flow's, but each is general code with one user, which stays (as the camera's
backends do). The package has no rules file; nothing it keeps needs one beyond `CLAUDE.md` **(deduction)**.

## The rules it touches

- `CLAUDE.md` Composition, prefer composition: E2 and E3 (built), a worker holds an engine and its model's steps
- `CLAUDE.md` Types, an ABC for a family of classes that share code: E3
- `CLAUDE.md` Packages, a package is one thing, and its name is that thing: E5 states the one thing
- `CLAUDE.md` Docstrings, reasons sit in the package's document: E5
- `CLAUDE.md` API evolution, remove obsolete pathways: E6
- `CLAUDE.md` Naming, the domain's term, one everywhere (class, setting, panel label): E3, the workers, their settings
  and timers; `glossary.md` follows: E4
- `CLAUDE.md` Performance, prefer the design with the lower latency, measured in the context of the app: F1, F2
