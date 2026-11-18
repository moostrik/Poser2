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
| C1 no locks     | pose         | 25.00 / 39.34            | 245 (B1's)     | 0                  | 0              |
| C1 no locks     | segmentation |                          | 122 (B1's)     | 0                  | 0              |
| C1 no locks     | optical flow |                          | 142 (B1's)     | 0                  | 0              |

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

## C. The TensorRT models side by side

Questions:
- CUDA graphs: capture each model's `execute_async_v3` per batch size and replay it, to cut the launch time on the
  CPU. Worth it, now the models run side by side?
- The pose's stream at a higher priority than the others', as the model on the critical path?

## E. The package held up again (2026-11-18)

- E1. One base for the five model workers: `pose.Predictor`, `segmentation.Predictor`, `optical_flow.Predictor`,
  `PersonDetector` and `FaceDetector` each set up their model on their own thread, wake on work, take the newest
  waiting (a batch, or an image per camera), run it, hand it on, catch per item and stop **(seen)**, written two ways:
  three hold a thread and count drops in their settings, two are a `Thread` and count them in a property. The base
  holds the thread, the wake, the newest-wins queue (by batch or by camera), the drop count in the settings, the setup
  and the stop; each keeps its model's step. The workers take their task's name, as textbook vision names them:
  `PoseEstimator`, `Segmenter`, `FlowEstimator`, `PersonDetector`, `FaceDetector`, and the base `ModelWorker`; the
  glossary's wording goes to the rules thread. The uploaders, `CameraSync`, `CaptureSync` and `RecordedPoses` keep
  their own queues (in order, sets, parts) **(deduction: they share less)**. Checked by the package's tests (the fake
  runners) and the check's run
- E3. `INFERENCE.md`: the package is one thing, the camera frames on the GPU and the models on them; the upload and
  the models stay one package, as every app uses them together and the crops and detectors take the upload's images
  **(seen)**. It says what the package's parts are and why (each model's thread, the crops' and outputs' events, the
  streams, no TensorRT lock), linking the plan's measurements, and takes in what `modules/flow/INPUT_THREADS.md`
  says of this package. `CameraSync` stays here: it groups one moment's frames of several cameras for the upload
  (`BatchUploader.submit_set`) **(seen)**, the input's first step, and left the camera package as not a camera
  (b5ab4515)
- E4. The leftovers: `Resolution` and `RESOLUTION_DIMS` out of the package's exports (no user outside **(seen)**); the
  empty `tools/` folder

## The package held up to the rules (the apps' plan F, agreed 2026-11-17)

No code in the package is one app's: the face model is smile's alone, the segmentation hd_trio's, the optical flow,
`RecordedPoses` and `CaptureSync` deep_flow's, but each is general code with one user, which stays (as the camera's
backends do). The package has no rules file; nothing it keeps needs one beyond `CLAUDE.md` **(deduction)**.

## The rules it touches

- `CLAUDE.md` Types, an ABC for a family of classes that share code: E1
- `CLAUDE.md` Packages, a package is one thing, and its name is that thing: E3 states the one thing
- `CLAUDE.md` Docstrings, reasons sit in the package's document: E3
- `CLAUDE.md` API evolution, remove obsolete pathways: E4
- `CLAUDE.md` Naming, the domain's term, one everywhere: E1, the workers' names; `glossary.md` follows
