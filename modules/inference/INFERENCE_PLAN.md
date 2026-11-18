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

F1, the pose's SimCC decode on the GPU (two max with their indices, a minimum, 17 points and scores a player back to
the CPU), against the numpy decode on the real model's outputs (crops of three photos and noise, 330 batches a size):
the same points and scores in every batch, on TensorRT and ONNX **(seen)**. From the model's end to the poses on the
CPU, mean / p99 ms, the GPU otherwise idle (`f1_decode.py`, not committed, 2026-11-18):

| Players | numpy, TensorRT (hd_trio) | GPU, TensorRT (hd_trio) | numpy, ONNX   | GPU, ONNX     |
|---------|---------------------------|-------------------------|---------------|---------------|
| 1       | 0.124 / 0.190             | 0.226 / 0.393           | 0.136 / 0.226 | 0.240 / 0.503 |
| 2       | 0.146 / 0.217             | 0.232 / 0.411           | 0.152 / 0.233 | 0.241 / 0.458 |
| 3       | 0.164 / 0.211             | 0.242 / 0.456           | 0.168 / 0.232 | 0.251 / 0.487 |
| 4       |                           |                         | 0.178 / 0.248 | 0.232 / 0.421 |
| 8       |                           |                         | 0.241 / 0.388 | 0.234 / 0.668 |

The GPU way is slower up to 7 players and as fast at 8, with a worse p99 at every size **(seen)**: copying both SimCC
outputs costs less than the decode's kernel launches and its own copy **(deduction)**. So F2 is not built: the decode
stays on the CPU. On a busy GPU the decode's kernels would also queue behind the other models' work **(deduction)**.
deep_flow's TensorRT engine takes one player only (a static batch of 1), so its row is hd_trio's engine.

## G. The package's root and the inference frame (2026-11-18, built)

G1 (2026-11-18) sorted the root: `preprocessing/` (the upload and the crop step), `model_worker` and the TensorRT
stub into `engine/`, `recorded_poses` into `pose/`, `detection/` named `person/`; the capture sync stays at the root.
G2 (2026-11-18) made the capture an `InferenceFrame`, its results keyed by type as the pose `Frame`'s features, and
`InferenceFrameSync` made with the types its app runs; against E3's run on the frames whose inputs were equal, every
output equal (the pose 529, the masks 916, the flows 876), and the view ran 2973 frames in 60 s **(seen)**. G3:
`INFERENCE.md` names the frame, its result types, `preprocessing` and `person`; the glossary takes none.
G4: pyright on the package, 30 errors to none: annotations (the staging events, the TensorRT stub's `nptype`, the
straightening's two values), narrowings (the tests' image and frame), the tests' fakes as the types they stand in
for, and one `cast`: a worker's settings read as their fields' values (`ModelWorkerSettings`), which pyright's
protocols cannot see through a `Field` descriptor **(seen)**. The scripts' own errors outside the package stay.

G5, the refactor's benchmark (`scripts/model_benchmark.py`, now with the person and face detectors on one uploaded
photo, `face.jpg`), 2026-11-18, 300 runs after 50, each model alone on the GPU: E1 (916ff89a, a copy outside the
repository) against G4 (5650faf1), mean / p99 ms, the outputs against E1's **(seen)**:

| Model                          | TensorRT E1   | TensorRT G4   | ONNX E1       | ONNX G4       | Outputs G4 against E1  |
|--------------------------------|---------------|---------------|---------------|---------------|------------------------|
| pose (hd_trio, batch 3)        | 3.23 / 4.31   | 3.34 / 4.69   | 3.55 / 5.27   | 3.69 / 5.11   | exact                  |
| segmentation (EXTREME, b. 3)   | 5.60 / 6.75   | 5.61 / 6.88   | 22.56 / 24.33 | 22.69 / 24.28 | exact (1)              |
| optical flow (deep_flow, b. 1) | 5.39 / 6.25   | 5.61 / 6.70   | (2)           | 13.30 / 14.79 | exact                  |
| person (deep_flow, 640x512)    | 1.34 / 2.57   | 1.42 / 2.97   | (2)           | 2.89 / 4.15   | exact                  |
| face (smile, SCRFD + 106)      | 4.16 / 6.22   | 4.77 / 6.18   | 5.08 / 6.49   | 5.24 / 6.67   | exact                  |

(1) ONNX's foreground is the crops times the mask since E2, as TensorRT's; the mask is exact. (2) At E1 the person
detector's ONNX runner refused the model's symbolic output size (`['batch', 84, 'anchors']`) and loaded nothing, so
the ONNX run stopped there; the engine since E2 takes it **(seen)**. A second face run on TensorRT gave 3.66 / 5.17
(G4) against 3.68 / 5.07 (E1): one run is no measure of a 0.6 ms difference **(seen)**. The refactor leaves every
model's output as it was, and its time within a run's spread.

## H. A better pose model, after G (2026-11-18)

The pose model costs 2.8 ms for one player on an idle GPU **(seen)**, so there is room for a better one. The research
([`reports/Higher quality pose models.md`](../../reports/Higher%20quality%20pose%20models.md), online sources) ranks
staying in the RTMPose family at 384x288 first: RTMPose-l 384x288 trained on Body8 reaches 78.3 COCO AP against 75.8
to 76.7 for ours **(site fact)**, at about 1.75x the time on an idle GPU, about 4.9 ms **(deduction: MMPose's ratio on
another GPU)**. ViTPose costs several times as much for a like gain, the one-stage models (RTMO, YOLO-pose) score
below ours, Sapiens is too heavy and non-commercial, and the video models need future frames (33 to 67 ms more)
**(site fact)**. Only `rtmpose-l_256x192.onnx` is in `data/models` **(seen)**: every other model file is a download,
asked for with its name, source and size before it is fetched (the five RTMPose ONNX files from OpenMMLab's
download server agreed 2026-11-18, each logged with its name and size).

- H1. Which checkpoint ours is: `rtmpose-l_256x192.onnx` (110.7 MB, May 2026) has no source in the repository
  **(seen)**; its outputs against the official COCO-only, AIC+COCO and Body8 ONNX files on the same crops tell. It
  sets the gain of every step after it
- H2. An FP16 engine of our model, with the head's ScaleNorm kept FP32 (first, decided 2026-11-18: it changes only
  the precision, and the time it saves is the room H3 needs). A public report has the norm's sum of squares overflow
  in FP16 **(site fact)**, likely the precision RTMPose lost here in FP16 **(deduction)**. Under TensorRT's strongly
  typed build the ONNX file sets the precision **(seen: `export_rtm_onnx_to_trt.py`)**, so FP16 is a conversion of
  the file, `tools/models/convert_onnx_to_fp16.py` **(seen)** with the norm's nodes kept FP32. Measured against the
  FP32 engine's points on the same crops (the difference in SimCC bins and in score) and in time; no fixed threshold:
  FP16 is a balance of speed and quality, weighed in H5 (decided 2026-11-18). The models after it reuse the
  conversion
- H3. RTMPose-l 384x288, Body8: the official ONNX file, its engines in `scripts/makemodels.bat` per app (deep_flow's
  at a static batch of 1 **(seen)**), the pose's `HIGH` resolution in the presets; the crop follows the model's size
  (each app sets it from the pose settings **(seen: `apps/deep_flow/main.py`)**), the SimCC bins follow the output,
  the decode stays **(seen: it reads the output's size)**. The pose settings' `ULTRA` and `EXTREME` load the
  384x288 file at a 384x512 and 768x1024 input **(seen)**, a size the engine refuses **(deduction)**: those fallbacks
  go, and a pose resolution without its own file is refused when the settings are initialized
- H4. RTMPose-x 384x288 (78.8 AP, about 8 ms **(deduction: a FLOP ratio)**), only if H2 and H3 leave room
- H5. The choice in the app: ours, H2, H3 and H4 in deep_flow as it is (`20261112-125832_measure`, -sim, the probe's
  as-is run, 60 s each, about 1.5 minutes a model), with the fluid on the GPU, where sharing the GPU sets most of
  the pose's 20 ms **(seen)**. It measures the pose's time (mean, p99) and its dropped batches; the table names its
  kind (a time), date, commit, recording and settings. With G5's benchmark. Each candidate, FP32 and FP16, with its
  AP, its points' difference and its time side by side; the choice is yours, and its preset is yours to save (a
  preset is written only by Save)

Questions:
- The licence of the Body8 training data (AI Challenger and others) is unresolved, for ours as for every candidate
- The SimCC distribution's width as each keypoint's uncertainty for the One Euro filter: cheap, untested

## C. The TensorRT models side by side

Questions:
- CUDA graphs: capture each model's `execute_async_v3` per batch size and replay it, to cut the launch time on the
  CPU. Worth it, now the models run side by side?
- The pose's stream at a higher priority than the others', as the model on the critical path?

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
  and timers
- `CLAUDE.md` Packages, a package is one thing; its files each hold one subject: G1
- `CLAUDE.md` Naming, one term everywhere, rename it everywhere in the same change: G2
- `CLAUDE.md` Composition, fixed structure through constructors: G2, the sync's result types
- `pose-data.md`, a read never raises, missing data reads empty: G2 takes the pose frame's contract
- `CLAUDE.md` Types, fix a type error in this order, suppress only as a last resort: G4
- `CLAUDE.md` Performance, measured in the context of the app: H5 chooses the model under the fluid
- `CLAUDE.md` Running code, a run on the user's go, said what it measures and how long: H5
- `flow.md` Flow measurements, a table names its kind, date, commit, recording and settings: H5
- `CLAUDE.md` Performance, prefer the design with the lower latency, measured in the context of the app: F1, the decode kept on the CPU
