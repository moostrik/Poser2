# The inference's timing

What the models cost and how their outputs held through the package's refactor (plans A to I, November 2026). The
design is in [INFERENCE.md](INFERENCE.md); this document holds the numbers. **(deduction)** marks what is inferred
from them.

## The configuration

| What      | Here                                                                    |
|-----------|-------------------------------------------------------------------------|
| GPU       | GeForce RTX 4090, 24 GB, driver 621.24                                  |
| Libraries | torch 2.14.1 (CUDA 13.0), TensorRT 11.4.0.106, ONNX Runtime 1.31.1      |
| Recording | `20261112-125832_measure` (still, then an arm sideways, slow then fast) |
| App       | deep_flow `-sim`, its preset: the pose every other frame, the fluid on  |

## How the outputs are checked

On the recording in the app, a probe (`inference_probe.py`, not committed) adds the segmentation and its crop
(`reset_interval` 1, so no run's drops carry into its masks), the previous crop and the optical flow, and the pose on
every frame (`pose_crop.interval` 1: at 2, which half of the frames is posed differs per run **(seen)**). It keys every
output by the recording's frame, crops every frame with one baseline run's boxes (the tracker gives each run slightly
other boxes **(seen)**), and compares two runs on the frames whose inputs were byte-equal (each crop's fingerprint as
handed to the model). The same engine on the same input gives the same numbers, so a difference is a fault
**(deduction)**. A frame whose crops differ while its box is the same has two players: the probe keys a frame's crop by
its time **(deduction)**. An old commit runs from a copy outside the repository, the models linked in.

`scripts/model_benchmark.py` times each model alone on the GPU: 300 runs after 50, the crop models on seeded crops (or
`--image`), the detectors on one uploaded photo, its outputs against an earlier run (`--reference`), on TensorRT or
ONNX Runtime.

## The refactor's steps, in the app

Mean / p99 ms of the pose as deep_flow runs (its preset), and the outputs against the run before **(seen, one run a
step unless said)**:

| Step                     | Model        | Pose mean / p99 (ms) | Frames compared | Frames that differ | Max difference   |
|--------------------------|--------------|----------------------|-----------------|--------------------|------------------|
| A1 e754ddee x2           | pose         | 25.07 / 39.23        | 163             | 0                  | 0                |
| A1 e754ddee x2           | segmentation |                      | 39              | 0                  | 0                |
| A1 e754ddee x2           | optical flow |                      | 45              | 1                  | 2.05 px          |
| A1 refactor              | pose         | 25.17 / 39.73        | 168             | 0                  | 0                |
| A1 refactor              | segmentation |                      | 43              | 2                  | 1.0              |
| A1 refactor              | optical flow |                      | 51              | 5                  | 7.34 px          |
| B1 crops' event          | pose         | 25.17 / 39.27        | 144             | 0                  | 0                |
| B1 crops' event          | segmentation |                      | 44              | 2                  | 1.0              |
| B1 crops' event          | optical flow |                      | 56              | 0                  | 0                |
| C1 no locks              | pose         | 25.00 / 39.34        | 245 (B1's)      | 0                  | 0                |
| C1 no locks              | segmentation |                      | 122 (B1's)      | 0                  | 0                |
| C1 no locks              | optical flow |                      | 142 (B1's)      | 0                  | 0                |
| D1 the color in a kernel | pose         | 25.19 / 38.46        | 472             | 70                 | 0.003, one 0.648 |
| D1 the color in a kernel | segmentation |                      | 445             | (2 masks NaN)      | foreground 0.038 |
| D1 the color in a kernel | optical flow |                      | 414             |                    | 0.028 px         |
| E1 the crop step         | pose         | 20.75 / 34.48        | 521 (B2's)      | 0                  | 0                |
| E1 the crop step         | segmentation |                      | 982 equal crops | 1, B2's run's      | 1.0              |
| E1 the crop step         | optical flow |                      | 952 equal pairs | 0                  | 0                |
| E2 two engines           | pose         | 20.84 / 34.66        | 547 (E1's)      | 0                  | 0                |
| E2 two engines           | segmentation |                      | 1000 equal      | 0                  | 0                |
| E2 two engines           | optical flow |                      | 951 equal pairs | 0                  | 0                |
| E3 the worker base       | pose         | 12.32 / 30.93 (1)    | 512             | 0                  | 0                |
| E3 the worker base       | segmentation |                      | 914             | 0                  | 0                |
| E3 the worker base       | optical flow |                      | 875             | 0                  | 0                |
| G2 the inference frame   | pose         | 12.73 / 27.75 (1)    | 529             | 0                  | 0                |
| G2 the inference frame   | segmentation |                      | 916             | 0                  | 0                |
| G2 the inference frame   | optical flow |                      | 876             | 0                  | 0                |

(1) The probe's run, all four models together, not deep_flow as it is.

- Before B1 the segmentation and the optical flow read their crops without waiting for them; the pose, which waited,
  never differed: the frames that differ in A1 are crops read before they were written **(deduction)**
- D1: the color's conversion in one CUDA kernel differs from torch ops' by at most one float16 step; a pose point that
  differs moves one SimCC bin (0.003), one point moved 0.648, a keypoint whose two peaks were nearly equal
  **(deduction)**. Its two NaN masks were the segmentation handing on its foreground while its stream still wrote it,
  the reason for `segmentation.Image.ready` (B2) **(seen)**
- C1, the TensorRT models without a lock between them, all four models in the probe: in 30 s the pose ran 1441
  batches against 675, the drops fell from 1604 to 472 (pose), 1604 to 106 (segmentation), 1612 to 584 (flow), the
  pose's mean / p99 from 20.9 / 51.2 to 15.7 / 38.4 ms, the segmentation's from 16.4 / 35.2 to 9.8 / 19.7 **(seen)**
- E1's faster pose in deep_flow as it is (20.75 against D1's 25.19) has no cause in E1, so it is not claimed
  **(deduction)**
- G2: the view ran 2973 frames in 60 s **(seen)**

The upload, `scripts/gpu_input_timing.py` on a 992x800 recording against torch ops: NV12 from 23 GPU ops to 2, 0.922
to 0.571 ms; BGR from 8 to 2, 0.353 to 0.240 ms; the image's largest difference 0.000488, an eighth of 1/255. In
deep_flow as it is the upload (`GPU Image Upload`) went from 5.00 / 23.71 to 2.95 / 17.84 ms **(seen)**.

## Each model alone, ONNX Runtime and TensorRT

`scripts/model_benchmark.py`, 2026-11-18, E1 (916ff89a) against G4 (5650faf1), mean / p99 ms **(seen)**:

| Model                          | TensorRT E1 | TensorRT G4 | ONNX E1       | ONNX G4       | Outputs G4 against E1 |
|--------------------------------|-------------|-------------|---------------|---------------|-----------------------|
| pose (hd_trio, batch 3)        | 3.23 / 4.31 | 3.34 / 4.69 | 3.55 / 5.27   | 3.69 / 5.11   | exact                 |
| segmentation (EXTREME, b. 3)   | 5.60 / 6.75 | 5.61 / 6.88 | 22.56 / 24.33 | 22.69 / 24.28 | exact (1)             |
| optical flow (deep_flow, b. 1) | 5.39 / 6.25 | 5.61 / 6.70 | (2)           | 13.30 / 14.79 | exact                 |
| person (deep_flow, 640x512)    | 1.34 / 2.57 | 1.42 / 2.97 | (2)           | 2.89 / 4.15   | exact                 |
| face (smile, SCRFD + 106)      | 4.16 / 6.22 | 4.77 / 6.18 | 5.08 / 6.49   | 5.24 / 6.67   | exact                 |

(1) ONNX's foreground is the crops times the mask, as TensorRT's; the mask is exact. (2) At E1 the person detector's
ONNX runner refused the model's symbolic output size and loaded nothing, so that run stopped there.

- A run's spread is about a millisecond: a second face run gave 3.66 / 5.17 against 3.68 / 5.07 **(seen)**
- The pose on ONNX ran 3.3 or 5.1 ms from one run to the next on the same code **(seen)**; cuDNN's exhaustive search
  choosing another algorithm at each load is the likely cause **(deduction)**
- ONNX Runtime against TensorRT: the pose is as fast; the segmentation and the flow take two to four times as long
  **(seen)**, as TensorRT fuses their many small steps **(deduction)**

## The segmentation every frame

RVM on one player, TensorRT FP16, at each size, mean / p50 / p99 ms, 2026-11-19, af0063c2 **(seen)**. Alone:
`scripts/model_benchmark.py --app superficial_flow --models segmentation`, 1000 runs after 100, the crop at the
model's size. In the app: low_flow `-sim` on `20261112-125832_measure`, 20 s, the segmentation on every frame
(1000 of 1000), the pose (256x192) on every other, the view and the color on, its `Segmentation` timer.

| Size      | Alone              | In the app          |
|-----------|--------------------|---------------------|
| 256x192   | 1.55 / 1.40 / 2.97 |                     |
| 384x288   | 1.62 / 1.43 / 2.87 |                     |
| 512x384   | 1.64 / 1.50 / 2.79 |                     |
| 1024x768  | 2.75 / 2.37 / 4.38 | 7.96 / 7.83 / 12.73 |

In the app the timer holds the wait for its crops and for the GPU the pose and the render share **(deduction)**.

## The pose models

Idle GPU, TensorRT, 300 crops of three photos, each model at its input size (`h2_fp16.py`, `h3_models.py`, not
committed), 2026-11-18 **(seen)**. A bin is half a pixel of the model's input.

| Engine                     | Mean / p99 ms | Visible points within one bin of its FP32 | Points against l 384x288 FP32 (median / 95%, crop width) |
|----------------------------|---------------|-------------------------------------------|----------------------------------------------------------|
| l 256x192 FP32, batch 1    | 2.89 / 3.96   | the reference                             | 0.0061 / 0.27                                            |
| l 256x192 FP16, batch 1    | 1.34 / 2.43   | 97.0% (99% within 3, max 160)             | 0.0061 / 0.27                                            |
| l 256x192 FP16, every node | 1.43 / 2.67   | 98.4%                                     |                                                          |
| l 256x192 FP32, batch 3    | 3.90 / 7.07   | the reference                             |                                                          |
| l 256x192 FP16, batch 3    | 1.79 / 5.16   | 97.4%                                     |                                                          |
| l 384x288 FP32, batch 1    | 3.03 / 4.08   | the reference                             | the reference                                            |
| l 384x288 FP16, batch 1    | 1.45 / 2.57   | 88% (of its finer bins)                   | 0 / 0.0026                                               |
| x 384x288 FP32, batch 1    | 4.31 / 5.55   | the reference                             |                                                          |
| x 384x288 FP16, batch 1    | 1.68 / 2.77   | 95%                                       |                                                          |

- FP16 takes half the time; the points that move more than a bin are mostly one jumping to another peak of its
  distribution **(seen)**. The engine with its norms in FP16 too does not collapse in TensorRT **(seen)**, as its
  reductions accumulate in FP32 **(deduction)**
- At one player the larger input costs 5% more: one crop does not fill the GPU **(deduction)**
- FP16 on ONNX Runtime, hd_trio's batch 3: 4.04 / 7.06 ms **(seen)**

In deep_flow as it is, the fluid on the GPU, 60 s a model, one run each, 2026-11-18 (the probe's `--as-is` run at
61d1ab9c) **(seen)**:

| Pose model     | COCO AP | Pose mean / p99 ms | Posed / dropped | Render interval mean / p99 ms |
|----------------|---------|--------------------|-----------------|-------------------------------|
| l 256x192 FP32 | 76.7    | 21.85 / 39.00      | 1497 / 0        | 12.93 / 39.73                 |
| l 256x192 FP16 | 76.7    | 20.56 / 36.15      | 1496 / 0        | 12.39 / 38.85                 |
| l 384x288 FP32 | 78.3    | 21.57 / 38.13      | 1496 / 0        | 13.14 / 40.12                 |
| l 384x288 FP16 | 78.3    | 20.14 / 35.95      | 1497 / 0        | 12.39 / 39.57                 |
| x 384x288 FP32 | 78.8    | 23.26 / 40.44      | 1498 / 0        | 13.58 / 40.27                 |
| x 384x288 FP16 | 78.8    | 20.76 / 38.08      | 1497 / 0        | 12.45 / 39.44                 |

Under the fluid the pose takes 20 to 23 ms whichever model: most of it is its wait for the GPU, not the model's own
1.3 to 4.3 ms **(deduction)**. At 59003bd2, HIGH in FP16: 21.0 / 43.5 ms, 1498 posed, none dropped **(seen)**.

## The pose's resolution for several players

hd_trio's and white_space's pose at `STANDARD` (192x256) and `HIGH` (288x384), TensorRT FP16, 2026-11-20, 20dd751c
**(seen)**. Alone: each app's own engine (hd_trio built for batch 3, white_space for 8), 300 batches after 50, seeded
crops at the model's size, idle GPU (`c2_pose_batches.py`, not committed, on `scripts/model_benchmark.py`), its
`Pose Estimation` timer, mean / p99 ms:

| Batch | hd_trio STANDARD | hd_trio HIGH | white_space STANDARD | white_space HIGH |
|-------|------------------|--------------|----------------------|------------------|
| 1     | 1.58 / 2.57      | 1.86 / 2.96  | 1.74 / 2.89          | 1.78 / 2.89      |
| 2     | 1.62 / 2.62      | 1.96 / 3.11  | 1.72 / 2.91          | 1.83 / 2.89      |
| 3     | 1.70 / 2.75      | 2.00 / 2.94  | 1.77 / 2.78          | 1.96 / 2.90      |
| 4     |                  |              | 1.81 / 2.87          | 2.06 / 2.96      |
| 5     |                  |              | 1.90 / 3.11          | 2.15 / 3.04      |
| 6     |                  |              | 1.91 / 2.82          | 2.20 / 3.15      |
| 7     |                  |              | 1.94 / 2.71          | 2.29 / 3.15      |
| 8     |                  |              | 2.02 / 2.84          | 2.39 / 3.17      |

In the app: each app `-sim`, its preset with the resolution set, 60 s after the first pose and 3 s more
(`c2_app_probe.py`, not committed); hd_trio on `intro_1` (two players a set), white_space on
`20260924-161443_3_person_long` (two or three); no recording holds more than three. Mean / p95 / p99 ms:

| What                   | hd_trio STANDARD     | hd_trio HIGH         | white_space STANDARD | white_space HIGH     |
|------------------------|----------------------|----------------------|----------------------|----------------------|
| Pose                   | 9.41 / 13.42 / 16.00 | 9.57 / 14.51 / 16.23 | 9.22 / 16.92 / 29.51 | 9.70 / 18.71 / 27.44 |
| Batches by size        | 2: 1373, 1: 12       | 2: 1372, 1: 14       | 3: 605, 2: 757       | 3: 606, 2: 757       |
| Dropped                | 2                    | 0                    | 0                    | 0                    |
| Render interval (mean) | 21.5                 | 21.3                 | 12.5                 | 13.3                 |

- Alone the model barely grows with the batch: most of its time is fixed cost; `HIGH` adds 0.04 to 0.37 ms
  (white_space, one to eight players) **(deduction from the table)**
- In the app `HIGH` adds 0.15 ms to hd_trio's pose and 0.5 ms (1.8 at p95) to white_space's, at two or three
  players; the pose's in-app time is mostly its wait for the GPU it shares with the render **(deduction)**
- One run each: white_space's render interval, 0.8 ms longer at `HIGH`, may be the run's spread **(deduction)**

## The pose's decode

The SimCC decode on the GPU (two max with their indices, a minimum, 17 points and scores a player back) against the
numpy decode on the CPU, the real model's outputs, 330 batches a size: the same points and scores in every batch
**(seen)**. From the model's end to the poses on the CPU, mean / p99 ms, idle GPU, 2026-11-18 **(seen)**:

| Players | numpy, TensorRT | GPU, TensorRT | numpy, ONNX   | GPU, ONNX     |
|---------|-----------------|---------------|---------------|---------------|
| 1       | 0.124 / 0.190   | 0.226 / 0.393 | 0.136 / 0.226 | 0.240 / 0.503 |
| 2       | 0.146 / 0.217   | 0.232 / 0.411 | 0.152 / 0.233 | 0.241 / 0.458 |
| 3       | 0.164 / 0.211   | 0.242 / 0.456 | 0.168 / 0.232 | 0.251 / 0.487 |
| 4       |                 |               | 0.178 / 0.248 | 0.232 / 0.421 |
| 8       |                 |               | 0.241 / 0.388 | 0.234 / 0.668 |

The GPU decode is slower up to 7 players and as fast at 8, its p99 worse at every size: copying both outputs costs
less than its kernel launches and its own copy **(deduction)**. The decode stays on the CPU.

## The results handed on finished

Each app `-sim` on `20261112-125832_measure`, 60 s after 3 s of play, its preset, before (0d842a69) and after
(4af51883) the producers hand on finished items to the readers outside the package, 2026-11-19, mean / p95 ms
(`c_probe.py`, not committed) **(seen)**. One run each.

| What                          | low_flow before | low_flow after | deep_flow before | deep_flow after |
|-------------------------------|-----------------|----------------|------------------|-----------------|
| GPU view                      | 2.81 / 5.86     | 2.87 / 5.92    | 0.19 / 0.19      | 0.19 / 0.19     |
| Capture latency               | 48.05 / 57.64   | 47.93 / 57.02  | 45.36 / 59.52    | 46.06 / 59.76   |
| Pose latency RAW              | 47.15 / 55.29   | 47.30 / 55.31  | 71.70 / 91.93    | 69.07 / 80.79   |
| Pose latency SMOOTH           | 49.95 / 58.43   | 50.05 / 58.36  | 74.99 / 96.42    | 72.22 / 84.55   |
| Render update                 | 4.67 / 8.36     | 4.66 / 8.36    | 9.76 / 28.25     | 8.96 / 23.32    |
| Wait pose step                | 0.24 / 0.43     | 0.04 / 0.18    | 0.55 / 1.37      | 0.04 / 0.11     |
| Wait upload (a frame queued)  | 0.50 / 0.91     | 0.48 / 0.88    | 2.33 / 11.95     | 2.86 / 13.12    |
| Wait upload finished          |                 | 0.48 / 0.90    |                  | 0.56 / 1.37     |
| Masks unsynced (share)        | 0.09            | 0.05           |                  |                 |

- low_flow's GPU view stays at 2.9 ms with the masks finished: the mask's wait is not what it costs **(seen)**
- The capture latency grows by at most the uploader's new wait, as expected **(seen)**; a frame waits about 0.5 ms
  longer in the uploader's queue, behind that wait **(seen)**
- The frame source's queue wait falls to 0.04 ms now the bang carries only capture times **(seen)**; why it was
  0.24 to 0.55 ms before is not known
- deep_flow's pose latency and render update fall at p95 by 11 and 5 ms; whether that is the change or a run's
  spread is not known from one run **(deduction)**

