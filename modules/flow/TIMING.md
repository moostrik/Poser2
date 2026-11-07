# The flow's timing

The GPU time of every stage of the 3D flow in deep_flow, pass by pass, and the defaults it sets: the
fluid's grid and the particles' count. The design of each stage is in [FLOW.md](FLOW.md) and
[FLUID_3D.md](FLUID_3D.md); this document holds their cost. **(deduction)** marks what is inferred
from the numbers.

## The configuration

| What              | Here                                                                                       |
|-------------------|--------------------------------------------------------------------------------------------|
| GPU               | GeForce RTX 4090, 24 GB                                                                    |
| Target            | 60 fps, 16.7 ms a frame                                                                    |
| App               | deep_flow, `launcher.py -app deep_flow -sim -ni -timing 60 -warmup 10`                     |
| Input             | the standmoves recording (`-sim`), one camera, 50 fps                                      |
| Preset            | `apps/deep_flow/data/settings/settings.json`, one setting changed per run                  |
| Pose              | on: the person detection and RTM pose on TensorRT, on the same GPU                         |
| Segmentation      | off: `mask_source` `DEPTH`, the RVM model not run                                          |
| Output            | the large layers at 2 × the output, 2160 × 3840; no upscale pass beyond that allocation    |
| `v_sync`          | off, so the frame rate shows the headroom                                                  |
| Warm-up, duration | 10 s running before the first sample, then 60 s                                           |

Every number is a mean and a p95 in ms over the run. A GPU time is a `GpuTimer` (`modules/gl/gpu_timer.py`),
two GL timestamps around a block, so blocks nest: `GPU fluid` holds its passes, `GPU bridge` its
filters. A timestamp pair is the GPU's wall time between the two, so it includes the CUDA work the
GPU runs in between: the pose's TensorRT kernels share the GPU, and the fluid's passes read longer
with them ([The pose's share](#the-poses-share)). The frame is `Render interval`, the time between
two render ticks; its fps is 1000 over its mean.

## The stages

At the preset as saved, the grid 448 × 800 × 64, MacCormack, the density on the grid, and at the
default this document sets, 256 × 448 × 128:

| Stage                 | Timer                          | 448 × 800 × 64 | 256 × 448 × 128 |
|-----------------------|--------------------------------|----------------|-----------------|
| motion                | `GPU motion`                   | 0.40 / 0.63    | 0.43 / 0.65     |
| bridge, lowpass       | `GPU bridge lowpass`           | 0.08 / 0.08    | 0.09 / 0.08     |
| bridge, bandpass      | `GPU bridge bandpass`          | 0.39 / 0.71    |                 |
| bridge, the mixes     | `GPU bridge mix`               | 0.07 / 0.06    | 0.07 / 0.06     |
| fluid, inject         | `GPU fluid inject`             | 0.96 / 1.34    | 0.30 / 0.55     |
| fluid, dampen         | `GPU fluid dampen`             | 0.52 / 0.83    | 0.35 / 0.60     |
| fluid, advect velocity | `GPU fluid advect velocity`   | 1.90 / 5.79    | 0.86 / 1.21     |
| fluid, viscosity      | `GPU fluid viscosity`          | 0              | 0               |
| fluid, vorticity      | `GPU fluid vorticity`          | 2.00 / 2.64    | 1.28 / 1.59     |
| fluid, advect temperature | `GPU fluid advect temperature` | 0.45 / 0.86 | 0.23 / 0.23    |
| fluid, buoyancy       | `GPU fluid buoyancy`           | 0              | 0               |
| fluid, pressure       | `GPU fluid pressure`           | 6.74 / 8.09    | 4.40 / 5.91     |
| fluid, density        | `GPU fluid density`            | 2.04 / 5.29    | 1.10 / 1.76     |
| fluid, composite      | `GPU fluid composite`          | 1.16 / 1.45    | 2.44 / 3.53     |
| fluid, the step       | `GPU fluid`                    | 14.86 / 21.64  | 10.68 / 12.09   |
| the frame             | `Render interval`              | 17.93 / 25.34, 56 fps | 13.29 / 18.50, 75 fps |

The bandpass runs only while a mix takes one of its readings; the preset takes the trail alone, so
its row is from a run with `body_force.acceleration` taking `change`, `start`, `bow`, `stop` and
`wake` at 1. The viscosity is 0 and the buoyancy has no gravity in the preset, so both skip their
dispatches. The pressure is the step's largest pass, its 40 Jacobi iterations; the composite marches
one step per layer over the 2160 × 3840 output, so it grows with `layers`, not with the grid's x
and y.

The motion measures once per camera frame, 44 a second on the recording; every other stage runs
every frame.

### The advection

At 448 × 800 × 64, the density on the grid:

| `advection`        | advect velocity | density     | the step      | the frame     |
|--------------------|-----------------|-------------|---------------|---------------|
| `SEMI_LAGRANGIAN`  | 0.61 / 0.94     | 0.68 / 1.21 | 13.13 / 16.72 | 18.63 / 33.99 |
| `MACCORMACK`       | 1.90 / 5.79     | 2.04 / 5.29 | 14.86 / 21.64 | 17.93 / 25.34 |

MacCormack costs 1.3 ms on the velocity and 1.4 ms on the density, three reads of each field where
the semi-Lagrangian step makes one. The semi-Lagrangian run's frame has CPU stalls in `Render update`
(10.9 ms mean against 5.7), so its frame is no measure of the advection.

### The grid

MacCormack, the density on the grid, `pressure.iterations` 40:

| Grid             | Depth, widths | Cells  | pressure     | composite   | the step      | the frame             |
|------------------|---------------|--------|--------------|-------------|---------------|-----------------------|
| 448 × 800 × 64   | 0.14          | 23 M   | 6.74 / 8.09  | 1.16 / 1.45 | 14.86 / 21.64 | 17.93 / 25.34, 56 fps |
| 448 × 800 × 128  | 0.29          | 46 M   | 10.94 / 12.65 | 2.06 / 2.10 | 30.43 / 52.66 | 33.96 / 55.76, 29 fps |
| 320 × 576 × 160  | 0.50          | 29 M   | 8.17 / 9.78  | 2.46 / 2.94 | 20.40 / 29.24 | 23.09 / 32.53, 43 fps |
| 256 × 448 × 128  | 0.50          | 15 M   | 4.40 / 5.91  | 2.44 / 3.53 | 10.68 / 12.09 | 13.29 / 18.50, 75 fps |

The depth is `layers` / `width`, the cells being cubic. 256 × 448 × 128 is the one grid that holds
60 fps, and it is half a width deep where the preset's is 0.14.

### The pose's share

The same run, 448 × 800 × 64, with the pose on and off (`pose.pose.enabled`):

| Pose | advect velocity | vorticity   | pressure    | density     | the step      | the frame              |
|------|-----------------|-------------|-------------|-------------|---------------|------------------------|
| on   | 1.90 / 5.79     | 2.00 / 2.64 | 6.74 / 8.09 | 2.04 / 5.29 | 14.86 / 21.64 | 17.93 / 25.34, 56 fps  |
| off  | 0.56 / 0.82     | 0.62 / 0.62 | 4.95 / 5.51 | 0.42 / 0.42 | 8.21 / 8.98   | 9.81 / 10.68, 102 fps  |

Without the pose the step takes 8.2 ms: the 6.6 ms between the two is the TensorRT kernels the GPU
runs between the fluid's passes, the person detection's 34 ms and the RTM pose's 14 ms of CPU-side
inference a cycle **(deduction)**. The budget is the app's, so every other table is with the pose on.

## The particles

`density.carrier` `PARTICLES`: the density's advection is the particles' (`GPU particles advect`, in
`GPU fluid density`), and the composite their draw and march (`GPU particles draw`,
`GPU particles march`, in `GPU fluid composite`). The emission is `GPU particles emit`, under 0.2 ms
in every run. `rate` 20 and the other particle settings at their defaults; the ring full.

At the field defaults, 32 M particles, on 448 × 800 × 64:

| `draw`, `overlap`   | advect      | draw            | march       | the frame               |
|---------------------|-------------|-----------------|-------------|-------------------------|
| `POINTS`, occluded  | 4.01 / 5.80 | 20.51 / 38.21   | 0.42 / 0.57 | 41.12 / 62.57, 24 fps   |
| `POINTS`, additive  | 4.31 / 6.49 | 5.15 / 6.65     | 0.08 / 0.11 | 24.47 / 29.25, 41 fps   |
| `STREAKS`, occluded | 4.44 / 5.62 | 275.40 / 625.40 | 1.03 / 1.27 | 295.42 / 646.63, 3 fps  |
| `BLOBS`, occluded   | 3.90 / 6.05 | 522.33 / 916.49 | 1.07 / 1.27 | 541.36 / 936.39, 2 fps  |
| `SURFACE`           | 2.48 / 3.55 | 1092.27 / 1569.24 | 1.15 / 1.37 | 1111.00 / 1589.10, 1 fps |

On 256 × 448 × 128, by count:

| `draw`, `overlap`   | Count | advect      | draw        | march       | the frame              |
|---------------------|-------|-------------|-------------|-------------|------------------------|
| the grid's density  |       | 1.10 / 1.76 | 2.44 / 3.53 |             | 13.29 / 18.50, 75 fps  |
| `POINTS`, occluded  | 8 M   | 1.37 / 2.15 | 9.97 / 13.46 | 0.46 / 0.67 | 21.27 / 25.36, 47 fps |
| `POINTS`, occluded  | 2 M   | 0.28 / 0.52 | 1.07 / 2.26 | 0.41 / 0.91 | 11.20 / 17.65, 89 fps  |
| `POINTS`, additive  | 16 M  | 3.09 / 4.66 | 2.91 / 4.05 | 0.13 / 0.25 | 15.59 / 18.34, 64 fps  |
| `STREAKS`, occluded | 1 M   | 0.14 / 0.15 | 0.35 / 0.81 | 0.33 / 0.66 | 10.04 / 12.00, 100 fps |
| `STREAKS`, occluded | 250 k | 0.07 / 0.20 | 0.53 / 1.94 | 0.28 / 0.46 | 10.23 / 17.81, 98 fps  |
| `BLOBS`, occluded   | 500 k | 0.10 / 0.32 | 6.48 / 30.19 | 0.37 / 0.63 | 16.53 / 40.48, 60 fps |
| `BLOBS`, occluded   | 100 k | 0.07 / 0.31 | 1.33 / 9.04 | 0.32 / 0.61 | 10.95 / 20.24, 91 fps  |
| `SURFACE`           | 250 k | 0.07 / 0.09 | 4.27 / 27.05 | 0.89 / 1.59 | 14.79 / 38.38, 68 fps |
| `SURFACE`           | 100 k | 0.04 / 0.04 | 0.84 / 2.52 | 0.69 / 1.36 | 10.95 / 15.62, 91 fps  |

The draw is the cost, and it is the fill at 2160 × 3840, not the count alone: points and streaks
grow faster than their count once many land on one pixel (8 M points 9 times the draw of 2 M), and
blobs and the surface by their sprites' area, `radius` 24 pixels, their p95 several times their mean
where a strong drop's sprite grows to twice the radius **(deduction)**. The advection is 0.2 ms a
million particles. At its budget below, every look draws in less than the grid's composite, which
marches 128 layers at 4K where the particles' march takes 16 slices.

## The budget

One count holds for every look, since `count` is a `Field.INIT`; what each look fits in 16.7 ms
with the pose, on 256 × 448 × 128:

| `draw`, `overlap`   | Count within 60 fps | Measured at it                   |
|---------------------|---------------------|----------------------------------|
| `POINTS`, occluded  | 2 M                 | 11.20 / 17.65                    |
| `POINTS`, additive  | 8 M **(deduction)** | 16 M at 15.59 / 18.34            |
| `STREAKS`, occluded | 1 M                 | 10.04 / 12.00                    |
| `BLOBS`, occluded   | 100 k               | 10.95 / 20.24, the p95 its spikes |
| `SURFACE`           | 100 k               | 10.95 / 15.62                    |

The defaults this sets (`modules/flow/fluid_3d/fluid_settings.py`): the grid `width` 256, `height`
448, `volume.layers` 128, half a width deep; the particles' `count` 2 M, the points' budget, the
default `draw`. The preset takes the grid and, for its streaks, 1 M.

## Open

- The blobs' and the surface's p95: a sprite that grows to twice `radius` on a strong drop fills
  four times the pixels, and the frame's p95 at 100 k blobs is 20 ms. A smaller `radius`, or the cap
  at less than twice, is the knob, judged on the look.
- The pose's share of the GPU, 6.6 ms of the step: the fluid's budget is what the pose leaves. A
  pose model on a smaller input, or the detection at a lower rate, gives the fluid its time back.
