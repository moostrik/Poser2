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

## The baseline

The defaults (`modules/flow/fluid_3d/fluid_settings.py`): the grid `width` 256, `height` 448,
`volume.layers` 128, half a width deep; `volume.injection_spread` 0.075 widths, a body's
thickness; the particles' `count` 2 M, the points' budget, and their ranges to what fits:
`count` 10 k to 8 M, `rate` 0.01 to 50, `radius` 1 to 48 pixels, `spread` 0 to 10. The preset
takes the grid, the spread and, for its streaks, 1 M.

Each recommendation timed as set, the other particle settings at their defaults:

| Run                          | advect velocity | density     | pressure    | composite   | the step      | `Render update` | the frame             |
|------------------------------|-----------------|-------------|-------------|-------------|---------------|-----------------|-----------------------|
| grid, `SEMI_LAGRANGIAN`      | 0.40 / 0.77     | 0.43 / 0.85 | 5.24 / 6.83 | 2.44 / 3.94 | 10.97 / 13.37 | 6.53 / 9.96     | 13.98 / 17.94, 72 fps |
| grid, `MACCORMACK`           | 1.11 / 1.72     | 1.28 / 2.18 | 4.81 / 6.67 | 2.43 / 3.61 | 12.17 / 15.11 | 6.24 / 9.43     | 15.28 / 19.29, 65 fps |
| grid, MacCormack, bandpass on | 1.10 / 1.68    | 1.14 / 1.92 | 4.59 / 6.46 | 2.40 / 3.70 | 11.58 / 14.87 | 8.98 / 23.95    | 16.02 / 26.47, 62 fps |
| 2 M `POINTS`, occluded       | 1.12 / 1.65     | 0.34 / 0.68 | 4.59 / 6.33 | 3.05 / 5.16 | 11.24 / 14.35 | 10.95 / 28.72   | 16.11 / 31.35, 62 fps |
| 8 M `POINTS`, additive       | 1.10 / 1.63     | 1.29 / 2.11 | 4.66 / 6.36 | 1.64 / 2.79 | 10.80 / 13.51 | 10.30 / 27.80   | 15.34 / 30.50, 65 fps |
| 1 M `STREAKS`, occluded      | 1.12 / 1.66     | 0.18 / 0.21 | 4.61 / 6.42 | 1.87 / 3.44 | 10.02 / 12.74 | 10.01 / 25.37   | 14.36 / 28.40, 70 fps |
| 100 k `BLOBS`, occluded      | 1.17 / 1.71     | 0.05 / 0.05 | 4.53 / 6.30 | 2.29 / 4.49 | 10.37 / 13.57 | 8.02 / 20.32    | 14.13 / 24.12, 71 fps |
| 100 k `SURFACE`              | 1.16 / 1.76     | 0.05 / 0.05 | 4.89 / 6.86 | 1.70 / 3.78 | 10.18 / 13.32 | 10.06 / 25.06   | 14.55 / 27.85, 69 fps |

Every baseline holds 60 fps in the mean, 62 to 72 fps; none holds it at the p95. Two things set the
p95, not the fluid. `Render update`, the CPU side of the render, spikes to a p95 of 20 to 29 ms
in the particle runs against 9 to 10 in the grid's. And the pose ran harder in these runs than in the
sweep above: the person detection 2800 to 2970 times a minute against 2040 to 2230, and with it the
GPU passes 1 to 1.5 ms longer (vorticity 1.67 against 1.28 at the same grid, the step 12.2 against
10.7). So two runs of one configuration differ by that much; the tables above compare runs made
back to back.

## Faster

The technical changes that would make the fluid and the particles faster, ordered by the time each is
expected to give back at the baseline: 256 × 448 × 128 with MacCormack, the step 12.2 ms, the output
at 2160 × 3840 and every stage as it is. Nothing here turns a feature off or lowers a resolution. An
estimate is **(deduction)** unless its source measured it; a source's numbers are on its own scene.

### The fluid

| # | Change                                                                                                                          | Its share now                     | Expected back           | Source |
|---|---------------------------------------------------------------------------------------------------------------------------------|-----------------------------------|-------------------------|--------|
| 1 | The pressure by multigrid: a V-cycle, restriction to 128 × 224 × 64 and down, smoothed by red-black Gauss–Seidel, warm-started from the last step's pressure as now; or that cycle as the preconditioner of a conjugate gradient (MGPCG) | 4.8 ms, 40 Jacobi iterations | 2.5 to 3.5 ms | one V-cycle converges like about a thousand Jacobi iterations at the cost of about ten (gist); 3 × the accuracy of Jacobi in the same time on the GPU (Glimberg 2009); convergence independent of the grid's size (McAdams 2010) |
| 2 | The pressure by red-black Gauss–Seidel with over-relaxation (SOR) in place of Jacobi: the cheap form of 1, one solve, two colors a sweep, each color from the other's fresh values | the same 4.8 ms | 1.5 to 2.5 ms, the same residual in about half the sweeps | red-black ordering keeps most of Gauss–Seidel's lead on Jacobi in parallel (Pyroclast; Astaroth); Jacobi converges slowly (FLIP on GPU) |
| 3 | The composite's early exit, which never fires: it stops only at `glow` 0, and the preset's 0.1 keeps every ray marching all 128 layers for the additive sum; instead the additive sum stopped with the transmittance where it is under the output's precision, and empty space skipped by a coarse occupancy volume (the density's max over 8 × 8 × 8 cells), the ray stepping a block at a time where it is empty | 2.4 ms, 128 steps over 8.3 M pixels | 0.5 to 1.5 ms, by how much of the box is empty or already opaque | min-max octrees and occupancy grids in volume renderers |
| 4 | Passes fused, each field read once: the dampen into the velocity's advection; the curl and the confinement force in one pass with the curl in shared memory; the divergence into the first Jacobi dispatch; the body force's and the boundary's injections in one pass | dampen 0.4, vorticity 1.7, inject 0.8, the divergence in the pressure | 1 to 1.5 ms | the stencils are bandwidth-bound; kernel fusion and blocking (Intel 3.5D blocking, Aalto) |
| 5 | The temperature cleared once when `buoyancy` goes to 0, not both buffers every step, and its injection skipped while it is 0; nothing reads it then | 0.4 ms | 0.4 to 0.5 ms | the code: `Fluid3D._advect_temperature` |
| 6 | The 3D stencil passes on 3D workgroups: the advections, the curl, the divergence and the gradient run 16 × 16 × 1, one layer a group, so a group's z neighbors come from another group's reads; 8 × 8 × 4 groups keep the three neighbors in the cache | the advections 1.5, vorticity 1.7, the gradient and the divergence in the pressure | 0.3 to 0.6 ms | 3D blocking of stencils (Intel 3.5D blocking) |
| 7 | MacCormack in one dispatch: the forward trace, the error and the clamp in the predictor's pass, the predictor kept in registers, so each field is written once and read once instead of twice | +0.7 ms on the velocity, +0.85 on the density | 0.4 to 0.6 ms | the step's three passes read each field three times |
| 8 | A sparse grid: bricks of 8³ or 16³ cells, only those with velocity or density stepped (GVDB, EmberGen 2) | every pass | with the body alone in the box, up to half the step; the velocity fades over 60 s and the walls are closed, so the water fills the box over time and the gain goes to 0; the density's passes keep it | Wu et al. 2018; JangaFX |

### The particles

| # | Change                                                                                                     | Its share now                                        | Expected back                     | Source |
|---|------------------------------------------------------------------------------------------------------------|------------------------------------------------------|-----------------------------------|--------|
| 1 | Points and streaks rasterized in compute, not through the vertex and blend units: each particle adds its value by fixed-point `atomicAdd` into its slice of a storage buffer; a streak as a Bresenham line of the same adds | the points' draw, 2.7 ms at 2 M, 10 at 8 M; the streaks' 1.5 at 1 M | 2 to 5 × on the draw | Schütz 2019: 5 to 10 × faster than `GL_POINTS` at 1-pixel points, less above 2 pixels; `compute_rasterizer` |
| 2 | Weighted blended order-independent transparency in place of the 16 depth slices: two targets at the output, no layered draw, no 1 GB of slices cleared and marched a frame, near ink over far by a depth weight | the slices' clear and march, 0.4 ms, and the layered draw | the march, the clear and a third to half of the draw | McGuire and Bavoil 2013 |
| 3 | The blobs' and the spheres' sprites cut at 2 sigma instead of their full quad: a gaussian beyond 2 sigma holds 5% of its mass; and the growth capped under twice `radius`, since a sprite twice as wide fills four times the pixels | the blobs' p95, 9 ms of draw at 100 k, 30 at 500 k | half the fill; most of the p95 | measured above: the mean 1.3 against the p95 9 |
| 4 | The surface's filter as a narrow-range filter, which keeps the edges with fewer passes, and the narrow band: only the particles at the liquid's surface drawn into the depth and thickness | the surface's march, 0.8 ms, and its draw | 0.3 to 1 ms; 2.4 × on the surface pass | Truong and Yuksel 2018; Oliveira and Paiva 2022 |
| 5 | The draw by the alive count: `glDrawArraysIndirect` with the count the emission's head gives, so the vertex shader runs for the particles there are, not for every slot of the ring | the vertex shader over `count` slots a frame | up to the ring's empty share; 0 once it wraps | the code: the draw over `count` |
| 6 | The ring sorted by the Morton order of the particles' cells now and then (a bitonic sort in compute), so a group's particles read the same velocity texels | the advection, 0.2 ms a million                      | under 0.5 ms at the baseline; 1.5 to 2 × at 8 M and more | a 3.2 × gain from sorting in a molecular-dynamics benchmark (NVIDIA forum); Morton-sorted SPH |

### Measured

Each change against the code before it, three runs back to back (`trial_runs.py`, the preset with the
density on the grid unless the row says otherwise): the pass before, with the change, before again. The
noise is the two befores' difference; a change counts as faster when it beats both by 2.5 times that.
The harness's check is `fluid_compare.py` on stillslowfast, its water numbers and four saved outputs
against the code before.

| # | Change                          | Pass, ms mean        | Before | Change | Before again | Noise | Faster | Harness and picture         | Kept as     |
|---|---------------------------------|----------------------|--------|--------|--------------|-------|--------|-----------------------------|-------------|
| 1 | the temperature cleared once    | advect temperature   | 0.32   | 0.00   | 0.30         | 0.02  | yes    | the same; 0 of 255 apart    | replacement |
|   |                                 | inject               | 0.66   | 0.52   | 0.66         | 0.00  | yes    |                             |             |
| 2 | the composite's empty blocks skipped, by a volume of each 8³ block's largest density | composite, with the occupancy pass | 2.46 | 3.49 | 2.45 | 0.01 | no | the same; 1 of 255 apart | dropped |
| 3 | 8 × 8 × 4 workgroups on the 13 stencil passes | the step | 10.30 | 10.09 | 10.33 | 0.03 | yes | the same; 0 of 255 apart | replacement |
|   |                                 | advect velocity      | 0.82   | 0.78   | 0.81         | 0.01  | yes    |                             |             |
|   |                                 | vorticity            | 1.25   | 1.22   | 1.26         | 0.01  | yes    |                             |             |
|   |                                 | inject               | 0.49   | 0.41   | 0.47         | 0.02  | yes    |                             |             |
|   |                                 | dampen               | 0.36   | 0.34   | 0.35         | 0.01  | no     |                             |             |
|   |                                 | density              | 1.11   | 1.15   | 1.15         | 0.04  | no     |                             |             |

Item 2 was slower in both forms, one thread a block (3.84 ms) and one workgroup a block (3.49): on the
recording the density fills most blocks, so the skip saves little and its pass and branch cost more. The
early exit at opacity was not built: with `glow` above 0 the additive sum behind opaque density is part of
the picture, so stopping there changes it.

### What sets the p95

Not the fluid: `Render update`, the CPU side of the render, spikes to a p95 of 20 to 29 ms in the
particle runs against 9 to 10 in the grid's, with a mean of 8 to 11. A py-spy profile of a particle
run finds the call; the p95 then falls to the mean, about 15 ms.

Sources: [McAdams, Sifakis, Teran 2010](https://pages.cs.wisc.edu/~sifakis/project_pages/mgpcg.html);
[Glimberg et al. 2009](https://diglib.eg.org/items/22d6a58b-5efa-4cc9-b204-24bbbe2b4108/full);
[a multigrid against Jacobi write-up](https://gist.github.com/61315/55f11aba9ba77c76aeb8f004e58f650f);
[Pyroclast, red-black Gauss–Seidel](https://arxiv.org/pdf/2603.14040);
[Astaroth, Poisson solvers](https://arxiv.org/pdf/2605.03563);
[FLIP on CPU and GPU](https://arxiv.org/pdf/2404.01931);
[3.5D blocking, Intel 2010](https://ieeexplore.ieee.org/document/5645463/keywords);
[compressible flow on GPUs, Aalto](https://ar5iv.arxiv.org/html/1707.08900);
[Wu et al. 2018, sparse volumes](https://people.csail.mit.edu/kuiwu/gvdb_sim.html);
[EmberGen 2's sparse solver](https://www.cgchannel.com/2025/01/check-out-the-new-features-due-in-embergen-2-0);
[min-max octree volume rendering](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/7242401);
[Schütz 2019, point clouds in compute](https://arxiv.org/pdf/1908.02681), [its code](https://github.com/m-schuetz/compute_rasterizer);
[McGuire and Bavoil 2013](https://www.jcgt.org/published/0002/02/09/paper.pdf);
[Truong and Yuksel 2018](https://www.cemyuksel.com/research/papers/narrowrangefilter.pdf);
[Oliveira and Paiva 2022](https://diglib.eg.org/handle/10.1111/cgf14510);
[sorting particles by a space-filling curve](https://forums.developer.nvidia.com/t/fast-texture-fetching-an-algorithm-for-fast-texture-fetching/5489);
[StarsX/Smoke3D, a compute-shader 3D smoke](https://github.com/StarsX/Smoke3D).

## Open

- The blobs' and the surface's p95: a sprite that grows to twice `radius` on a strong drop fills
  four times the pixels. The cut at 2 sigma and the lower cap ([The particles](#the-particles-1), 3)
  are the fix, judged on the look.
- The run-to-run spread of the pose's load, 1 to 1.5 ms of the step between two runs of one
  configuration: the person detection's rate is not fixed.
