# Resolutions

The sizes the flow's picture and motion pass through, from the camera to the 4K output, why each is what
it is, and the plan for the rest. **(deduction)** marks what is inferred, **(site fact)** what the operator
saw. All numbers are deep_flow's preset.

## The rules

- A quantity is made at the size of what it comes from and carried at the size where it is used.
- No size above what the information holds: an enlargement only where the result is seen, at the output.
- No detail thrown away before it is used: a step down of 2× or more averages what it leaves (area, box or
  mip level); a step up, at the output, is smooth (a cubic B-spline).

The camera is a given **(site fact)**: its sizes are the same on USB and the network; a finer depth would
help, more color would not.

## The chain

| Data                         | Made from                                   | Size                       | Its detail                                                |
|------------------------------|---------------------------------------------|----------------------------|-----------------------------------------------------------|
| camera color                 | `P800` cut to 5:4 (`direct_size`)           | 992 × 800, 50 fps          | the crop around the player, at most 800 rows               |
| camera depth                 | Neural Depth `NEURAL_480`                   | 480 × 300                  | one depth pixel about 15 view rows at the blocky recording's distance |
| the image (the view)         | the view's window, drawn once               | 448 × 800, the motion's    | the window's rows, one to one at most                      |
| the mask                     | the depth's cut, guided onto the image      | 448 × 800                  | edges the image's                                          |
| `depth_mask`                 | the view's depth cut by near and far        | 448 × 800                  | a staircase of depth pixels at the edges                   |
| the depth on the view        | the aligned depth, read across a surface    | 448 × 800                  | smooth within a surface, a depth pixel's step at an edge   |
| x/y                          | the image (Lucas–Kanade, 7 × 7 window)      | 448 × 800                  | smooth over ~7 px                                          |
| z, the z position            | the depth (the depth velocity, its track)   | 448 × 800                  | the depth's                                                |
| the bridge's readings, dye   | x/y, z, the image × the mask                | 448 × 800, per capture     | held between the camera's frames                           |
| the fluid                    | the readings, read bilinearly (1.56× down)  | 288 × 512 × 64             | motion finer than a cell averages away                     |
| the density                  | the dye, color × amount                     | 448 × 800 × 24             | the dye's; the fine density's color, read linearly by the composite: a B-spline read cost 0.75 ms and showed nothing on the hands recording |
| the fine density             | the dye's amount, by a B-spline (`fine_density.cubic`) | 2160 × 3840 × 24 | its own threads, from MacCormack and the turbulence        |

The tilt is a homography the host applies where the frame is drawn into the view (`tilt_correction`), so
the frame stays 800 rows at any tilt; a recording holds the frame as delivered. Which IMX586 mode the
Camera node makes 800 rows from (2000 × 1500 or 1920 × 1080) is not checked.

The view stage and what is drawn from it (`VIEW_LAYERS` in `apps/deep_flow/render.py`) are allocated at the
motion's `width` and `height` and follow them, as the view is there for the fluid **(site fact)** and the other
readers, the debug panels and the color mask over the composite, show the same detail enlarged. A larger view read
back bilinearly 2.4× down would hand the solve an aliased image. The fluid reads the bridge's outputs
bilinearly: at 1.56× that loses little, as the readings are smooth over the motion's window and the
lowpass's blur **(deduction)**.

| Pass, GPU per frame                          | 448 × 800  | 1080 × 1920 |
|----------------------------------------------|------------|-------------|
| the view, as the center layers, guide off    | 0.18 ms    | 0.46 ms     |
| the guided filter, 1 depth pixel             | 0.05 ms    | 0.22 ms     |
| the guided filter, 2 depth pixels            | 0.06 ms    | 0.35 ms     |

The density and the fine density ([FLUID_3D.md](FLUID_3D.md#the-densitys-detail)): one RGBA density, color
× amount, at its own `density.width` × `height` × `depth`, the dye's size by the fine density's layers in the
preset, 8.6 M texels against the fine density's 199 M, RGBA16F as 8 bits would round at every advection; it is
the fine density's color and what the buoyancy weighs, by uv. The density at the simulation's 288 × 512 cells
blurred the dye's color **(site fact)**. On the show recording, `-sim`, 40 s, two runs: the fluid 9.6 / 9.1 ms,
its density pass 6.1 / 5.8, where the density on the grid beside a color volume took 10.1 / 9.2 and 6.7 / 5.9.

## The mask and the depth mask

The player's shape and where the depth is the player's are two things, made in one place, the view stage
(`ViewLayer`, [INPUT_THREADS.md](INPUT_THREADS.md#the-view-stage)):

- `mask`: the depth mask, its edge moved onto the camera image's by the guided filter
  (He 2010), then the AA's threshold at 0.5, a crisp edge where the image puts it. Where the image has no
  edge the filter only averages the cut, and the threshold makes that a smoothed outline. `guided_radius`
  is in the cut's (the depth's) pixels, how far its edge may be off, turned into the view's each frame from
  the window; `edge_contrast` is the image's brightness spread (0 to 1) above which the edge follows the
  image's, squared inside as the filter's ε. It gates x/y in the motion with `use_mask`, and the dye's color
  in the bridge.
- `depth_mask`, the view's depth cut by `near` and `far`, is the motion's `depth_mask`: it gates z on top of
  the mask, always, and it is the z position's weight with the mask. The z position's value is held to 0..1
  where the track lags past the range. A mask from another source (RVM) is open
  ([FLOW.md's Open](FLOW.md#7-the-depths-mask-with-another-mask-source)).

The preset: `guided_radius` 2, `edge_contrast` 0.03, and no erosion, as one pulls the guided edge off the
image's. Measured on stillslowfast, the still part, the motion's thresholds at 0; z and x/y in
widths/s, p99 within 6 px of the mask's edge and inside it; fit is the image's brightness gradient along
the mask's edge, higher the closer the edge sits on the image's. One run's p99 swings by about ±0.02; fit
holds to ±0.0003.

| Mask                                   | Runs | z at edge (p99 / mean) | z inside | x/y at edge | Fit    |
|----------------------------------------|------|------------------------|----------|-------------|--------|
| the cut, erosion 1, z by the mask      | 3    | 0.053 / 0.0040         | 0.087    | 0.030       | 0.0308 |
| guided 2, erosion 0                    | 3    | 0.050 / 0.0036         | 0.057    | 0.028       | 0.0449 |
| guided 1.5, erosion 0                  | 3    | 0.053 / 0.0036         | 0.064    | 0.028       | 0.0435 |
| guided 1.5, erosion 1                  | 1    | 0.052 / 0.0041         | 0.063    | 0.025       | 0.0276 |
| guided 1.5, erosion 2                  | 1    | 0.053 / 0.0041         | 0.066    | 0.030       | 0.0210 |
| guided 2, erosion 0, contrast 0.015    | 1    | 0.053 / 0.0037         | 0.106    | 0.027       | 0.0450 |
| guided 2, erosion 0, contrast 0.06     | 1    | 0.065 / 0.0041         | 0.078    | 0.027       | 0.0442 |

## The depth's pixels in the fluid

The depth reaches the fluid two ways: z, into the velocity's z and, through the speed (the length of x, y,
z), into the dye's amount and the warmth; and the z position, the layer the velocity, the dye and the warmth
go in. Each shows the depth's pixels in two places:

| Where           | z                                                          | the z position                         |
|-----------------|------------------------------------------------------------|----------------------------------------|
| at the edge     | cut along `depth_mask`'s staircase, a band a depth pixel wide inside the mask's outline | its weight cut along the staircase |
| inside the body | smooth: the depth is read across a surface                 | smooth, for the same reason            |

The view stage reads the depth into the view across a surface (`DepthUpscale`, `view_window.edge_aware_depth`):
each view pixel takes the four depth pixels around it, interpolated where they lie within the motion's
`z.tolerance` of each other, else the nearest, so an edge never reads as a distance in between, and a 0, no
depth, is never blended. On stillslowfast, 3 runs, the motion's thresholds at 0: neighbouring pixels inside
the body with the very same depth 11% (67% read nearest), z inside the mask p99 0.040 widths/s (0.057), z at
the edge 0.041 (0.050), the mask's fit unchanged. A depth pixel's own flicker stays, as a soft bump of its
size; removing it would filter the depth's noise.

The dye's outline is the mask's: its color and the x/y of its amount stop at the guided edge; only z's
share of the amount steps along the staircase.

## Plan

The input first, as good as it gets, before the 4K density's heavy side.

### 1. The depth read three ways, side by side

The staircase starts where the depth is read into the view: `depth_mask` and the z position step by depth
pixels because the read takes a depth pixel's value across an edge. A view setting, `depth_read`, chooses the
read, so the three are compared on the same recording:

| `depth_read` | At an edge, a view pixel takes                                  | Source                         |
|--------------|-----------------------------------------------------------------|--------------------------------|
| `SURFACE`    | the depth pixel it lies in: the edge on the depth's pixels      | the current read               |
| `MASK`       | the nearest depth pixel on the side the mask gives it           | our own rule, no paper         |
| `BILATERAL`  | a weighted mean of the 5 × 5 around it, weighed by the image    | Kopf et al. 2007, eq. 3        |

`MASK` reads as `SURFACE` does wherever that depth is on the pixel's side of the mask: a depth in `near`..`far`
inside the mask, another depth (not 0) outside it. A pixel whose depth is on the wrong side takes the nearest
depth pixel on its side within `guided_radius`, as that is how far the guided filter (He 2010) may move the
mask's edge off the depth's; with none in reach it keeps its own. The depth's edge then lies on the mask's. The
mask is made from a depth read, so `MASK` reads twice: `SURFACE`, the cut and the guided mask, then the read
with that mask and the cut again. No paper does this; it carries the guided filter's edge into the depth
without new parameters.

`BILATERAL` is joint bilateral upsampling (Kopf, Cohen, Lischinski, Uyttendaele 2007,
[the paper](https://johanneskopf.de/publications/jbu/paper/FinalPaper_0185.pdf)). Each view pixel takes the depth
pixels of a 5 × 5 window around it. Each pixel is weighed by two Gaussians:

- f, the spatial Gaussian, of its distance in depth pixels, with σd = 0.5.
- g, the range Gaussian, of the colour distance between the camera image at the view pixel and at that depth
  pixel's place, on colour 0 to 1.

These follow the paper: its window, its σd, and the range filter on the high-resolution guidance image. It
differs in two places:

| Deviation                                             | Why                                                       |
|-------------------------------------------------------|-----------------------------------------------------------|
| a depth pixel without depth (0) is left out of the sum | the paper's solution is dense; a 0 averaged in pulls an edge toward 0 m |
| σr is a view setting, `bilateral_contrast`, the paper's 0.1 the default | the paper sets it per application; 0.1 is its value for stereo depth on colour 0 to 1 |

Where the image shows no edge, `BILATERAL` blends the depths across a depth edge into a depth in between. The
paper's filter does this too. `SURFACE` and `MASK` never do.

What each read does to z and to what is made from the depth **(deduction)**:

| Read        | z's edge                  | z inside                         | Risks                                              | Cost                 |
|-------------|---------------------------|----------------------------------|----------------------------------------------------|----------------------|
| `SURFACE`   | the depth's pixels        | the depth's, interpolated        | the staircase                                      | the read             |
| `MASK`      | the mask's, the image's   | as `SURFACE`                     | a depth pixel that straddles the edge is taken whole | a second read and cut, the pixels within `guided_radius` at the edge |
| `BILATERAL` | the image's edges         | close to `SURFACE`'s             | the image's texture and its noise in z; a depth in between where the image has no edge; the guided mask moves with `depth_mask` | 25 depth and 25 image reads a pixel |

z is the depth's change over time per pixel, so a pixel that changes side between two frames reads as z. With
`MASK` the depth's edge and `depth_mask`'s are the mask's, so no depth from behind the player lies inside
the mask and z's gate cuts where the edge is. With `BILATERAL` the camera image's noise moves the weights every
frame, which moves the depth where nothing moved. Inside the body it gains little: at σd 0.5 a neighbour one
depth pixel away weighs 0.135 of the pixel itself.

No read adds depth detail inside the body: z takes x/y's resolution at the outline and stays the depth's,
interpolated, inside it.

#### The tests

The goal: z at the view's size, x/y's, with no more noise than `SURFACE` gives, preferably less.

| Test           | Recording                    | What                                                       | Passes when                               |
|----------------|------------------------------|------------------------------------------------------------|-------------------------------------------|
| correctness    | measure                      | `scripts/view_check.py --read`, the shader against numpy   | no pixel off by over 1 mm                 |
| the staircase  | measure                      | the share of the mask's pixels whose z position has no weight | below `SURFACE`'s, near 0               |
| the edge       | measure                      | the fit along `depth_mask`'s edge                          | above `SURFACE`'s                         |
| noise          | measure, the still part      | z p99 inside `SURFACE`'s mask and at its edge              | at most `SURFACE`'s                       |
| the signal     | measure, slow and fast       | z on the moving player, fast ÷ noise                       | at least `SURFACE`'s                      |
| x/y            | measure                      | x/y at the edge, a control                                  | unchanged                                 |
| texture        | hands                        | the depth in the view, by eye, on the clothes and the hands | no image pattern in the depth             |
| the look       | edges, drag 0, fresh injection | the 4K output along the outline, per read                 | no steps                                  |
| cost           | show, one marked loop        | each read's GPU time                                        | reported                                  |

Every number goes into one table per read, the motion's thresholds at 0, every read measured in `SURFACE`'s
mask so the regions are the same. The staircase's share is the cause measured; `MASK`'s `depth_mask` lies on
the mask by its rule, so its overlap with the mask says nothing.

Measured on the measure recording, `-sim`, its first 16 s from the in marker (still 1–5 s, the slow move
5.5–10 s, the fast 11.5–16 s), one run each, the motion's thresholds at 0, z in widths/s at `z_gain` 0.4:

| Read        | No z position | Fit    | z noise inside / edge | Slow  | Fast  | Fast ÷ noise | x/y at edge |
|-------------|---------------|--------|-----------------------|-------|-------|--------------|-------------|
| `SURFACE`   | 8.0%          | 0.0413 | 0.183 / 0.134         | 0.016 | 0.028 | 0.153        | 0.124       |
| `MASK`      | 7.7%          | 0.0613 | 0.155 / 0.127         | 0.017 | 0.027 | 0.177        | 0.125       |
| `BILATERAL` | 10.1%         | 0.0590 | 0.138 / 0.115         | 0.015 | 0.024 | 0.177        | 0.159       |

A second `MASK` run gave noise 0.179 / 0.151: the noise differences lie within one run's swing. With `MASK`
the view's depth and `depth_mask` cover the whole mask; its 7.6% without a z position all lie where the depth
track has no depth. That is the depth velocity's step filter (`DepthStepFilter`): a depth sample whose 3 × 3
samples span a step beyond `tolerance` has no depth, a band a depth sample wide along every outline, along the
staircase with `SURFACE` and along the mask's outline with `MASK`. It keeps z from jittering at a silhouette;
the z position takes its depth from the same track.

### 2. The dye at its readings' depth, and no `layer`

The dye's amount and its depth come from two places. The amount is mixed from the readings (`Bridge3DLayer._mix`,
`boundary.density`, `boundary.temperature`, the body force's), which the trails make, advected and blurred. Its
depth is the motion's z position now (`input.z_position`), raw. Where the trail's blur spreads the amount into
the band without a depth, the pixel has an amount and no z position, and `slab_center` puts it at
`input.layer`: the step along the outline the composite shows. The velocity has no such step, as its amount
and its z position both come from its readings (`ZPositionMix`).

The fix, the velocity's: each dye output's z position is mixed from the readings it is mixed from, by the same
mix (`ZPositionMix` with `boundary.density`, `boundary.temperature`, `body_force.density`,
`body_force.temperature`), held by the density bridge as its amount is. The amount and its depth then come from
one trail, and the trail's blur fills the band. The particles and the fine density take the density's.

`layer` goes: a pixel without a z position puts nothing in, for the velocity, the acceleration, the densities
and the warmth (`slab_center`, `uTargetLayer`, `input.layer` and its preset key removed). A capture without depth
then puts nothing in; a 2D camera needs a depth of its own, made from the image, an item of
[FLOW.md's Open](FLOW.md#open). [BRIDGE.md's z positions](BRIDGE.md#the-z-positions) and
[FLUID_3D.md's inputs](FLUID_3D.md#the-inputs) change with it.

Judged on the edges recording at drag 0, fresh injection, `MASK`: the steps along the outline; and the share of
the dye's amount that lands without a z position (0 by the rule: the amount it loses).

### 3. The velocity for the fine density

The 4K amount is carried by the 288 × 512 velocity read straight between its cells, 7.5× coarser, whose
kinks show as the cells in a sharp density. A tricubic read where the amount is carried: 8 trilinear reads
for 1 at each of 199 M texels a step, twice that with MacCormack **(deduction)**; kept if the look gains
for its cost.

## Open

- The whole chain before the fluid, camera to bridge, timed as one at the preset: what each pass costs
  against what it gives the fluid.
- A finer depth would give the mask and z a finer edge at the source.
