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
| the 4K density's amount      | the dye, by a B-spline (`upscaled.cubic`)   | 2160 × 3840 × 24           | its own threads, from MacCormack and the turbulence        |
| the 4K density's color       | the dye's color (`upscaled.color` DYE)      | 448 × 800 × 24             | the dye's, read linearly by the composite                  |

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

The 4K density's color volume costs 1.3 ms of the density pass, 8.6 M texels against the amount's 199 M,
RGBA16F as 8 bits would round at every advection; SIMULATION, the simulation density's 288 × 512 color, is
the cheaper choice.

## The mask and the depth mask

The player's shape and where the depth is the player's are two things, made in one place, the view stage
(`ViewLayer`, [INPUT_THREADS.md](INPUT_THREADS.md#1-deep_flow-the-input-refactor)):

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
| inside the body | steps from block to block: the depth velocity's window of depth samples reads the NEAREST depth's blocks | the track steps from block to block |

The dye's outline is the mask's: its color and the x/y of its amount stop at the guided edge; only z's
share of the amount steps along the staircase.

## Plan

The input first, as good as it gets, before the 4K density's heavy side.

### 1. The depth read across a surface

Where the depth is cut into the view, in the flow's input stage ([INPUT_THREADS.md](INPUT_THREADS.md),
built there), each view pixel reads the four depth pixels around it and interpolates them where they lie
within the same-surface tolerance of each other (the motion's `z.tolerance`, shared), else takes the
nearest, so an edge never reads as a distance in between. A depth pixel of 0, no depth (where the
alignment's scatter lands nothing), is never blended: with one among the four it takes the nearest.

| Problem                         | Fixed  | Why                                                                     |
|---------------------------------|--------|-------------------------------------------------------------------------|
| z's blocks inside               | yes    | the depth samples change smoothly inside a surface, so z does           |
| the z position's blocks inside  | yes    | the track follows the smooth depth                                      |
| the staircase at the edge       | no     | at an edge it takes the nearest; the depth holds no finer edge          |
| a depth pixel's own flicker     | softened | it spreads as a soft bump of a depth pixel, without edges; removing it would filter the depth's noise |

At the source, every reader gets it, with no setting. A blur in the bridge would fix only the dye's
amount, need a depth pixel's radius and fade z along the outline; a joint bilateral upsampling of the
depth (Kopf 2007) would also move its edges onto the image's, at more cost **(deduction)**. Judged by z's
step between neighbouring pixels inside the body, the noise at the edge and inside (`mask_probe` on
stillslowfast), and the dye at drag 0.

### 2. The velocity for the 4K density

The 4K amount is carried by the 288 × 512 velocity read straight between its cells, 7.5× coarser, whose
kinks show as the cells in a sharp density. A tricubic read where the amount is carried: 8 trilinear reads
for 1 at each of 199 M texels a step, twice that with MacCormack **(deduction)**; kept if the look gains
for its cost. The composite's read of the color by a B-spline, likewise.

## Open

- The whole chain before the fluid, camera to bridge, timed as one at the preset: what each pass costs
  against what it gives the fluid.
- A finer depth would give the mask and z a finer edge at the source.
