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

### 1. The z position without the depth's staircase

The blocks in the 4K density are made at the injection: the bridge's boundary density, color × amount, is
smooth; each of its pixels goes into the volume at its own depth, the slab around its z position
(`slab_center`), and the z position steps. It is the motion's z position as made (`ZPosition`), its weight the
depth mask × the mask: the depth mask is the depth's own cut, a staircase of depth pixels; inside the mask's
smooth outline but outside that staircase its weight is 0, and a pixel without a z position goes in at
`input.layer` instead. So neighbouring pixels along the outline go in at their own depth or at `layer`, and the
composite shows the step between those layers as a staircase. The velocity takes the trails' z positions,
which the lowpass's blur smooths (`blur`), so the water's placement has no such steps; the densities and the
warmth take the z position raw.

The fix: the z position spread over its neighbours by a normalized blur where it is made. It is held as z ×
weight in red over the weight in green, so a blur of both channels and their ratio gives each pixel the
weighted mean of its neighbours' z: inside the body a smooth placement, and in the staircase band, where the
weight is 0, its neighbours' z instead of a jump to `layer`; the weight blurred with it says how much of the
pixel has a z position. The radius in depth pixels, as the step it bridges is one, turned into the view's
from the depth's spacing in the view (`depth_step`), a setting of the z position beside its range. It
places where z goes in, not what z is: no filter on the depth or on z.

Judged on the edges recording at drag 0, fresh injection: the steps along the outline.

### 2. The velocity for the fine density

The 4K amount is carried by the 288 × 512 velocity read straight between its cells, 7.5× coarser, whose
kinks show as the cells in a sharp density. A tricubic read where the amount is carried: 8 trilinear reads
for 1 at each of 199 M texels a step, twice that with MacCormack **(deduction)**; kept if the look gains
for its cost.

## Open

- The whole chain before the fluid, camera to bridge, timed as one at the preset: what each pass costs
  against what it gives the fluid.
- A finer depth would give the mask and z a finer edge at the source.
