# The view

The flow's view of a capture, the stage before the motion: the camera image and the aligned depth cut to one
window at the motion's size, the depth read into it, and the player's two masks. What comes before it, the
uploader's corrected camera frame, is in [INPUT_THREADS.md](INPUT_THREADS.md#the-uploader); the sizes it works
at and why are in [RESOLUTION.md](RESOLUTION.md). **(deduction)** marks what is inferred.

The stage is `ViewLayer` (settings `view`, `modules/render/layers/flow/view_layer.py`); its geometry and the
depth's read are plain modules in `modules/flow/view` (`view_window.py`, `DepthUpscale`).

## The stage

Once per capture, in order:

| Step            | Makes                                                                                          |
|-----------------|------------------------------------------------------------------------------------------------|
| the window      | the image cut to the view's fixed window (`fixed_x`, `fixed_y`, `fixed_height`), at the motion's `width` × `height` |
| the depth read  | the aligned depth read into the same window ([The depth read](#the-depth-read)), a `DepthView` ([MOTION.md](MOTION.md#when-it-measures)) |
| the depth mask  | where the view's depth lies from `near` to `far`                                                |
| the mask        | the depth mask, its edge moved onto the image's ([The mask and the depth mask](#the-mask-and-the-depth-mask)) |
| the foreground  | the image where the mask is                                                                     |
| the pose        | the pose's points placed in the window, for the pose drawn over the view                        |

Its outputs are the image, the foreground, the depth, the mask and the depth mask; the panel picks the motion's
picture and the density's among them (`inputs`). The density's color is the foreground's look
(`ForegroundLook`: the blend over time, the cel shade, the hue shift, the sharpening) on the view's foreground
(`foreground_look`), once per capture.

The color is resampled twice, straightened in the uploader and cut to the view here, a little softer than one
resample **(deduction)**, for keeping the camera's work and the flow's apart.

| Setting                              | Preset        | Is                                                                        |
|--------------------------------------|---------------|---------------------------------------------------------------------------|
| `fixed_x`, `fixed_y`, `fixed_height` | 0.5, 0.5, 1   | the window's center and height on the camera image, shares of it          |
| `near`, `far`                        | 0.1, 1.7 m    | the player's depth range, the box's front and back                        |
| `guided_radius`                      | 2             | how far the image moves the mask's edge, in depth pixels; 0 off           |
| `edge_contrast`                      | 0.03          | the image's brightness spread (0 to 1) above which the edge follows the image's |
| `depth_read`                         | `SURFACE`     | how the depth is read into the view                                       |
| `bilateral_contrast`                 | 0.1           | `BILATERAL`'s σr, a color distance on 0 to 1                              |

## The depth read

The aligned depth is coarser than the view (`NEURAL_480`, 480 × 300 on the whole camera image), so each view
pixel reads it between its pixels (`DepthUpscale`). `depth_read` chooses how; one of the three stays
([PLAN.md](PLAN.md#1-the-3d-fluids-input)).

| `depth_read` | Within a surface                  | At an edge, a view pixel takes                              | Source                  |
|--------------|-----------------------------------|-------------------------------------------------------------|-------------------------|
| `SURFACE`    | the four depth pixels interpolated | the depth pixel it lies in: the edge on the depth's pixels | our own rule            |
| `MASK`       | as `SURFACE`                      | the nearest depth pixel on the side the mask gives it       | our own rule            |
| `BILATERAL`  | a weighted mean of the 5 × 5 around it | the same mean, weighed by the image                     | Kopf et al. 2007, eq. 3 |

`SURFACE` (`view_window.edge_aware_depth`): the four depth pixels around a view pixel, interpolated bilinearly
where they lie within the motion's `z.tolerance` of each other, else the one it lies in, so an edge never reads
as a distance in between; a 0, no depth, is never blended.

`MASK` (`mask_sided_depth`) reads as `SURFACE` does wherever that depth is on the pixel's side of the mask: a
depth in `near`..`far` inside the mask, anything else, no depth included, outside it. A pixel whose depth is on
the wrong side takes the nearest depth pixel on its side within `guided_radius`, as that is how far the guided
filter may move the mask's edge off the depth's; with none in reach it keeps its own. The depth's edge and the
depth mask's then lie on the mask's. The mask is made from a depth read, so `MASK` reads twice: `SURFACE`, the
cut and the guided mask, then the read with that mask and the cut again. It carries the guided filter's edge
into the depth without a setting of its own.

`BILATERAL` (`joint_bilateral_depth`) is joint bilateral upsampling (Kopf, Cohen, Lischinski, Uyttendaele 2007,
[the paper](https://johanneskopf.de/publications/jbu/paper/FinalPaper_0185.pdf)). Each view pixel takes the
depth pixels of a 5 × 5 window around it, each weighed by two Gaussians: f, of its distance in depth pixels,
σd = 0.5; and g, of the color distance between the camera image at the view pixel and at that depth pixel's
place, on color 0 to 1, σr = `bilateral_contrast`. The window, σd and the range filter on the high-resolution
image are the paper's. It differs in two places:

| Deviation                                              | Why                                                                      |
|--------------------------------------------------------|--------------------------------------------------------------------------|
| a depth pixel without depth (0) is left out of the sum | the paper's solution is dense; a 0 averaged in pulls an edge toward 0 m  |
| σr is a setting, the paper's 0.1 the default           | the paper sets it per application; 0.1 is its value for stereo depth     |

What each read gives z **(deduction)**:

| Read        | z's edge                | z inside             | Risks                                                              | Cost                                 |
|-------------|-------------------------|----------------------|--------------------------------------------------------------------|--------------------------------------|
| `SURFACE`   | the depth's pixels      | the depth's, interpolated | the depth mask a staircase of depth pixels                    | the read                             |
| `MASK`      | the mask's, the image's | as `SURFACE`         | a depth pixel that straddles the edge is taken whole               | a second read and cut                |
| `BILATERAL` | the image's edges       | close to `SURFACE`'s | the image's texture and noise in z; a depth in between where the image has no edge; the mask moves with the depth mask | 25 depth and 25 image reads a pixel |

z is the depth's change over time per pixel, so a pixel that changes side between two frames reads as z. With
`MASK` no depth from behind the player lies inside the mask, and z's gate cuts where the edge is. With
`BILATERAL` the camera image's noise moves the weights every frame, which moves the depth where nothing moved;
inside the body it gains little, as at σd 0.5 a neighbour one depth pixel away weighs 0.135 of the pixel
itself. No read adds depth detail inside the body: z takes x/y's resolution at the outline and stays the
depth's, interpolated, inside it. A depth pixel's own flicker stays, as a soft bump of its size; removing it
would filter the depth's noise.

Measured on the measure recording (`20261112-125832_measure`), `-sim`, its first 16 s from the in marker (still
1–5 s, the slow move 5.5–10 s, the fast 11.5–16 s), one run each, the motion's thresholds at 0, z in widths/s
at `z_gain` 0.4, every region from `SURFACE`'s mask: noise is the still part's p99 per pixel 6 px and more
inside the mask and within 6 px of its edge; slow and fast the mean inside; fit the image's brightness gradient
along the depth mask's edge, higher the closer it sits on the image's; "no z position" the share of the mask
where the motion's z position has no weight.

| Read        | No z position | Fit    | z noise inside / edge | Slow  | Fast  | Fast ÷ noise | x/y at edge |
|-------------|---------------|--------|-----------------------|-------|-------|--------------|-------------|
| `SURFACE`   | 8.0%          | 0.0413 | 0.183 / 0.134         | 0.016 | 0.028 | 0.153        | 0.124       |
| `MASK`      | 7.7%          | 0.0613 | 0.155 / 0.127         | 0.017 | 0.027 | 0.177        | 0.125       |
| `BILATERAL` | 10.1%         | 0.0590 | 0.138 / 0.115         | 0.015 | 0.024 | 0.177        | 0.159       |

A second `MASK` run gave noise 0.179 / 0.151: the noise differences lie within one run's swing. With `MASK`
the view's depth and the depth mask cover the whole mask, and its share without a z position lies where the
depth velocity's track has no depth: its step filter (`DepthStepFilter`,
[MOTION.md](MOTION.md#depth_velocity-z)) leaves a band a depth sample wide along every outline, whatever the
read. The bridge's outputs take their z positions from the trails, which fill it
([BRIDGE_3D.md](BRIDGE_3D.md#the-z-positions)).

Against reading the nearest depth pixel everywhere, on the stillslowfast recording, 3 runs, the motion's
thresholds at 0, `SURFACE` gives: neighbouring pixels inside the body with the very same depth 11% (67%), z
inside the mask p99 0.040 widths/s (0.057), z at the edge 0.041 (0.050), the mask's fit unchanged.

## The mask and the depth mask

The player's shape and where the depth is the player's are two things:

- `mask`: the depth mask, its edge moved onto the camera image's by the guided filter (He 2010), then the AA's
  threshold at 0.5, a crisp edge where the image puts it. Where the image has no edge the filter only averages
  the cut, and the threshold makes that a smoothed outline. `guided_radius` is in the cut's (the depth's)
  pixels, how far its edge may be off, turned into the view's each frame from the window; `edge_contrast` is
  squared inside as the filter's ε. It gates x/y in the motion with `use_mask`, and the dye's color in the
  bridge.
- `depth_mask`, the view's depth cut by `near` and `far`, is the motion's `depth_mask`: it gates z on top of
  the mask, always, and it is the z position's weight with the mask. The z position's value is held to 0..1
  where the track lags past the range.

The dye's outline is the mask's: its color and the x/y of its amount stop at the guided edge. With `SURFACE`
the depth mask is cut along the depth's pixels, a staircase inside the mask's outline, and z's share of the
dye's amount steps along it.

The preset has no erosion, as one pulls the guided edge off the image's. Measured on stillslowfast, the still
part, the motion's thresholds at 0; z and x/y in widths/s, p99 within 6 px of the mask's edge and inside it;
fit along the mask's edge. One run's p99 swings by about ±0.02; fit holds to ±0.0003.

| Mask                                | Runs | z at edge (p99 / mean) | z inside | x/y at edge | Fit    |
|-------------------------------------|------|------------------------|----------|-------------|--------|
| the cut, erosion 1, z by the mask   | 3    | 0.053 / 0.0040         | 0.087    | 0.030       | 0.0308 |
| guided 2, erosion 0                 | 3    | 0.050 / 0.0036         | 0.057    | 0.028       | 0.0449 |
| guided 1.5, erosion 0               | 3    | 0.053 / 0.0036         | 0.064    | 0.028       | 0.0435 |
| guided 1.5, erosion 1               | 1    | 0.052 / 0.0041         | 0.063    | 0.025       | 0.0276 |
| guided 1.5, erosion 2               | 1    | 0.053 / 0.0041         | 0.066    | 0.030       | 0.0210 |
| guided 2, erosion 0, contrast 0.015 | 1    | 0.053 / 0.0037         | 0.106    | 0.027       | 0.0450 |
| guided 2, erosion 0, contrast 0.06  | 1    | 0.065 / 0.0041         | 0.078    | 0.027       | 0.0442 |

## Checked

`scripts/view_check.py [--read surface|mask|bilateral]` plays a recording's captures through the uploader and
the view stage and holds each read to its numpy reference:

| Read        | Pixels off the reference, 40 captures of the measure recording                           |
|-------------|-------------------------------------------------------------------------------------------|
| `SURFACE`   | none by over 1 mm                                                                         |
| `MASK`      | none; 0.02% a tie between two depth pixels on the mask's side equally near, either the rule's |
| `BILATERAL` | none by over 2 cm, at most 1.2 cm: the GPU blends the image with coarser weights          |

The mask overlaps the depth mask 98% with `SURFACE` and exactly with `MASK`. Without the app:
`modules/flow/tests/test_view_window.py`, the window's geometry and the three reads.
