# Resolutions

Every size the flow's picture and its motion pass through, from the camera to the 4K output, what each
stage needs, and the plan to give each what it needs at a cost deep_flow can run. "Code" is what the files
show; **(deduction)** marks what is inferred, **(site fact)** what the operator saw; the depth's size is its
model's nominal one.

The camera is a given **(site fact)**: its color and depth sizes are set for USB and network alike, and the
chain makes the most of them; more color detail would not help, a finer depth would, later. The OAK4's
color frame is the P720 sensor mode (1280 × 720) cut to 5:4, 896 × 720 (`delivered_size`,
`modules/camera/oak4/pipeline.py`); its tilt is a homography the host applies where the frame is drawn into
the view (`tilt_correction`), not more rows, so the frame stays 720 rows at any tilt (the old OAK backend,
`modules/camera/oak/`, warps on the camera into a taller frame instead). A recording holds the frame as
delivered, 896 × 720: a replay shows the detail live does.

The rule: a quantity is made at the size of what it comes from and carried at the size where it is used.
The dye's picture, its color and its outline, comes from the camera image and is seen at 4K; the motion's
speed comes from the motion; the water's motion is the fluid's grid. hd_trio's 2D chain keeps it by running
the whole bridge at the output's 2160 × 3840 (its fluid 1088 × 1920), so the dye is drawn at full size. The
3D bridge's filters run a dozen passes a tick, too many at 4K: the 3D chain splits the dye instead, its
picture at full size, its speed from the motion's size.

## Now

The picture, what becomes the dye (deep_flow's preset):

| Stage                                     | Size                                      | What it does to the dye                                     |
|-------------------------------------------|-------------------------------------------|-------------------------------------------------------------|
| camera color                              | 896 × 720 (P720 cut to 5:4), cropped around the player | the real detail: the crop at most 720 rows tall      |
| depth, the player's mask                  | about 480 lines (NEURAL_480), cropped     | the mask's edge in the depth's pixels                        |
| center view, image and mask               | 448 × 800, the motion's size              | drawn once from the crop, read by the flow as made           |
| bridge: the dye, image × mask × speed     | 448 × 800                                 | about all the real detail: 800 rows against at most 720      |
| upscaled density, its color               | 448 × 800 × 24, the dye's size            | the dye's color as made, carried with the amount             |
| upscaled density, its amount              | 2160 × 3840 × 24                          | filled from the 448 × 800 dye, stretched 4.8×                |
| composite                                 | 2160 × 3840                               |                                                              |

The motion, what moves the water:

| Stage                                     | Size                                      | What it does to the motion                                   |
|-------------------------------------------|-------------------------------------------|--------------------------------------------------------------|
| camera color                              | 896 × 720, 50 fps                         | the real detail and the real frame rate                      |
| center view, the motion's input           | 448 × 800                                 | the crop at the motion's size: no resample after it          |
| motion x/y (Lucas–Kanade)                 | 448 × 800, 4 levels, 7 × 7 window         | smooth over about 7 px there, 34 px at 4K                    |
| motion z (the depth velocity)             | 448 × 800, from about 480 depth lines     | coarser than x/y by the depth's pixels                       |
| the mask on the velocity (`use_mask`)     | 448 × 800                                 | the depth's staircase at the motion's size                   |
| bridge readings and mixes                 | 448 × 800, at the 60 Hz render tick       | held between the camera's frames                             |
| fluid: the drag and the body force        | 288 × 512 × 64                            | the motion finer than a cell averaged away                   |
| upscaled density, carried by the fluid    | 2160 × 3840 × 24                          | moved by a velocity 7.5× coarser, linear between its cells   |

The first draw after a start is the 448 × 800 dye with the depth's outline, stretched 4.8×: on the blocky
recording its edge steps 15 to 40 px at 4K **(site fact)**, the depth's pixels (the saved mask holds a
column for 14 to 16 rows of 448 × 800 and jumps 2 to 3).

What has its place: the motion at 448 × 800, finer than the water, which the dye's speed and the readings'
shapes use; the fluid at 288 × 512, the budget; the bridge's dye at 448 × 800, which keeps about all the
camera's real detail (800 rows against at most 720 in the crop). What does not: the mask at the depth's
pixels, and the dye stretched 4.8× from 448 × 800 to 4K, straight between its texels. The water cannot take motion finer than its cell, so a finer motion gains the fluid
nothing; the 4K density's own detail comes from its advection (MacCormack) and the turbulence below the
grid, never from the motion.

## The chain as it should run

Two rules, from the fluid's side. No size above what the information holds: an enlargement only where the
result is seen, at the output, never between stages that read it back smaller. And no detail thrown away
before it is used: a step down of 2× or more averages what it leaves (area, a box or a mip level), never
point- or bilinear-samples a smaller grid out of a larger one, which drops texels and aliases; a step up, at the
output, is smooth (a cubic B-spline), not straight between texels.

| Data                      | Its information           | Made at                         | Read by                           | Resampled                              |
|---------------------------|---------------------------|---------------------------------|-----------------------------------|----------------------------------------|
| the image, for the flow   | the crop, at most 720 rows | 448 × 800, from the camera crop in one draw | the motion, the bridge's dye | once, the crop to 448 × 800            |
| the player's mask         | the depth, ~480 lines     | 448 × 800, guided by that image  | the motion, the bridge's dye      | once, the depth to 448 × 800, then guided |
| the velocity, the readings | the motion               | 448 × 800                        | the bridge, the fluid             | down to the fluid's grid, bilinear      |
| the dye: color and amount | the image × mask × speed  | 448 × 800                        | the fluid's densities             | down to 288 × 512, bilinear; up to 4K, a B-spline |
| the 4K density's color    | the dye's color           | a color field at the dye's 448 × 800, its depth the 4K amount's | the composite | carried with the amount; up to 4K, a B-spline |
| the water's velocity      | the fluid                 | 288 × 512 × 64                   | the 4K density's advection        | read smoothly (tricubic), weighed by cost |

- **The center view at the motion's size.** The crop is there for the fluid **(site fact)**; the center
  layers' other readers, the debug panels and the color mask over the composite, are for debugging. So
  deep_flow allocates its center layers (`CENTER_LAYERS` in `apps/deep_flow/render.py`) at the motion's
  `width` and `height` and follows them when they change: the crop is drawn once, at the size the motion's
  luminance (`LuminanceFrames`) and the bridge read it, where a larger view read back bilinearly 2.4× down
  would hand the solve an aliased image; the guided filter runs at that size (`guided_radius` is in widths);
  and the debug views show the same detail enlarged on screen, as the crop holds no more. The center layers
  take 0.18 ms of GPU a frame at 448 × 800 against 0.46 ms at 1080 × 1920 (`GPU center`, deep_flow's
  preset with the guided filter off, 40 s of `-sim`). The guided filter alone, median of 40 on its own:

  | `guided_radius` | 1080 × 1920 | 448 × 800 |
  |-----------------|-------------|-----------|
  | 0.01 widths     | 0.22 ms     | 0.05 ms   |
  | 0.02 widths     | 0.35 ms     | 0.06 ms   |
  | 0.03 widths     | 0.47 ms     | 0.07 ms   |
- **The 4K density's color at the dye's size.** The upscaled density keeps its color in a volume of its own at
  the dye's width and height by the amount's layers (`upscaled.color` DYE, [FLUID_3D.md](FLUID_3D.md#the-densitys-detail)),
  448 × 800 × 24 at the preset, 8.6 M texels against the amount's 199 M, and costs 1.3 ms of the density pass;
  the simulation density's 288 × 512 color (SIMULATION) stays as the cheaper choice. The composite reads the
  color linearly between its texels.

What changes against now, and why:

- **Into the fluid, bilinear as now.** The fluid reads the bridge's 448 × 800 outputs at its 288 × 512 cells
  bilinearly (`inject_3d`, `add_to_faces_3d`). At 1.56× that drops little: the velocity and the readings are
  smooth over the motion's 7 × 7 window and the lowpass's blur, and the 4K amount reads the dye at 448 × 800
  itself (`inject_amount_3d`), not through the grid **(deduction)**.
- **Up once, smoothly, at the output.** The 4K amount is filled by a B-spline already (`upscaled.cubic`); the
  composite's read of the color the same, in place of its linear read, if the look gains for its cost.
- **The velocity for the 4K density.** Read straight between its 288 × 512 cells, its kinks show as the
  coarse cells in a sharp density; a tricubic read where the amount is carried, timed against today's. It
  is 8 trilinear reads for 1, at each of the amount's 199 M texels a step, MacCormack's two reads twice
  that **(deduction)**.

## Plan

### 1. The velocity for the 4K density

A tricubic read of the 288 × 512 velocity where the 4K amount is carried; kept if the look gains for its
cost.

## Open

- The whole chain before the fluid, camera to bridge (the center layers, the mask, the motion, the bridge),
  checked for performance as one: each pass timed at deep_flow's preset, and what it costs against what it
  gives the fluid.

- A finer depth would give the mask a finer outline at the source; the camera stays as it is now.
