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
| center view, image and mask               | 1080 × 1920                               | upscaled at least 2.7× to here: no new detail                |
| bridge: the dye, image × mask × speed     | 448 × 800                                 | about all the real detail: 800 rows against at most 720      |
| simulation density, its color             | 288 × 512 × 64                            | the 4K density's color                                       |
| upscaled density, its amount              | 2160 × 3840 × 24                          | filled from the 448 × 800 dye, stretched 4.8×                |
| composite                                 | 2160 × 3840                               |                                                              |

The motion, what moves the water:

| Stage                                     | Size                                      | What it does to the motion                                   |
|-------------------------------------------|-------------------------------------------|--------------------------------------------------------------|
| camera color                              | 896 × 720, 50 fps                         | the real detail and the real frame rate                      |
| center view, the motion's input           | 1080 × 1920                               | upscaled: no new detail                                      |
| motion x/y (Lucas–Kanade)                 | 448 × 800, 4 levels, 7 × 7 window         | smooth over about 7 px there, 34 px at 4K                    |
| motion z (the depth velocity)             | 448 × 800, from about 480 depth lines     | coarser than x/y by the depth's pixels                       |
| the mask on the velocity (`use_mask`)     | 1080 × 1920, read at 448 × 800            | the depth's staircase at the motion's size                   |
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

## What each needs

| Quantity               | Comes from                         | Needed at                                   |
|------------------------|------------------------------------|---------------------------------------------|
| the mask's outline     | the depth, steered by the image    | the image's size, the view's 1080 × 1920    |
| the dye's color        | the camera image                   | the 4K density's size                        |
| the dye's amount       | the motion's speed (the readings)  | smooth from the motion's 448 × 800           |
| the velocity           | the motion                         | the fluid's grid                             |
| the water's motion     | the fluid                          | its grid; the 4K density moved by it, read smoothly |

## Plan

### 1. The mask's outline from the image

The coarse depth mask's edge moved to where the camera image's edges run, at the view's 1080 × 1920
(guided filter, He et al. 2010; joint bilateral upsampling, Kopf et al. 2007): `center.mask.guided_radius`
(widths, 0 off) and `guided_epsilon`, in `CenterMaskLayer`, guided by the capture image, deep_flow's depth
mask only. Built: the staircase becomes a smooth contour in the saved masks; 0.22 ms at 0.01 widths, 0.35 at
0.02, 0.47 at 0.03. The motion takes the same mask.

### 2. The dye at full size

The bridge's dye, the image × the mask, at the view's 1080 × 1920 (or the upscaled density's size) instead
of the bridge's 448 × 800; only its amount, the readings' speed, from 448 × 800, read smoothly (a cubic
B-spline) where the dye is made. One pass at the larger size; the filters stay at 448 × 800. It adds no
camera detail, as 448 × 800 holds about all of it: what it gains is the guided mask's outline at the view's
size and a smoother enlargement to 4K; weighed by that against step 1 alone.

### 3. The 4K density's color from the dye

The 4K density takes its color from the 288 × 512 simulation density. With the dye at full size (2), the
color can follow it at injection, in a color field finer than the simulation's (a middle size, e.g. 1080 ×
1920 × 24) or the amount's own channels; measured against the memory and the density pass's time.

### 4. The water at the 4K density

The 4K density moved by the 288 × 512 velocity read linearly, straight between its cells: a smooth
(tricubic) read of the velocity where the amount is carried, weighed against its cost (eight reads where one
is now).

## Open

- A finer depth would give the mask a finer outline at the source; the camera stays as it is now.

- Where the mask is made (the depth's size) against the view: the guide moves the edge at the view's size;
  a mask cut at the image's size from the start is an option once the camera runs as an OAK App.
