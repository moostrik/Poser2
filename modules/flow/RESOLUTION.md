# Resolutions

Every size the flow's picture and its motion pass through, from the camera to the 4K output, what each
stage needs, and the plan to give each what it needs at a cost deep_flow can run. "Code" is what the files
show; **(deduction)** marks what is inferred, **(site fact)** what the operator saw; the camera's and the
depth's sizes are their models' nominal ones.

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
| camera color                              | 1280 × 720 (P720), cropped around the player | the real detail, about 700 to 900 px tall in the crop **(deduction)** |
| depth, the player's mask                  | about 480 lines (NEURAL_480), cropped     | the mask's edge in the depth's pixels                        |
| center view, image and mask               | 1080 × 1920                               | upscaled to here: no new detail                              |
| bridge: the dye, image × mask × speed     | 448 × 800                                 | the 1920 image cut back to 448                               |
| simulation density, its color             | 288 × 512 × 64                            | the 4K density's color                                       |
| upscaled density, its amount              | 2160 × 3840 × 24                          | filled from the 448 × 800 dye, stretched 4.8×                |
| composite                                 | 2160 × 3840                               |                                                              |

The motion, what moves the water:

| Stage                                     | Size                                      | What it does to the motion                                   |
|-------------------------------------------|-------------------------------------------|--------------------------------------------------------------|
| camera color                              | 1280 × 720, 50 fps                        | the real detail and the real frame rate                      |
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
shapes use; the fluid at 288 × 512, the budget. What does not: the dye cut to 448 × 800, and the mask at the
depth's pixels. The water cannot take motion finer than its cell, so a finer motion gains the fluid
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
0.02, 0.47 at 0.03. On the first draw it is not yet seen, as the dye is cut to 448 × 800 after it (step 2).
The motion takes the same mask.

### 2. The dye at full size

The bridge's dye, the image × the mask, at the view's 1080 × 1920 (or the upscaled density's size) instead
of the bridge's 448 × 800; only its amount, the readings' speed, from 448 × 800, read smoothly (a cubic
B-spline) where the dye is made. One pass at the larger size; the filters stay at 448 × 800.

### 3. The 4K density's color from the dye

The 4K density takes its color from the 288 × 512 simulation density. With the dye at full size (2), the
color can follow it at injection, in a color field finer than the simulation's (a middle size, e.g. 1080 ×
1920 × 24) or the amount's own channels; measured against the memory and the density pass's time.

### 4. The camera

P1080 gives real detail where upscaling gives none; a setting to try, at the camera pipeline's cost.

### 5. The water at the 4K density

The 4K density moved by the 288 × 512 velocity read linearly, straight between its cells: a smooth
(tricubic) read of the velocity where the amount is carried, weighed against its cost (eight reads where one
is now).

## Open

- Where the mask is made (the depth's size) against the view: the guide moves the edge at the view's size;
  a mask cut at the image's size from the start is an option once the camera runs as an OAK App.
