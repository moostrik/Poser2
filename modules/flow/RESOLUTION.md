# Resolutions

The sizes the flow's picture and motion pass through, from the camera to the 4K output, and why each is what
it is. **(deduction)** marks what is inferred, **(site fact)** what the operator saw. All numbers are
deep_flow's preset. What is next is in [PLAN.md](PLAN.md).

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
| camera depth                 | Neural Depth `NEURAL_480`                   | 480 × 300                  | one depth pixel about 2.7 view rows at the whole image's height |
| the image (the view)         | the view's window, drawn once               | 448 × 800, the motion's    | the window's rows, one to one at most                      |
| the mask                     | the depth's cut, guided onto the image      | 448 × 800                  | edges the image's                                          |
| `depth_mask`                 | the view's depth cut by near and far        | 448 × 800                  | a staircase of depth pixels at the edges                   |
| the depth on the view        | the aligned depth, read across a surface    | 448 × 800                  | smooth within a surface, a depth pixel's step at an edge   |
| x/y                          | the image (Lucas–Kanade, 7 × 7 window)      | 448 × 800                  | smooth over ~7 px                                          |
| z, the z position            | the depth (the depth velocity, its track)   | 448 × 800                  | the depth's                                                |
| the bridge's readings, dye   | x/y, z, the image × the mask                | 448 × 800, per capture     | held between the camera's frames                           |
| the fluid                    | the readings, read bilinearly (1.56× down)  | 288 × 512 × 64             | motion finer than a cell averages away                     |
| the density                  | the dye, color × amount                     | 448 × 800 × 24             | the dye's; the fine density's color, read linearly by the composite: a B-spline read costs 0.75 ms and shows nothing on the hands recording |
| the fine density             | the dye's amount, by a B-spline (`fine_density.cubic`) | 2160 × 3840 × 24 | its own threads, from MacCormack and the turbulence        |

The view's rows, the depth read and the two masks are the view stage's ([VIEW.md](VIEW.md)).

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
the fine density's color and what the buoyancy weighs, by uv. A density at the simulation's 288 × 512 cells
blurs the dye's color **(site fact)**. On the show recording, `-sim`, 40 s, two runs: the fluid 9.6 / 9.1 ms,
its density pass 6.1 / 5.8; a density on the grid beside a color volume takes 10.1 / 9.2 and 6.7 / 5.9.
