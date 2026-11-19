# superficial_flow's plan

A new app, `superficial_flow`: equal to deep_flow, on the 2D flow, with a higher-resolution solver. The same camera
(one OAK4, `.venv-oak4`), view, pose, monitors and panels; the 2D motion, bridge and fluid in place of the 3D's.
It is to the 2D flow what deep_flow is to the 3D, the place it is built and tried. The app comes first, without
the flow, so each 2D stage is tried in it as it is built. Both apps take the flow plan's `color` stage and the
bridge's `filters` group (its D8, D9, built).

The rule: the 2D and the 3D are separate implementations, each with its own kernels, textures and classes; the 2D's
kernels carry no z (RG velocities, no z positions), so they cost only what x/y costs. They share the contracts: the
stage layers' base, the inputs and outputs, the settings' shape and names, the Python both run around their
kernels (in an ABC, as a family of classes that share code), and kernels that hold no z already.

| Stage  | Shared                                                                                  | 2D: its own                                         | 3D: its own                                          |
|--------|-----------------------------------------------------------------------------------------|-----------------------------------------------------|------------------------------------------------------|
| view   | none: each app its own; the flow takes what `MotionInput` and `BridgeInput` hold        |                                                     |                                                      |
| motion | `MotionLayer`, its input and output; the x/y solve (`LucasKanadeFlow`, z-free)          | `LucasKanadeFlow2D`: compose, mask, guard, RG       | `LucasKanadeFlow3D`: z, z position, `smooth_like_z`  |
| bridge | `BridgeLayer`, its input and output; the ABC `Bridge` with the readings' mixing, the ports, the sizes; the settings; `FrameMax`, `decay_weight`, the blur's weights | `Bridge2D`: lowpass, bandpass, mix, peak kernels, RG, no z positions | `Bridge3D`: its kernels as they are |
| fluid  | `FluidLayer`, its input and what each injection means; the panel's groups and names; `TimeStep`; the blend and LUT | `Fluid2D` and its injection kernels | `Fluid3D`, the march |

What goes, from `modules` only (hd_trio's copies stay): the 2D fluid's density channel per injection,
`DensityColorize` and `Injection.tint`.

Latency is the first measure in both apps. The camera stays at 50 fps (the 50 Hz mains in Europe): the boxes and the
segmentation keep up with it (20 ms a frame), the pose at its interval (25 fps), the render with the flow at HD
aims at 60 fps (16.7 ms); falling short is no blocker for now; the inference moves to the camera later, out of this plan. A step adds no frame of delay: the motion and the bridge run once per
capture, the fluid once per tick, as in deep_flow. deep_flow's kernels are not changed by any step.

The 2D fluid's grid is HD by default: 1080 × 1920 (9:16), half the output's 4K, so each cell holds 2 × 2 of the density's pixels; about 18 times the
3D's 256 × 448 cells per layer. Its density is at the output's 2160 × 3840, carried by the grid's velocity, as
`Fluid2D` already holds it (its density at the output's size, apart from the grid's). The grid is set larger or smaller by the times after each fluid step; the multigrid pressure (D3) is what
keeps it fast, its coarse levels rounded up where a side halves to an odd size (1080 to 135 after three).

hd_trio moves onto this 2D flow later, in a plan of its own, so the 2D stays built for it: several cameras, an
injection per camera with its weight (the player's motion, the similarity) and its own color (the track
color), one fluid per screen. Until then hd_trio runs its own copy, unchanged.


## D. The 2D fluid

The 3D fluid's features that have a meaning on a plane, each as the 2D's own kernels, each checked in the GPU
harness and seen in superficial_flow; the march's light, shadows and fog need a depth and stay 3D only:

- **D7.** The particles.
- **D8.** The one-pass buoyancy checked (built 2026-11-19, as the 3D's, the flow plan's E9: `Buoyancy` adds
  dt × gravity × (sink × amount − buoyancy × warmth) straight into the velocity's swap, the warmth read only while
  the lift runs, skipped when the sink and the lift are both 0 or gravity is 0, the force itself only for its
  monitor). superficial_flow's preset has gravity (0, 0), so the pass is skipped there.

  Question: gravity in superficial_flow's preset. `gravity_x`/`_y` are a direction (−1 to 1), the sink and the
  buoyancy carry the unit (widths/s²); deep_flow's is 0.02 along z, which has no meaning on a plane. Down the screen
  at deep_flow's length, (0, −0.02), or the 2D's default, (0, −1)?

Each step updates the documents it makes stale (FLOW.md, MOTION.md, TIMING.md).

## E. The mask finished when it is handed over

The GPU waits up to about 8 ms of each capture on the segmentation before it starts the view, the motion, the
bridge and the fluid (the GPU view timer about 2.9 ms; deep_flow's 0.19). `Segmenter` waits for the model on its own
thread, then queues the mask's resize back to the crop and the foreground (crop × mask) and calls its callbacks at
once; the render's stream waits on the mask's `ready` event (`ViewMask.place`) and the PBO's unmap
(`modules/gl/tensor.py`) makes GL wait for that CUDA work. Those steps run when the GPU gets to them, behind the
pose model **(deduction)**. With a heavier fluid or render the wait would cut the frame rate; it adds latency to the
flow either way.

- **E1.** `Segmenter` waits for its `written` event on its own thread before its callbacks, so the inference
  frame sync hands over only masks that are finished on the GPU; the event stays, waiting on it costs nothing. The
  inference frame sync's `mask_wait` grows by those steps, about what the GPU waits now, and the render stops waiting
  **(deduction)**. hd_trio, which runs the same `Segmenter` (its mask and foreground layers), gets it too.
- **E2.** `CropExtractor` waits for its `crops_ready` on its own thread before its callbacks, for the same reason (the
  crop debug layer waits on it in every app); the pose model and the segmentation wait on it on the GPU already.

deep_flow and white_space hand GL nothing unfinished: their images and depth come from the uploader, which waits on
its own thread (`image_upload.py`); smile hands GL no CUDA result.

## Checks

| Step     | Check                                                                                               |
|----------|-----------------------------------------------------------------------------------------------------|
| D7       | GPU harness for each new kernel; -sim                                                               |
| D8       | GPU harness against the two-pass result; -sim, the GPU fluid time before and after                  |
| E1, E2   | -sim: the GPU view timer, the inference frame sync's `mask_wait`, the flow's latency, before and after; hd_trio's segmentation layers still show |

## Rules it touches

- Keep presets and settings schemas in sync with code changes: D7, D8.
- Shaders get checked in a GPU harness: D7, D8.
- Prefer the design with the lower latency, measured in the context of the app: D7, D8, E1.
- The part that works on the item has its own thread and does all its work there, its inner steps included: E1, E2.
- An item handed to a callback is not changed afterwards: E1, E2.
- A plan is named after the package, app or folder it is for, and sits in it: this plan, in `apps/superficial_flow`.
