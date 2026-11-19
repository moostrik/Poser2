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

What goes, from `modules` only (hd_trio's copies stay): `GradientFlow` and its shader; `bridge_2d`'s old classes
(`SmoothTrail`, `DensityBridge`, `TemperatureBridge`, its `VelocityMagnitude` copy), replaced by `Bridge2D`; `TintColor` and
`BridgeInput.tint`, as the 2D bridge takes the color stage's color as the 3D does; the 2D fluid's density channel per
injection, `DensityColorize` and `Injection.tint`.

Latency is the first measure in both apps. The camera stays at 50 fps (the 50 Hz mains in Europe): the boxes and the
segmentation keep up with it (20 ms a frame), the pose at its interval (25 fps), the render with the flow at HD
aims at 60 fps (16.7 ms); falling short is no blocker for now; the inference moves to the camera later, out of this plan. A step adds no frame of delay: the motion and the bridge run once per
capture, the fluid once per tick, as in deep_flow. deep_flow's kernels are not changed by any step; its bridge, whose
Python moves into the ABC at C1, is timed before and after C1, back to back ([TIMING.md](../../modules/flow/TIMING.md)), and may not
grow. superficial_flow's chain is timed once, when the plan is built, its rows in TIMING.md; that time sets the
grid's size.

The 2D fluid's grid is HD by default: 1080 × 1920 (9:16), half the output's 4K, so each cell holds 2 × 2 of the density's pixels; about 18 times the
3D's 256 × 448 cells per layer. Its density is at the output's 2160 × 3840, carried by the grid's velocity, as
`Fluid2D` already holds it (its density at the output's size, apart from the grid's). The grid is set larger or smaller by the times after each fluid step; the multigrid pressure (D3) is what
keeps it fast, its coarse levels rounded up where a side halves to an odd size (1080 to 135 after three).

hd_trio moves onto this 2D flow later, in a plan of its own, so the 2D stays built for it: several cameras, an
injection per camera with its weight (the player's motion, the similarity) and its own color (the track
color), one fluid per screen. Until then hd_trio runs its own copy, unchanged.


## A. The app

- **A2.** The 2D view and the color: the view, the app's own, deep_flow's fixed window on the camera image
  (`crop.center_x`, `center_y`, `height`) without the depth's `near` and `far`, a full mask until A5, its monitor;
  the window stays put, so the motion measures the player, not the window. It hands the flow what `MotionInput`
  and `BridgeInput` hold; those dataclasses are the contract (no view `Protocol`). The color stage as deep_flow's.
  Checked: superficial_flow in -sim, the view and color monitors showing.
- **A3.** The boxes every frame, the poses at their interval: `pose_interval` moves from the frame source
  (`TrackerFrameSource`) to the pose model's input. The frame source feeds `RAW` every frame, so its box steps
  (`BBoxEuroSmoother`, `BBoxPredictor`) and the crop step run every frame; the crop step hands the pose model
  every `pose_interval`-th crop, and what follows the pose model (the rest of `RAW`, `CLEAN`, `SMOOTH`) runs only
  on a new pose. The asked callback, which tells the capture sync whether a frame's poses were asked, moves with
  the interval; the dense optical flow (RAFT), also fed by the crop step, takes the pose's interval. No new stage;
  the apps' pipeline specs stay as they are; the frame source's tests change with it. Checked: the pose and
  tracker tests; deep_flow in -sim, its poses as before (25 fps at 50 fps and interval 2).
- **A4.** The segmentation every frame (RVM, `modules/inference/segmentation`), before the pose: the crop step
  cuts a second size, `CropExtractor(pose_crop, segmentation_crop)` as hd_trio's, every frame, and the
  `Segmenter` runs on those crops. Its result carries the box it was cut at (added where the crop is cut; today
  it carries none). Its own lane in the time series. Checked: RVM's time per frame alone
  (`scripts/model_benchmark.py`), first, as nothing in the repository measures it yet; superficial_flow in -sim,
  a mask every frame; its time from the lane.
- **A5.** The mask in the view: the segmentation is the capture sync's base, made with `(Masks,)`: a frame is its
  image with the mask and foreground made from it (`Masks.by_camera`). `mask_wait` (8 ms in deep_flow's preset) is
  set from A4's measured time, so a frame does not go out with an older mask. The poses are not in the sync: the
  interpolated poses belong to the render's tick, not a capture, and go only to the pose monitor and the sound;
  poses from the camera, once detection moves there, join the sync as `Poses`. The board's segmentation images
  and hd_trio's mask and foreground layers are not used. The color stage takes the frame's foreground with its
  mask, made together so they match pixel for pixel, placed in the fixed window at the box they were cut at,
  where they are in the camera image; the motion takes the mask. Checked: superficial_flow in -sim on
  `measure`, the mask on the player and the color cut by it.

## B. The 2D motion

- **B1.** The 2D motion. `LucasKanadeFlow2D`, its own class and compose kernel: the x/y velocity in widths/s,
  RG16F, from the x/y solve, its threshold, `use_mask`, `x_gain`, `y_gain` and its guard; no z, no z position.
  `Motion2DLayer` on it; `GradientFlow` goes. Both motions take the one x/y solve (`LucasKanadeFlow`): it reads
  only the two luminance frames and holds no z. Its one tie to z is a setting, `smooth_like_z` (x/y through z's
  smoothing), read only by the 3D's compose; it moves out of the shared `xy` group into the 3D's own (deep_flow's
  preset: `xy.smooth_like_z` to `smooth_like_z`), so the 2D's panel shows no z. With it change the flow layers'
  tests (`modules/render/modules/flow/tests`: `test_flow_systems`, `test_layer_names`), `scripts/shader_check.py`
  and the exports of `modules/flow` and `modules/render`. Checked: those tests; superficial_flow with the motion
  wired, its monitor showing the velocity.

## C. The 2D bridge

- **C1.** The 2D bridge. A package `modules/flow/bridge/` holds what both bridges share: the ABC `Bridge` (the
  readings' mixing, the frames' merge, the ports, the sizes), the filters' and mixes' settings, `FrameMax`,
  `decay_weight` and the blur's weights, out of `bridge_3d`; `Bridge3D` on it with its kernels as they are.
  `Bridge2D` in `bridge_2d`, on the same ABC: its own lowpass, bandpass, mix and peak kernels on an RG velocity,
  no z positions, the density's color the color stage's; the outputs the 3D's (velocity, acceleration, density with its color, temperature, expansion), the
  same groups and names on the panel. `Bridge2DLayer` on it. BRIDGE_2D.md written, BRIDGE_3D.md for what moved.
  The old classes', `TintColor`'s and `BridgeInput.tint`'s call sites change with it: the flow layers' tests,
  `scripts/shader_check.py`, the exports of `modules/flow` and `modules/render`.
  Checked: the new kernels in the GPU harness; the tests; deep_flow's bridge timed and looking as before;
  superficial_flow with the bridge wired, its monitor showing the readings.

## D. The 2D fluid

- **D1.** The 2D fluid in the app, wired on the motion and the bridge (one injection per bridge, its monitors, the
  flow's lanes in the time series), and taking the 3D's inputs ([FLUID_3D.md](../../modules/flow/FLUID_3D.md#the-inputs)): the boundary by a drag
  (`velocity_drag`), the body force's acceleration × dt, the density over, added or filled with its color, in RGB
  or Oklab, its release, the temperature by a drag, the expansion as a source in the pressure solve; the density
  held as its color times its coverage and drawn through the LUT at the panel's blend, as the 3D's. The injection
  passes are its own 2D fragment shaders. Checked: the injection passes in the GPU harness, then superficial_flow in
  -sim on `measure`, the fluid moving with the player.
- **D2.** The 2D fluid's panel as the 3D's, for what the 2D has: `guard`, `wall`, `width`, `height` (rounded up to a multiple of 8, not 32); `time`
  (`time_scale` added); `input` (gravity in it); `velocity` (`advection_scale` for `self_advection`, `evaporation_time` and
  `evaporation_knee` for `dampen_*`, the viscosity in widths²/s); `vorticity`; `temperature` (`enabled` first);
  `pressure`; `density` (`color_space`, `sink`, `advection_scale` for `speed_offset`); each step's `enabled` as the
  3D has. Each by the 3D's law where the 2D's differs. The pressure's advection (not in the textbook) and `profile`
  go. The values, as defaults and in superficial_flow's preset: deep_flow's for every setting the two share (its
  drags, gains, blend, color space, sink, viscosity, vorticity, temperature, pressure); a textbook default where
  deep_flow has none or its value has no meaning on a plane (the sizes, the gravity along z). The 3D features'
  settings take deep_flow's values as they come (D3 to D6). FLUID_2D.md written; `fluid_2d.py`'s pipeline docstring
  goes. Checked: a test that each setting the 2D shares with the 3D has its path, name and unit, and in
  superficial_flow's preset deep_flow's value, unless listed as a textbook default.

The 3D fluid's features that have a meaning on a plane, each as the 2D's own kernels, each checked in the GPU
harness and seen in superficial_flow; the march's light, shadows and fog need a depth and stay 3D only:

- **D3.** The pressure solvers: red-black SOR and the multigrid beside Jacobi, `solver` and `cycles` on the panel.
- **D4.** MacCormack advection with its limiter, for the velocity and for the density.
- **D5.** The walls' slip (`slip_time`).
- **D6.** The three vorticity confinements, each its strength and radius in widths.
- **D7.** The particles.

Each step updates the documents it makes stale (FLOW.md, MOTION.md, TIMING.md).

## Checks

| Step     | Check                                                                                               |
|----------|-----------------------------------------------------------------------------------------------------|
| A2       | -sim: the view and color monitors                                                                   |
| A3       | pose and tracker tests; deep_flow in -sim, its poses as before (25 fps)                             |
| A4       | RVM alone (`model_benchmark.py`); -sim: a mask every frame; its time from its lane                  |
| A5       | -sim: the mask on the player, the color cut by it                                                   |
| B1       | the flow tests; deep_flow in -sim once (`smooth_like_z` moved); superficial_flow's motion monitor   |
| C1       | GPU harness for the 2D kernels; deep_flow's bridge timed before and after, back to back            |
| D1       | GPU harness for the injection passes; -sim, the fluid moving with the player                        |
| D2       | the settings test: shared settings with deep_flow's values                                          |
| D3 to D7 | GPU harness for each new kernel; -sim                                                               |
| End      | the full suite once; the inference at 50 fps, the render at 60; deep_flow's times                |

## Rules it touches

- Update all affected call sites in the same change: A3.
- Use an `ABC` for a family of classes that share code: C1.
- Similar things share the noun and differ by a qualifier (`Bridge2D`, `Bridge3D`): B1, C1.
- Remove obsolete pathways; no parallel legacy ones: B1, C1, D1, D2.
- One term everywhere; when a term changes, rename it everywhere in the same change: B1, D2.
- Keep presets and settings schemas in sync with code changes: B1, D2 to D7.
- A settings class mirrors the code it configures; a stage's ports follow its steps: B1, C1, D2.
- Shaders get checked in a GPU harness: C1, D1, D3 to D7.
- Prefer the design with the lower latency, measured in the context of the app: A4, A5, C1, D1 to D7.
- A document describes what is built: C1 (BRIDGE_2D.md), D2 (FLUID_2D.md).
- A plan is named after the package, app or folder it is for, and sits in it: this plan, in `apps/superficial_flow`.
