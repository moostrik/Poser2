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
superficial_flow's chain is timed once, when the plan is built, its rows in TIMING.md; that time sets the
grid's size.

The 2D fluid's grid is HD by default: 1080 × 1920 (9:16), half the output's 4K, so each cell holds 2 × 2 of the density's pixels; about 18 times the
3D's 256 × 448 cells per layer. Its density is at the output's 2160 × 3840, carried by the grid's velocity, as
`Fluid2D` already holds it (its density at the output's size, apart from the grid's). The grid is set larger or smaller by the times after each fluid step; the multigrid pressure (D3) is what
keeps it fast, its coarse levels rounded up where a side halves to an odd size (1080 to 135 after three).

hd_trio moves onto this 2D flow later, in a plan of its own, so the 2D stays built for it: several cameras, an
injection per camera with its weight (the player's motion, the similarity) and its own color (the track
color), one fluid per screen. Until then hd_trio runs its own copy, unchanged.


## D. The 2D fluid

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
| D2       | the settings test: shared settings with deep_flow's values                                          |
| D3 to D7 | GPU harness for each new kernel; -sim                                                               |
| End      | the full suite once; the inference at 50 fps, the render at 60; deep_flow's times                |

## Rules it touches

- Remove obsolete pathways; no parallel legacy ones: D2.
- One term everywhere; when a term changes, rename it everywhere in the same change: D2.
- Keep presets and settings schemas in sync with code changes: D2 to D7.
- A settings class mirrors the code it configures; a stage's ports follow its steps: D2.
- Shaders get checked in a GPU harness: D3 to D7.
- Prefer the design with the lower latency, measured in the context of the app: D2 to D7.
- A document describes what is built: D2 (FLUID_2D.md).
- A plan is named after the package, app or folder it is for, and sits in it: this plan, in `apps/superficial_flow`.
