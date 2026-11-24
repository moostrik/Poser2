# low_flow's plan

A new app, `low_flow`: equal to deep_flow, on the 2D flow, with a higher-resolution solver. The same camera
(one OAK4, `.venv`), view, pose, monitors and panels; the 2D motion, bridge and fluid in place of the 3D's.
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
harness and seen in low_flow; the march's light, shadows and fog need a depth and stay 3D only:

- **D7.** The particles.
- **D8.** The one-pass buoyancy in -sim: the GPU fluid time (built and checked on the GPU 2026-11-19, as the 3D's:
  one draw against the two it replaced within one step of the velocity's half precision, skipped at gravity 0).
  low_flow's settings file has gravity down the screen at deep_flow's length, (0, −0.02).

Each step updates the documents it makes stale (FLOW.md, MOTION.md, TIMING.md).

## G. The water's speed of sound

A push reaches the whole field at once, as incompressible water does; `pressure.reach` (built, FLUID_3D.md's "The
pressure") limits how far. The water gets a speed of sound too, so a push travels instead of arriving at once.

- **G4.** The water's speed of sound in the 2D, `pressure.sound_speed` (widths/s; 0 = off, the solve as now):
  artificial compressibility (Chorin 1967) with the pressure carried along by the water, the low-Mach form of a
  compressible fluid's pressure: ∂P/∂t + u·∇P = −c² ∇·u, and the velocity takes its gradient, so a push runs outward
  as a front at c, carried a little by the water it runs through (faster with the flow than against it, seen where c
  is near the water's speed); the far corner answers when the front arrives, not weaker, and the water can overshoot
  and swing back, ripples. Stepped explicitly it holds only while a front crosses less than a cell a step, c < 0.056
  widths/s at 60 fps and 1080 wide (the field crossed in 30 s). Stepped by Crank–Nicolson, the trapezoidal step, it
  holds at every speed (1 width/s crosses the field in 1.8 s, 10 in 0.18) and keeps a wave's energy, where backward
  Euler would damp every ripple shorter than about 2π c dt within frames. Its new pressure is the screened solve the
  reach already has, the term 1/(c dt / 2)² on the same diagonal as the reach's (the two add), the same multigrid: on
  the right-hand side the divergence and the last step's pressure carried by the velocity (the advection's shader,
  one pass) with its Laplacian; two passes more than today, no sub-steps. In the code's pressure, which is dt × the
  physical one (the gradient subtracts ∇p with no dt, `gradient.frag`); the carried pressure is rescaled by dt /
  dt_last, as the measured step changes frame to frame. `pressure.fade_time` (s, to 1% as the other fade_times; 0 =
  never) settles the ripples, a term dt/τ on the diagonal, τ = fade_time / ln 100. At c → ∞ it is today's solve; at
  the reach's 0 there is no pressure, so no sound either. Closed walls reflect a front; open walls let it out, the
  textbook's absorbing wall (Engquist and Majda 1977, first order: ∂P/∂t + c ∂P/∂n = 0 on the face, in the solve's
  wall ghost), where the pressure 0 on the face would reflect it inverted. The field's default 1 width/s (the field
  crossed in 1.8 s), the settings files 0, off. FLUID_2D.md says the setting is the 2D's alone for now. Question: the 3D's
  G4, after it is seen in low_flow.

## Checks

| Step     | Check                                                                                               |
|----------|-----------------------------------------------------------------------------------------------------|
| D7       | GPU harness for each new kernel; -sim                                                               |
| D8       | -sim, the GPU fluid time                                                                            |
| G4       | GPU harness, the 2D, the walls closed and open: a point push's front at c × t (1 and 10 widths/s, no sub-steps), carried with a flow; its energy kept with `fade_time` 0, a ripple of 0.1 widths at 10 widths/s surviving; a closed wall reflecting it, an open one letting it out; `fade_time` settling it; a dt changing frame to frame without a kick; c very large the same as today's solve; then in the app |

## Rules it touches

- Keep settings files in sync with code changes: D7, G4.
- Shaders get checked in a GPU harness: D7, G4.
- Prefer the textbook unit for a quantity: G4 in widths/s and s.
- Similar things share the noun and differ by a qualifier: G4 (`pressure.fade_time`).
- Name a concept after what it is in the domain: G4 (`sound_speed`).
- Prefer the design with the lower latency, measured in the context of the app: D7, D8, G4.
- A plan is named after the package, app or folder it is for, and sits in it: this plan, in `apps/low_flow`.
