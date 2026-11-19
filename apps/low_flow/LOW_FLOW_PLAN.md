# low_flow's plan

A new app, `low_flow`: equal to deep_flow, on the 2D flow, with a higher-resolution solver. The same camera
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
harness and seen in low_flow; the march's light, shadows and fog need a depth and stay 3D only:

- **D7.** The particles.
- **D8.** The one-pass buoyancy in -sim: the GPU fluid time (built and checked on the GPU 2026-11-19, as the 3D's:
  one draw against the two it replaced within one step of the velocity's half precision, skipped at gravity 0).
  low_flow's preset has gravity down the screen at deep_flow's length, (0, −0.02).

Each step updates the documents it makes stale (FLOW.md, MOTION.md, TIMING.md).

## G. The pressure's reach

Since the multigrid (D3) the pressure is solved over the whole grid each step: a push in one corner moves the water
in the other at once, as in the 3D. With the walls closed the water pushed away comes back round through the whole
box **(deduction)**. The solve then needs the divergence to sum to 0, and the velocity's own does, whatever the
injections add to it: its differences cancel cell to cell and the walls' ghosts carry nothing through (FLUID_2D.md,
FLUID_3D.md's "The expansion") **(deduction, from the divergence's code)**. The expansion is the one net source, its
mean taken off first.

- **G1.** The 2D's expansion mean exact, as the 3D's: today it is the expansion's top mip level (`uMeanLod`,
  `Fluid2D._enforce_incompressibility`), a half float that counts the obstacles' cells. A mip chain of a size that is
  not a power of two (1080 × 1920) averages unevenly, so that level is not the mean and the source does not sum to 0
  **(deduction, from OpenGL's mip rule; checked in G1's harness)**. `ExpansionMean2D`, the 2D's own twin of
  `ExpansionMean3D` (no 2D sum exists to reuse): each workgroup sums its water cells' expansion and counts them into
  a buffer, one group sums the partials into the mean, a float, nothing read back; the obstacles' cells left out. It
  runs only while the expansion is on (off in low_flow's preset), so it is not what moves the far corner today.
- **G2.** `pressure.reach` (widths, 0 to 2 by 0.01; 0 = off, the water incompressible, as now): the pressure solves
  the screened Poisson equation (∇² − 1/reach²) p = div, so a push's pressure falls off as K₀(r/reach) in the 2D and
  e^(−r/reach)/r in the 3D, the screened equation's Green's functions. The water is then no longer incompressible:
  past the reach it gives a little instead of passing the push on, a knob for the look, not a fix. The term 1/reach²
  goes on the diagonal of every solver and the residual, the same on every multigrid level (it is in widths): the
  2D's `JacobiPressure`, `JacobiPressureCompute`, `SorPressure2D`, `CoarsePressure2D`, `ResidualRestrict2D`, and the
  3D's `jacobi_pressure_3d`, `SorPressure3D`, `CoarsePressure3D`, `ResidualRestrict3D`, their `.py` wrappers (β =
  1/(2αx + 2αy + 1/reach²)), `MultigridPressure2D` and `multigrid_pressure.py`, and `Fluid2D` and `Fluid3D`, which
  pass the reach on and decide the expansion's mean; the gradient stays. Above 0 the solve with closed walls is no
  longer singular: the pressure's constant cannot drift, and the expansion's mean comes off only at reach 0 with the
  walls closed (else it adds a uniform sink). Both presets take 0; FLUID_2D.md and FLUID_3D.md's "The pressure"
  describe it, FLUID_3D.md's divergence-after rows and both documents' "The expansion" numbers holding at reach 0.
  The 3D's solvers are Flow Fluid's, asked for feedback.

## Checks

| Step     | Check                                                                                               |
|----------|-----------------------------------------------------------------------------------------------------|
| D7       | GPU harness for each new kernel; -sim                                                               |
| D8       | -sim, the GPU fluid time                                                                            |
| G1       | GPU harness, the walls closed, the expansion on, 1080 × 1920: the mean in its float buffer against the expansion's sum; the divergence's sum within its half float's rounding, before and after |
| G2       | GPU harness, the walls closed, 2D and 3D, each solver: a point source at the box's middle at reach 0.05, its pressure against K₀(r/reach) (2D) and e^(−r/reach)/r (3D) where it is a few reaches from every wall, at reach 0 the result the same to the bit before and after, no drift with the walls closed; the settings and preset tests; then in the app, the walls closed |

## Rules it touches

- Keep presets and settings schemas in sync with code changes: D7, G2.
- Shaders get checked in a GPU harness: D7, G1, G2.
- Prefer the design with the lower latency, measured in the context of the app: D7, D8, G1, G2.
- Prefer the textbook unit for a quantity: G2 (the reach in widths).
- The 2D's panel is the 3D's, each setting at the 3D's path: G2.
- Fix a problem where it is made, not where it shows: G1.
- A plan is named after the package, app or folder it is for, and sits in it: this plan, in `apps/low_flow`.
