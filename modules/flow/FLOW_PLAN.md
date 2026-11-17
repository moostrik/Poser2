# The flow's plan

What is next for the flow, the one place that says it: the main lines (A to J), along the flow's chain (the view,
the motion, the bridge, the fluid's physics, its look), then those across the stages; each line's steps in the
order they are worked (B1, B2, ...), what is not decided as a question; what is parked. A built step leaves the
plan; the others keep their numbers. How each stage works is in its own document ([FLOW.md](FLOW.md) lists
them); a step links to the section it changes. **(deduction)** marks what is inferred, **(site fact)** what the
operator saw.

## Main lines

| #   | Line                                                                         | State                            | Documents                                          |
|-----|------------------------------------------------------------------------------|----------------------------------|----------------------------------------------------|
| A   | [The measurements](#a-the-measurements)                                      | built                            | [FLOW.md](FLOW.md#the-goals)                       |
| B   | [The view](#b-the-view)                                                      | built                            | [VIEW.md](VIEW.md)                                 |
| C   | [The motion](#c-the-motion)                                                  | built; C2 the values tonight     | [MOTION.md](MOTION.md)                             |
| D   | [The bridge](#d-the-bridge)                                                  | built; D3 at the screen          | [BRIDGE_3D.md](BRIDGE_3D.md)                       |
| E   | [The 3D fluid's physics](#e-the-3d-fluids-physics)                           | the temperature                  | [FLUID_3D.md](FLUID_3D.md)                         |
| F   | [The 3D fluid's look at 4K](#f-the-3d-fluids-look-at-4k)                     | open                             | [FLUID_3D.md](FLUID_3D.md), [TIMING.md](TIMING.md) |
| H   | [One unit across the stages](#h-one-unit-across-the-stages)                  | not started                      | [FLOW.md](FLOW.md)                                 |
| I   | [The 2D flow follows the 3D](#i-the-2d-flow-follows-the-3d)                  | not started                      | [FLOW.md](FLOW.md)                                 |
| J   | [Every setting and its description](#j-every-setting-and-its-description)    | last, after the refactor         | `CLAUDE.md`, `.claude/rules`                       |

D3 and E2 need the operator at the screen; the other lines go on meanwhile.

The apps on deep_flow's input have their own plan, [apps/APPS_PLAN.md](../../apps/APPS_PLAN.md); recording and
-sim, deep_flow's live checks among them, theirs, [RECORDING_PLAN.md](../recording/RECORDING_PLAN.md). The
chain's timing is in [TIMING.md](TIMING.md) and [INPUT_THREADS.md](INPUT_THREADS.md#the-cadence).

The 3D fluid's lines are weighed by [its goal](FLUID_3D.md#the-goal): sharp, translucent ink patterns, layers
distinct.

## A. The measurements

Built: the rules in `.claude/rules/flow.md`, "Flow measurements"; the motion's measuring, its recordings and its
tables in [MOTION.md's settings](MOTION.md#the-settings); the view's baseline in [VIEW.md](VIEW.md#the-depth);
the recordings gone and what that means for their tables in [FLOW.md](FLOW.md#recordings-gone).

Further measurements on the motion are free where a result points to something (Matthias, 2026-11-16), each
stated before it is made.

## B. The view

Its goal and what is built: [VIEW.md](VIEW.md). New points go here.

## C. The motion

From line A's measurements (four recordings, every frame) and the short runs after it (one capture of hands and of
measure): z needs smoothing to be quiet, and what cuts a large, fast z most is a long smooth (kept 84% at 25 ms,
43% at 100), which the catch-up gives back (82 to 85%) at almost no noise at rest; the dead zone costs z and lag
with smoothing and is the quiet one only without; `tolerance` and `same_surface` show no effect; the x/y solve,
the step filter and z's window earn their place. The aim: quiet at rest, the most z kept, the least lag.

Built (2026-11-17): the 1€ filter's rule in place of the catch-up; one setting per job (`edge` within a frame,
`max_speed` between frames, `xy_rest` and `xy_halving` for z's smoothing, `tolerance` and `same_surface` gone);
described in [MOTION.md](MOTION.md#z-the-depth-velocity) with their tables.

Tried and undone (2026-11-17): the dead zone before the filter, a backlash on the measured depth. Quieter at
rest and later: the filter sees nothing until the zone opens, so its lag comes after the zone's where the two
overlap now (hands, unsmoothed, `dead_zone` 0.02: 52 ms against 7; `smooth` 25: 90 against 65; measure's still
noise 2.4 / 3.1 against 9.1 / 5.1).

- **C2.** The values, one run on the four recordings unattended, then judged by Matthias in the app. The settings'
  defaults and MOTION.md's reasons follow on his word; the preset when he saves it. No new recording.
  - The surfaces that are not the player left out: a still surface inside the mask that wanders, as the shiny desk
    in a corner does, reads as z and is no measure of the motion. Each recording's frames are looked at first; a
    surface in the mask that is not the player gets a box (uv) in `exclude` in the recording's `segments.json`,
    and the compare measures no pixel inside it. The recordings' still stretches are looked at for the same.
  - Each row one change from the defaults, on one capture of each recording (`hands_1`, `measure_1`, `edges_1`,
    `show_1`; three agreed within 0.1), the code at the commit of the run, `dead_zone` 0:

    | Group           | Rows                                                                                   |
    |-----------------|----------------------------------------------------------------------------------------|
    | the reference   | raw (`smooth` 0, the thresholds 0, `max_speed` 0); the defaults                       |
    | z's smoothing   | `smooth` 50, 100, 200 × `xy_halving` 0.25, 0.5, 1 (9); `xy_rest` 0.25, 1               |
    | z's cut         | `threshold` 0.25, 1, 2                                                                 |
    | z's surface     | `edge` 0.05; z's `window` 9×9; `max_speed` 4, 8                                        |
    | x/y             | `levels` 5; x/y's `window` 9×9; `damping` 0.02, 0.045; `threshold` 0.25, 0.4; `z_smooth` on |

    About 26 rows a recording, about 45 minutes in all. Each row: z's still noise inside and at the edge, z kept
    and lag, x/y's still noise, kept and lag, the restarts at rest and moving, the motion's ms.
  - The outcome, a table a recording into MOTION.md, and a proposal: the values quiet at rest (still noise near
    0 after the cut) with the most z kept at the least lag on all four; a value that holds on one recording only
    is named so. Matthias judges it in the app before a default changes.

## D. The bridge

What is built: [BRIDGE_3D.md](BRIDGE_3D.md), and what the fluid does with it,
[FLUID_3D.md](FLUID_3D.md#the-inputs).

Seen in the app, 2026-11-17: the stop as an acceleration gives nice effects **(site fact)**.

- **D3.** The acceleration tuned at the screen: which readings (`stop`, `change`, `wake`) and its `gain`.

## E. The 3D fluid's physics

- **E1.** The temperature fixed: its advection moves to the density's, carried by the projected velocity (Fedkiw
  2001, Bridson), where today it is carried mid-step, after the confinement and before the projection, by a
  velocity not yet free of divergence; the buoyancy reads the temperature of the step before.
- **E2.** The temperature and the expansion checked in the app, each in its own panel: what each does to the
  water and whether it works. The temperature after E1, with `buoyancy` above 0.

## F. The 3D fluid's look at 4K

What is built: [FLUID_3D.md](FLUID_3D.md#the-density).

- **F1.** The blobs' and the surface's p95: a sprite that grows to twice `radius` on a strong drop fills four
  times the pixels; the cut at 2 sigma and the lower cap ([TIMING.md](TIMING.md#the-particles-1)), judged on the
  look.

## H. One unit across the stages

- **H1.** Every length in the flow in the smaller side's widths: the motion's velocity, the bridge's readings, the
  fluid's inputs (`spread`), its look (`opacity`, `fog`), the box's depth, the monitor's scale. Today a width is
  the image's, the smaller side in portrait and the longer in landscape, so a preset means less when the box
  turns; in portrait nothing changes. One rule for every stage, made where each length is made.

## I. The 2D flow follows the 3D

Once the 3D is finished, the 2D rebuilt in `modules/flow` on the 3D's principles. Until then the 2D runs as on
`main` ([FLOW.md's constraints](FLOW.md#constraints)).

- **I1.** The 2D fluid takes the boundary by a drag and the body force's acceleration times dt, as the 3D does
  ([FLUID_3D.md](FLUID_3D.md#the-inputs)), in place of adding the bridge's velocity as a source times dt.
- **I2.** The 2D motion on the Lucas–Kanade solve, in widths/s with its velocity gate, in place of `GradientFlow`.
- **I3.** The 2D bridge rebuilt on the 3D bridge's filters, the three copied classes (`DensityBridge`,
  `TemperatureBridge`, `VelocityMagnitude`) shared again.
- **I4.** Each 2D stage its document as it is rebuilt (`BRIDGE_2D.md`, `FLUID_2D.md`); `fluid_2d.py`'s 21-line
  pipeline docstring goes with it.

## J. Every setting and its description

- **J1.** Every setting and its description gone through, last, after the refactor, against the rule: a `Field`
  description is one short line of panel text, what and its unit; and whether every quantity is in the flow's
  textbook units, which a scan cannot judge.

## Parked

- **The range (was C3).** `edge`, `max_speed` and the thresholds are in metres or px, tuned at 0.65–1.5 m;
  stereo noise grows with the square of the distance, so at 2–3.5 m they may need to be in disparity, and a
  depth's noise from frame to frame may come near `max_speed`'s 12 cm and start the track over at rest
  **(deduction)**. Parked until the camera is moved to its place; then a recording at 2–3.5 m and they measured
  there.
Not worked on until asked for; each with what is known.

- **z from x/y's spread (looming).** A surface coming toward the camera grows in the image, so its x/y motion
  spreads: its speed toward the camera is its depth times the flow's divergence over 2 (time to contact, Lee
  1976). A second measurement of z, clean where the depth's is noisy, needing texture and a patch facing the
  camera; the filter would fuse the two. Looked at on the captures, 2026-11-17 (one each, the motion raw, the
  flow blurred 3 to 6 px, the signed mean over the player's moving pixels against the depth's z): it follows a
  push (hands, correlation +0.59 at half the size) and a sideways arm less (measure, +0.40 at twice the size),
  and free movement the wrong way (show, −0.42): a sideways limb, a turn or a limb passing in front spreads the
  flow too. At rest it is 2 to 5 times quieter per pixel than the depth's z. Not fused or tried further.

- The numpy versions of the flow's shaders removed, module by module, with the tests built on them: shaders are
  checked in a GPU harness. About 10 files in `modules/` and `apps/` carry them (the motion's, the bridge's).
- The flow's stages in `modules/flow`, the render only drawing them. A stage now sits in two packages (the bridge
  is `Bridge3DLayer` in the render, the view `ViewLayer` in deep_flow), as each subclasses the render's
  `LayerBase` and modules may not import the render **(deduction)**. The textbook split: each stage a plain flow
  class as `Fluid3D` is, with its settings, ports, inputs and outputs, and a thin render layer per stage that
  calls it, beside the monitor.
- The 4K upscale: depth in 3D at the grid and detail in 2D at 4K, a march at the grid's size, one
  edge-aware upscale of the flat result and a 4K detail layer advected by the grid's velocity; what
  was found is in [FLUID_3D.md](FLUID_3D.md#the-densitys-detail).
- The transfer curve in the particles' march: their slices hold a sum of the grid's layers, so the curve needs
  its point in that sum ([FLUID_3D.md](FLUID_3D.md#the-look)); with the particles.
- The particles, and how they combine with the fine density.
- The particles' weight: the particles splatted into an `r32ui` grid by fixed-point
  `imageAtomicAdd` (it needs integers), converted to the density volume the buoyancy reads.
- The sparse grid: each step a pass marks the 8 × 8 × 8 bricks whose velocity or density is above
  a floor, grown by one brick, into a list with an atomic counter; every pass but the pressure then
  runs `glDispatchComputeIndirect` over the list, one workgroup a brick, and a brick off the list
  keeps its values. The volumes stay dense: 24 GB holds them, so GVDB's atlas, which saves memory,
  buys nothing here. The multigrid's coarse levels stay whole, an eighth and less of the work. Its
  gain is the share of bricks off the list; first one cheap pass counts the active share on a
  recording, and the grid is built only if it is under half.
- MacCormack's cost at 4K: a 4K texel reads 41 texels a step against semi-Lagrangian's 16 (the
  predictor, the velocity again, the forward trace, the cell, the limiter's eight), 3.7 ms for the
  4K × 16 density, where each field's `advection` already lets the 4K density run semi-Lagrangian. The
  way down left: the predictor reading its eight texels itself and keeping their min and max for the
  limiter, about a fifth less **(deduction)**.
- The density's detail carried by a flow map (BiMocq², Qu et al. 2019): the coordinates moved on
  the coarse grid and the 4K density read through them, baked every half second; its benefit to be
  shown first.
- Vortex particles shed by the body for swirls below the grid (He et al. 2011; Weißmann and
  Pinkall 2009).
- Each field panel showing its own input gains as well, the same values as `input`'s: shared
  between sibling panels, which `share` cannot do (it runs parent to child), so either hidden fields
  at the top shared down to both, or a share between panels. In a field panel the system's name is
  dropped as in `input`.
- The camera as a standalone OAK app: the depth's alignment, the PC's now, then needs a home.
- A camera without depth, its mask from another source: deep_flow's mask is the depth's
  ([VIEW.md](VIEW.md#the-mask)); a mask from RVM, for a camera without depth, would come back from hd_trio, on
  the crop, cut into the view's window.
- A depth for a 2D camera: the 3D fluid puts a pixel in at its z position and nothing where it has none
  ([FLUID_3D.md](FLUID_3D.md#the-bodys-depth)), so a 2D camera gives it nothing. A depth made from the image
  (a monocular depth model) on the view would give the motion its z and z position as a depth camera's does.
  A single-frame model (Depth Anything V2) is good on a frame but flickers frame to frame, which the depth
  velocity would read as z; video models keep it consistent and stream: Video Depth Anything (its smallest
  model about 30 fps), Online Video Depth Anything (42 fps on an A100), FlashDepth (24 fps at 2K) **(site fact,
  from their papers, not measured here)**. Which one, its latency against the camera's, and whether its depth
  holds the depth velocity's needs is the first question.

## Rules it touches

- A `Field` description is one short line of panel text: J1.
- A built step leaves the plan; plans and commits: one commit per checked step, on the user's literal "go".

- Compare a result only with rows of the same recording, settings and code paths: C2.

Not designed yet, so their rules are not named: D3, E1, E2, F1, H1, I1 to I4.
