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
| A   | [The measurements](#a-the-measurements)                                      | rules set; A1, A2 open           | [FLOW.md](FLOW.md#the-goals)                       |
| B   | [The view](#b-the-view)                                                      | built                            | [VIEW.md](VIEW.md)                                 |
| C   | [The motion](#c-the-motion)                                                  | built; steps open                | [MOTION.md](MOTION.md)                             |
| D   | [The bridge](#d-the-bridge)                                                  | built; the body force            | [BRIDGE_3D.md](BRIDGE_3D.md)                       |
| E   | [The 3D fluid's physics](#e-the-3d-fluids-physics)                           | the temperature                  | [FLUID_3D.md](FLUID_3D.md)                         |
| F   | [The 3D fluid's look at 4K](#f-the-3d-fluids-look-at-4k)                     | open                             | [FLUID_3D.md](FLUID_3D.md), [TIMING.md](TIMING.md) |
| H   | [One unit across the stages](#h-one-unit-across-the-stages)                  | not started                      | [FLOW.md](FLOW.md)                                 |
| I   | [The 2D flow follows the 3D](#i-the-2d-flow-follows-the-3d)                  | not started                      | [FLOW.md](FLOW.md)                                 |
| J   | [Every setting and its description](#j-every-setting-and-its-description)    | last, after the refactor         | `CLAUDE.md`, `.claude/rules`                       |

D1 needs the operator at the screen; the other lines go on meanwhile.

The apps on deep_flow's input have their own plan, [apps/APPS_PLAN.md](../../apps/APPS_PLAN.md); recording and
-sim, deep_flow's live checks among them, theirs, [RECORDING_PLAN.md](../recording/RECORDING_PLAN.md). The
chain's timing is in [TIMING.md](TIMING.md) and [INPUT_THREADS.md](INPUT_THREADS.md#the-cadence).

The 3D fluid's lines are weighed by [its goal](FLUID_3D.md#the-goal): sharp, translucent ink patterns, layers
distinct.

## A. The measurements

The goal: a measurement that can be trusted the first time, and kept only while it still says something. Two
kinds:
- **Time**: what a pass costs, in ms on the GPU or the CPU. It holds while the pass's code and size are the same,
  whatever the recording; it goes stale when the pass changes.
- **Result**: what a stage makes (z's noise and its real z, the fit, the water's speed, the look). It holds only
  for its recording, its settings and its code paths; 2026-11-16 showed all three move it (a panel setting slipped
  into a capture, a baseline compared without the gate it was made with, one run swinging ±0.6).

The rules are in `.claude/rules/flow.md`, "Flow measurements".

What is measured today (the tables with numbers in the flow's documents, 73), by the recording they name:
| Recording                         | Still on disk        | Kind            | Where                                                        |
|-----------------------------------|----------------------|-----------------|--------------------------------------------------------------|
| `20261112-125832_measure`         | yes                  | result, time    | VIEW.md's depth tables; TIMING.md's view (one path); INPUT_THREADS.md's waits and cadence |
| `20261112-124918_hands`           | yes                  | time            | TIMING.md's stages (15 November); INPUT_THREADS.md           |
| `20261112-130012_edges`           | yes                  | time            | TIMING.md's view (`SURFACE`, `MASK`, gone)                   |
| `20261112-130105_show`            | yes                  | time            | RESOLUTION.md, MONITOR.md                                    |
| stillslowfast (`20261103-184710`) | no, its capture only | result, time    | FLUID_3D.md (pressure, advection, particles, look), VIEW.md's mask table, TIMING.md (speed-ups, baseline, the look) |
| standmoves (`20261103-215324`)    | no, its capture only | result, time    | BRIDGE_3D.md, MOTION.md, FLUID_3D.md's measuring, TIMING.md  |
| blocky (`20261111-091147`)        | no, its capture only | result, time    | INPUT_THREADS.md's cadence, TIMING.md's box                  |
| `20261111-123241`                 | no                   | time            | INPUT_THREADS.md's cadence                                   |
| none named                        |                      | mostly time     | about half: BRIDGE_3D.md's filters' GPU, FLUID_3D.md's walls and expansion, TIMING.md's particles and budget |

- **A1.** Every table on a recording no longer on disk (stillslowfast, standmoves, blocky, `20261111-123241`)
  marked as not reproducible, beside the recording it names; measured again when its stage is next worked on.
- **A2.** The motion's measuring made fit for A3 and A4, in `scripts/depth_velocity_compare.py`: the row with
  x/y beside z (still noise inside, slow, fast, fast ÷ noise), the lag of a side behind the other over the moving
  stretches (ms, from the two series' best shift), several captures in one call with their mean and spread, and
  the row written with its stamp: kind, date, commit, recording, captures, and the settings of each side. A
  recording's stretches are read from a `segments.json` in the recording's own folder, written once, where they
  now sit beside each capture and go with it.
- **A3.** New baselines for today's view and motion (the depth mask gone, the fill, edge 0.1 m, erode and
  feather):
  - the result in [VIEW.md's depth table](VIEW.md#the-depth), raw z and x/y on the measure recording, three
    captures; its rows still name the depth mask's gate and `view.depth.tolerance`;
  - the view's time in [TIMING.md](TIMING.md#the-view), at the preset's `erode` and `feather` and at 2 and 4,
    as at 0 their passes are skipped;
  - a new column in [TIMING.md's stages](TIMING.md#the-stages), every timer of one run on the hands recording.
- **A4.** What each part of the motion does for [its goal](MOTION.md#the-goal), before C: least noise while
  keeping the most real motion, at the lowest latency. The parts work in groups, measured in the chain's order,
  each with the groups after it neutral:
  1. the x/y solve: `levels`, `window`, `damping`; z's track is carried along it;
  2. what counts as one surface: `tolerance`, the step filter (`edge`), `same_surface`, z's `window`;
  3. z over time: `smooth`, `catch_up`, and `tolerance` again, as it also starts a track over;
  4. the cut at rest: `dead_zone`, z's `threshold`, x/y's `threshold`; measured last, everything on.

  `use_mask` and `xy_smooth` are left out: the mask says where, not how noisy, and `xy_smooth` is a convenience.
  A part's neutral value is its off where it has one (`damping` 0, `same_surface` off, `smooth` 0, `catch_up` 0,
  `dead_zone` 0, the thresholds 0, `levels` 1, z's `window` 1×1) and else the end of its range that does least:
  `edge` 0.3 m, `tolerance` 0.5 m, x/y's `window` 3×3. The steps:
  - the four recordings gone through first: the still and the moving stretches of each in its `segments.json`,
    and what each holds (the distance, what the player does, its stretches, the columns it gives) in a table in
    MOTION.md, above the results. The measure recording
    has moves slow then fast, so it gives every column; the others give the still noise, how z follows the real
    depth, and the lag, and slow against fast only where they hold such a move **(deduction)**;
  - pass 1, the preset against neutral, on the four recordings: each part alone, and the pairs that depend on
    each other together (step filter × `tolerance`, `smooth` × `catch_up`, `smooth` × `tolerance`,
    `dead_zone` × `threshold`);
  - pass 2, values, for the parts that matter, on the measure and the show recording and any recording where
    pass 1 came out differently: half and double the preset's value inside its range (`levels` and the windows a
    step down and up), the pairs in a small grid.

  All rows of a recording replay its same three captures; the captures stay until Matthias is asked. The table
  goes in [MOTION.md's settings](MOTION.md#the-settings), in place of the reasons from recordings that are gone.
  A part that does nothing for the goal, or a value that holds on one recording only, is a question for C. The
  recordings are at 0.65–1.5 m; 2–3.5 m stays C3's.

Further measurements on the motion are free where a result points to something (Matthias, 2026-11-16), each
stated before it is made.

- **A5.** When A3 and A4 are done: the rows of [VIEW.md's depth table](VIEW.md#the-depth) that name the depth
  mask's gate and `view.depth.tolerance` deleted; until then they stay beside the new baseline, to compare.

## B. The view

Its goal and what is built: [VIEW.md](VIEW.md). New points go here.

## C. The motion

From A4's measurements (four recordings, every frame) and the short runs after it (one capture of hands and of
measure): z needs smoothing to be quiet, and what cuts a large, fast z most is a long smooth (kept 84% at 25 ms,
43% at 100), which the catch-up gives back (82 to 85%) at almost no noise at rest; the dead zone costs z and lag
with smoothing and is the quiet one only without; `tolerance` and `same_surface` show no effect; the x/y solve,
the step filter and z's window earn their place. The aim: quiet at rest, the most z kept, the least lag.

Tried and undone (2026-11-17): the dead zone before the filter, a backlash on the measured depth. Quieter at
rest and later: the filter sees nothing until the zone opens, so its lag comes after the zone's where the two
overlap now (hands, unsmoothed, `dead_zone` 0.02: 52 ms against 7; `smooth` 25: 90 against 65; measure's still
noise 2.4 / 3.1 against 9.1 / 5.1).

- **C1.** The catch-up's rule replaced by the 1€ filter's (Casiez, Roussel, Vogel 2012): the smoothing time
  shortens as the speed rises, two numbers, the smoothing at rest and how fast it opens; the speed it opens on
  to hold x/y's, which carries little noise. The catch-up's Emas and their texture go. Checked on one capture of
  hands and of measure against the catch-up's rows (`smooth` 25 and 50 with `catch_up` 8): still noise, z
  kept, lag; about 5 minutes a run.
- **C2.** The values, after C1, each measured on the four recordings and judged in the app: the smoothing's two
  numbers and z's `threshold`, with `dead_zone` 0; x/y's `threshold` around 0.4; a step up on `levels`, x/y's
  `window` and `damping` against their ms. The step filter and z's window stay unless a row shows a clear gain
  that keeps the z. The settings' defaults and MOTION.md's reasons follow; the preset when Matthias saves it.
  No new recording.
- **C3.** The range: `edge` and the thresholds are in metres or px, tuned at 0.65–1.5 m; stereo noise grows
  with the square of the distance, so at 2–3.5 m they may need to be in disparity **(deduction)**. A recording
  at 2–3.5 m, made by the operator, and they measured there.

Questions, for C:
- The dead zone: with z smoothed, the smooth and the threshold do its job (still noise 0 on measure and edges
  without it) and it costs a quarter to half of z and 20 to 50 ms. Removed (the setting, the anchor, its part of
  the catch-up's rule) once C2's run on the four recordings holds without it?
- `tolerance` and `same_surface` showed no effect on the four recordings, none of which has the case they are
  for, an arm moving sideways 10 to 20 cm before the chest. Measured on such a recording first, or constants now?
- The numpy reference of the z track (`track_depth`) and its tests: kept, or gone with the compare's `--verify`
  of z?

## D. The bridge

What is built: [BRIDGE_3D.md](BRIDGE_3D.md).

- **D1.** The body force's acceleration made useful: looked into on a recording, what it does to the water and
  what it should do in the look, then tuned.

## E. The 3D fluid's physics

- **E1.** The temperature fixed: its advection moves to the density's, carried by the projected velocity (Fedkiw
  2001, Bridson), where today it is carried mid-step, after the confinement and before the projection, by a
  velocity not yet free of divergence; the buoyancy reads the temperature of the step before.

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

Not worked on until asked for; each with what is known.

- **z from x/y's spread (looming).** A surface coming toward the camera grows in the image, so its x/y motion
  spreads: its speed toward the camera is its depth times the flow's divergence over 2 (time to contact, Lee
  1976). A second measurement of z, clean where the depth's is noisy, needing texture and a patch facing the
  camera; the filter would fuse the two. To look at once C1 and C2 are built, first on the captures: how the
  spread follows the depth's z on the hands and the show recording **(deduction, not measured)**.

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
- A measured table names its kind, date, commit, recording and settings: A1 to A4.
- Ask to measure a time again when its pass's code or size changes: A3.
- A built step leaves the plan; plans and commits: one commit per checked step, on the user's literal "go".

- Shaders get checked in a GPU harness: C1.
- Compare a result only with rows of the same recording, settings and code paths: C1, C2.

Not designed yet, so their rules are not named: C3, D1, E1, F1, H1, I1 to I4.
