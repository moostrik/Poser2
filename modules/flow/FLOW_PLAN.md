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
  recordings are at 0.65–1.5 m; 2–3.5 m stays C2's.

Further measurements on the motion are free where a result points to something (Matthias, 2026-11-16), each
stated before it is made.

- **A5.** When A3 and A4 are done: the rows of [VIEW.md's depth table](VIEW.md#the-depth) that name the depth
  mask's gate and `view.depth.tolerance` deleted; until then they stay beside the new baseline, to compare.

## B. The view

Its goal and what is built: [VIEW.md](VIEW.md). New points go here.

## C. The motion

- **C1.** The depth velocity's `track` renamed by what it holds (the followed disparity, its rate and its anchor):
  `DepthVelocity._track` and the monitor's port `z.track`; a word that says nothing on its own.

- **C2.** The range: `edge`, `tolerance` and `dead_zone` are in metres, tuned at 0.65–1.5 m; stereo noise grows
  with the square of the distance, so at 2–3.5 m they may need to be in disparity **(deduction)**. A recording at
  2–3.5 m, the three measured there on the raw output.
- **C3.** `z.tolerance` split in two: one number, 0.2 m, now does three things in `depth_velocity_shader.frag`:
  which depth samples a window averages as one surface and which carried texels count as the same surface
  (space), and how far a new depth may lie from the track before it starts over (time). A number for space and
  one for time, each measured on the raw output; 0.2 m is large on a body, set by the job in time.
- **C4.** The smoothing (`z.smooth`, `z.catch_up`) judged on its own, the raw output against the smoothed
  ([FLOW.md](FLOW.md#the-goals)), once A2 is built.

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

- Name a concept after what it is and rename it everywhere in the same change: C1.
- A `Field` description is one short line of panel text: J1.
- A measured table names its kind, date, commit, recording and settings: A1 to A4.
- Ask to measure a time again when its pass's code or size changes: A3.
- A built step leaves the plan; plans and commits: one commit per checked step, on the user's literal "go".

Not designed yet, so their rules are not named: C2 to C4, D1, E1, F1, H1, I1 to I4.
