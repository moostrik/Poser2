# The flow's plan

What is next for the flow, the one place that says it: the main lines (A to J), along the flow's chain (the view,
the motion, the bridge, the fluid's physics, its look), then those across the stages; each line's steps in the
order they are worked (B1, B2, ...), what is not decided as a question; what is parked. A built step leaves the
plan; the others keep their numbers. How each stage works is in its own document ([FLOW.md](FLOW.md) lists
them); a step links to the section it changes. **(deduction)** marks what is inferred, **(site fact)** what the
operator saw.

## Main lines

| #   | Line                                                                         | State                             | Documents                                          |
|-----|------------------------------------------------------------------------------|-----------------------------------|----------------------------------------------------|
| A   | [The measurements](#a-the-measurements)                                      | built                             | [FLOW.md](FLOW.md#the-goals)                       |
| B   | [The view](#b-the-view)                                                      | built                             | [VIEW.md](VIEW.md)                                 |
| C   | [The motion](#c-the-motion)                                                  | built; C2 the values, C3 range    | [MOTION.md](MOTION.md)                             |
| D   | [The bridge](#d-the-bridge)                                                  | built; D6 the checks              | [BRIDGE_3D.md](BRIDGE_3D.md)                       |
| E   | [The 3D fluid's physics](#e-the-3d-fluids-physics)                           | built; E6 the app's check         | [FLUID_3D.md](FLUID_3D.md)                         |
| F   | [The 3D fluid's look at 4K](#f-the-3d-fluids-look-at-4k)                     | F1, F2 white                      | [FLUID_3D.md](FLUID_3D.md), [TIMING.md](TIMING.md) |
| H   | [One unit across the stages](#h-one-unit-across-the-stages)                  | not started                       | [FLOW.md](FLOW.md)                                 |
| I   | [The 2D flow follows the 3D](#i-the-2d-flow-follows-the-3d)                  | in superficial_flow's plan        | [FLOW.md](FLOW.md)                                 |
| J   | [Every setting and its description](#j-every-setting-and-its-description)    | last, after the refactor          | `CLAUDE.md`, `.claude/rules`                       |

E6 needs the operator at the screen; the other lines go on meanwhile.

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
`max_speed` between frames, `rest_speed` and `halving_speed` for z's smoothing, `tolerance` and `same_surface` gone);
(2026-11-18) the measuring's sums on the GPU, the 1€ rule opened by the track's own speed too, the stop's ms;
described in [MOTION.md](MOTION.md#z-the-depth-velocity) with their tables.

Tried and undone (2026-11-17): the dead zone before the filter, a backlash on the measured depth. Quieter at
rest and later: the filter sees nothing until the zone opens, so its lag comes after the zone's where the two
overlap now (hands, unsmoothed, `dead_zone` 0.02: 52 ms against 7; `smooth` 25: 90 against 65; measure's still
noise 2.4 / 3.1 against 9.1 / 5.1).

- **C2.** The values, measured 2026-11-17 on the four recordings, the tables in MOTION.md
  ([the motion's values](MOTION.md#depth_velocity-z)); the surfaces in the mask that are not the player left
  out by `exclude` in each recording's `segments.json`. Proposed: `levels` 5, z's `threshold` 0.5 or 1 by
  eye; `max_speed` stays 6 (its job is another surface at the player's edge, which `exclude` leaves out of
  these runs, and it was tested before); `smooth_like_z` off; `halving_speed` stays 0.5: since the 1€ rule opens on
  the track's own speed (2026-11-18) it lags 4 ms on hands at a still noise of 0.64, 0.25 lags 2 ms at 3.7.
  `levels` 5 measured again after C5 (MOTION.md): x/y follows 20% more, z 1 ms less lag, a later stop on hands; the defaults then set with Matthias at the PC, MOTION.md's reasons following, the preset
  when he saves it.
- **C3.** The range. `edge`, `max_speed` and the thresholds are in metres or px, tuned at 0.65–1.5 m; stereo
  noise grows with the square of the distance, so at 2–3.5 m they may need to be in disparity, and a depth's
  noise from frame to frame may come near `max_speed`'s 12 cm and start the track over at rest **(deduction)**.
  A recording at 2–3.5 m, with the camera in its place, and they measured there.

## D. The bridge

What is built: [BRIDGE_3D.md](BRIDGE_3D.md), and what the fluid does with it,
[FLUID_3D.md](FLUID_3D.md#the-inputs).

The expansion is a source of volume where the body is, a fountain: water is made there and flows out. Seen in the
app, 2026-11-17, with the `mask` share: it works; with the trail alone it hardly shows, as the velocity's drag holds
the same water **(site fact)**.

Built (2026-11-17): the density's color with its coverage, and its amount apart (Porter and Duff 1984: alpha is
coverage); the luma matte in the view ([VIEW.md](VIEW.md#the-stage)), the bridge's two outputs
([BRIDGE_3D.md](BRIDGE_3D.md#the-mix)); the density's blend, over or added, and its color space, RGB or Oklab
([FLUID_3D.md](FLUID_3D.md#the-densitys-blend)), committed before the app's check on Matthias's word. Built
(2026-11-18): the view's white balance; the density's color a stage of its own between the view and the motion
([COLOR.md](COLOR.md)); the bridge's `readings` group named `filters`.

- **D6.** The checks. In the app: no dark rim at any gain; the luma matte from 0 to 1; `over` leaves no patches
  where the input meets older density; `fill` against `over`; `add` builds up without going white; `oklab`
  mixes without turning muddy; the monitor's `IMAGE` shows what the composite shows; `color.white_balance`
  pulls a warm room light to white. The injection's and the march's times measured again, with -sim while
  Matthias is away; the preset saved after.

## E. The 3D fluid's physics

The temperature as the textbook has it (Fedkiw, Stam, Jensen 2001; Bridson 2015; GPU Gems 3 ch. 30): the
warmth and the dye passive scalars carried by the same projected velocity; the buoyancy (sink × amount −
buoyancy × warmth) along gravity, at the faces, added before the projection; the heat leaves by mixing and by
cooling toward the room's. Built (2026-11-18): the order, the density's weight by its amount, the warmth from
the room's, the warmth from the motion alone; checked in the GPU harness ([FLUID_3D.md](FLUID_3D.md#the-buoyancy)).
Built (2026-11-18): gravity under `input`, the sink the density's, the temperature carried as the density is; an
`enabled` in every fluid step that can be off, a step running while enabled and its strength above 0.
Built (2026-11-19): the buoyancy in one pass, `AddToFaces3D`'s buoyancy mode, the warmth read only while the lift
is on, skipped at gravity 0 too, the force at the cells only for its monitor; checked in the GPU harness, the faces
the two passes' within the force's half precision, the pass 0.114 ms against 0.152 ms (`fluid_pass_check.py`).

- **E6.** Checked in the app, in its own panel: what the temperature does to the water and whether it works,
  with `buoyancy` above 0.

## F. The 3D fluid's look at 4K

What is built: [FLUID_3D.md](FLUID_3D.md#the-density).

- **F1.** The blobs' and the surface's p95: a sprite that grows to twice `radius` on a strong drop fills four
  times the pixels; the cut at 2 sigma and the lower cap ([TIMING.md](TIMING.md#the-particles-1)), judged on the
  look.
- **F2.** White by choice, in the shown image only, the stored density untouched. First, FLUID_3D.md states how the
  march mixes the density's colors now: as light, adding, or as ink, absorbing. Then one setting in the march,
  `highlight_white`: how far the brightest, densest parts go toward white, 0 = the pure color. After the march has
  summed a pixel's light, a shoulder as film has: as the pixel's brightness nears full, its color loses saturation
  toward white and keeps its hue (in Oklab, a and b shrink as L rises). Thin density keeps its color; only where it
  piles up does it whiten. If the march is ink, white comes another way, the background through thin density, and
  the step is planned again. Checked in the app: at 0 the colors as stored; turned up, the dense cores whiten and
  the thin edges stay colored.

## H. One unit across the stages

- **H1.** Every length in the flow in the smaller side's widths: the motion's velocity, the bridge's readings, the
  fluid's inputs (`spread`), its look (`opacity`, `fog`), the box's depth, the monitor's scale. Today a width is
  the image's, the smaller side in portrait and the longer in landscape, so a preset means less when the box
  turns; in portrait nothing changes. One rule for every stage, made where each length is made.

## I. The 2D flow follows the 3D

Its plan is superficial_flow's ([SUPERFICIAL_FLOW_PLAN.md](../../apps/superficial_flow/SUPERFICIAL_FLOW_PLAN.md)):
the app on the 2D flow, the 2D motion, bridge and fluid as their own kernels beside the 3D's.

## J. Every setting and its description

- **J1.** Every setting and its description gone through, last, after the refactor, against the rule: a `Field`
  description is one short line of panel text, what and its unit; and whether every quantity is in the flow's
  textbook units, which a scan cannot judge.

## Parked

Not worked on until asked for; each with what is known.

The 3D fluid's physics:

- The heat's diffusion: ∂T/∂t = k∇²T, solved implicitly as the viscosity is (Bridson 2015); left out, as Fedkiw
  2001 has none and the advection already blurs; added if it is missed.

The 3D fluid's look:

- The 4K upscale: depth in 3D at the grid and detail in 2D at 4K, a march at the grid's size, one
  edge-aware upscale of the flat result and a 4K detail layer advected by the grid's velocity; what
  was found is in [FLUID_3D.md](FLUID_3D.md#the-densitys-detail).
- The particles: how they combine with the fine density; the transfer curve in their march, whose slices hold a
  sum of the grid's layers, so the curve needs its point in that sum ([FLUID_3D.md](FLUID_3D.md#the-look)); their
  weight, splatted into an `r32ui` grid by fixed-point `imageAtomicAdd` (it needs integers), converted to the
  density volume the buoyancy reads.
- The density's detail carried by a flow map (BiMocq², Qu et al. 2019): the coordinates moved on
  the coarse grid and the 4K density read through them, baked every half second; its benefit to be
  shown first.
- Vortex particles shed by the body for swirls below the grid (He et al. 2011; Weißmann and
  Pinkall 2009).

Its speed at 4K:

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

The body in the fluid:

- Obstacles with a velocity of their own: the fluid's obstacles (built, static, no caller today) with a solid
  face holding the obstacle's velocity instead of 0 (Bridson 2015, ch. 5), so a moving body pushes the water
  aside ahead of it and draws it in behind.

The cameras:

- The view's crop, mask and depth move to `modules/flow` when a second app takes the 3D flow; the color, deep_flow's
  look, stays in the app.
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
- Ask to measure a time again when its pass's code or size changes: D6.

Not designed yet, so their rules are not named: C3, F1, F2, H1.
