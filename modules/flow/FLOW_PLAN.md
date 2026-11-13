# The flow's plan

What is next for the flow, the one place that says it: the main lines in order, each line's items, the
details that belong to no line, and what is parked. How each stage works is in its own document
([FLOW.md](FLOW.md) lists them); an item links to the section it changes. **(deduction)** marks what is
inferred, **(site fact)** what the operator saw.

## Main lines

| #   | Line                                                                      | State                                                         | Documents                                                    |
|-----|---------------------------------------------------------------------------|---------------------------------------------------------------|--------------------------------------------------------------|
| 1   | [The flow held up to the rules](#1-the-flow-held-up-to-the-rules)         | built; the settings left, after the refactor                  | `CLAUDE.md`, `.claude/rules`                                 |
| 2   | [The 3D fluid's input](#2-the-3d-fluids-input)                            | built; the depth read to choose                               | [VIEW.md](VIEW.md), [BRIDGE_3D.md](BRIDGE_3D.md)             |
| 3   | [The 3D fluid's look at 4K](#3-the-3d-fluids-look-at-4k)                  | next                                                          | [FLUID_3D.md](FLUID_3D.md), [TIMING.md](TIMING.md)           |
| 4   | [One unit across the stages](#4-one-unit-across-the-stages)               | not started                                                   | [FLOW.md](FLOW.md)                                           |
| 5   | [The apps on deep_flow's input](#5-the-apps-on-deep_flows-input)          | its own plan                                                  | [APPS_PLAN.md](../../apps/APPS_PLAN.md)                      |
| 6   | [The 2D flow follows the 3D](#6-the-2d-flow-follows-the-3d)               | not started                                                   | [FLOW.md](FLOW.md)                                           |
| 7   | [A camera without depth](#7-a-camera-without-depth)                       | not started                                                   | [FLOW.md](FLOW.md), [VIEW.md](VIEW.md)                       |

Recording and -sim, deep_flow's live checks among them, have their own plan,
[RECORDING_PLAN.md](../recording/RECORDING_PLAN.md).

The 3D fluid's lines are weighed by [its goal](FLUID_3D.md#the-goal): sharp, translucent ink patterns, layers
distinct.

## 1. The flow held up to the rules

The flow's packages (`modules/flow`: `motion`, `view`, `bridge_2d`, `bridge_3d`, `fluid_2d`, `fluid_3d`,
`visualization`, `shaders` and its files) and its layers (`modules/render/layers/flow`), 210 Python files, held
against `CLAUDE.md` and `.claude/rules` by a scan of every file and by reading what a scan cannot judge. Form
only: a name, a file, a docstring, an import; nothing changes what the flow does for a rule's sake.

Met throughout: other packages imported from their `__init__`, one `from` line a package, no `__all__`, no
`import *`; time from `perf_counter`; loggers by `getLogger(__name__)`, no `print`; no `# type: ignore`; every
public function with a return hint; OpenGL imported by name, no `GL_TIMEOUT_IGNORED`.

### Steps

| #    | Step                                                                       | Rule                                                                     |
|------|----------------------------------------------------------------------------|--------------------------------------------------------------------------|
| 1.10 | every setting and its description gone through, last, after the refactor   | a `Field` description is one short line of panel text: what and its unit |

`fluid_2d.py`'s module docstring, a 21-line pipeline, goes with the 2D fluid's rebuild and its document
([line 6](#6-the-2d-flow-follows-the-3d)).

Not judged by a scan: whether every unit means something on its own, beyond the names above. The flow runs on the
render thread, so the concurrency rules touch none of it.

## 2. The 3D fluid's input

The input as good as it gets, before the 4K density's heavy side. Built: every capture reaches the motion
([INPUT_THREADS.md](INPUT_THREADS.md)), the mask's edge on the image's and the depth read across a surface
([VIEW.md](VIEW.md)), one density and its fine density ([FLUID_3D.md](FLUID_3D.md#the-density)), each output at
its readings' depth ([BRIDGE_3D.md](BRIDGE_3D.md#the-z-positions)).

- The depth read: `view.depth_read` holds `SURFACE` and `MASK` ([VIEW.md](VIEW.md#the-depth-read)). One stays,
  chosen together by eye on the edges recording; the other goes.
- The whole chain before the fluid, camera to bridge, timed as one at the preset: what each pass costs against
  what it gives the fluid.
- The dye's amount weighed by the image's brightness, and z's share in the dye's amount: discussed, not
  designed.
- No captures: while no capture comes (the camera stalls, -sim is paused) the bridge keeps its last output and
  the fluid keeps taking it, where the motion holds its last velocity. What the flow should do then is not
  decided.

## 3. The 3D fluid's look at 4K

### The fade and the evaporation

Each field's `fade_time` (seconds to 1%, a share at a time) and `evaporation_time` (seconds a full value takes
to reach 0, a fixed amount at a time) do not yet give what is wanted **(site fact)**
([FLUID_3D.md](FLUID_3D.md#the-fade-and-the-evaporation)); what each should do for the look is the first
question.

### The temperature

Whether the temperature needs a field of its own. Buoyancy needs only a scalar the water is lighter or
heavier by: the density alone can be it (dyed water sinks), and the temperature adds currents apart from the dye
(warm clear water rising), the textbook smoke's soot and heat. Off in the preset (`buoyancy` 0), and nothing
is injected into it then.

Where it is kept, its advection moves to the density's: today it is carried mid-step, after the
confinement and before the projection, by a velocity not yet free of divergence; the textbook carries every
scalar by the same projected velocity (Fedkiw 2001, Bridson), and the buoyancy reads the temperature of the
step before.

Making the density sink needs no temperature: `temperature.sink` (widths/s² per unit of density, along
gravity) weighs dyed water wherever it is, with `buoyancy` 0, in the same pass, its direction `gravity_x`,
`_y`, `_z`. It sits in the temperature's panel though it is the density's weight: if the temperature goes,
`sink` and the gravity move to a panel of their own or the density's.

### Details

- The body force's acceleration: tested, tweaked and made useful.
- The velocity loses its energy faster than expected **(site fact)**, with the plain step on the velocity: the
  plain step's own loss (a swirl keeps 52% of its speed in 2 s, MacCormack 62%), the drag now over the body's
  whole depth, or another cause; MacCormack with the clamp on the velocity is the first thing to try.
- The particles: worked on, and how they combine with the fine density.
- The masking in the bridge: the mask gates the velocity in the motion (`use_mask`) and the density's color in
  the bridge; in the bridge alone it is one gate in one stage.
- The blobs' and the surface's p95: a sprite that grows to twice `radius` on a strong drop fills four times
  the pixels. The cut at 2 sigma and the lower cap ([TIMING.md](TIMING.md#the-particles-1)) are the fix, judged
  on the look.
- The run-to-run spread of the pose's load, 1 to 1.5 ms of the step between two runs of one configuration:
  the person detection's rate is not fixed.

## 4. One unit across the stages

Every length in the flow is in widths: the motion's velocity (widths/s of the camera's image), the bridge's
readings, the fluid's inputs (`spread`), its look (`opacity`, `fog`), the box's depth, the monitor's scale.
In portrait the width is the smaller side; in landscape it is the longer, and the same settings would
then mean less against what is seen. One unit across the stages, the smaller side, keeps a preset's look
and feel when the box turns: in portrait nothing changes. It reaches past the fluid (the motion, the
bridge, the monitor), so it is one rule for all of them, made where each length is made.

## 5. The apps on deep_flow's input

Not the flow's: its own plan, [apps/APPS_PLAN.md](../../apps/APPS_PLAN.md).

## 6. The 2D flow follows the 3D

The 2D system (`GradientFlow`, `bridge_2d`, `Fluid2D`, hd_trio) runs as on `main`
([FLOW.md's constraints](FLOW.md#constraints)). It follows the 3D's principles once the 3D is finished:

- The 2D fluid: it adds the injected velocity times dt (`Fluid2D.add_velocity`), as Stam's Stable Fluids
  adds a source; but the bridge's velocity is a velocity, in view widths per second like the
  fluid's, so it is taken as an acceleration, `velocity.gain` acts as a rate, and the fluid stays
  near 0.05 to 0.1 of the motion's speed **(deduction)**, a monitor needing a scale of 20 to 30 to
  show it. The 3D fluid takes the boundary by a drag and the body force's acceleration times dt
  ([FLUID_3D.md](FLUID_3D.md#the-inputs)); the 2D fluid is to follow it: the drag, the two systems,
  the one grid.
- The 2D motion on the Lucas–Kanade solve, in widths/s with its velocity gate, in place of `GradientFlow`.
- The 2D bridge rebuilt on the 3D bridge's filters, the three copied classes (`DensityBridge`,
  `TemperatureBridge`, `VelocityMagnitude`) shared again.
- hd_trio's density fade from motion: its wiring pass writes `density.fade_time` every tick from the pose's
  motion, a settings field written per tick. A per-tick value on the fluid layer would take it off the
  settings.

Each 2D stage gets its document as it is rebuilt (`BRIDGE_2D.md`, `FLUID_2D.md`).

## 7. A camera without depth

- The depth's mask with another mask source: deep_flow's mask is the depth's ([VIEW.md](VIEW.md#the-mask-and-the-depth-mask));
  a mask from RVM, for a camera without depth, would come back from hd_trio. With it the motion's `depth_mask`,
  the depth's own cut that gates z, is still the view stage's, and RVM's mask, on the crop, would have to be
  cut into the view's window.
- A depth for a 2D camera: the 3D fluid puts a pixel in at its z position and nothing where it has none
  ([FLUID_3D.md](FLUID_3D.md#the-bodys-depth)), so a 2D camera gives it nothing. A depth made from the image
  (a monocular depth model) on the view would give the motion its z and z position as a depth camera's does.

## The motion's details

- The false z at a limb's edge: a limb moving sideways before the body reads false z at its edges: the leading
  edge carries 43% of a sweep's z, the trailing edge 12%. The cause is the depth's ramp at the limb's edge. The
  camera's temporal filter is off and its confidence does not sharpen the ramp. Accepted for now; the cue not
  yet tried is a forward-splat count of the x/y motion (0 = uncovered, above 1 = covered).
- The range: `edge`, `tolerance` and `dead_zone` are in metres and tuned at 0.65–1.5 m. Stereo noise grows
  with the square of the distance, so at 2–3.5 m they may need to be in disparity **(deduction)**. This needs
  a recording at that distance.
- The depth velocity's `track`: its memory is named `track` in the code (`DepthVelocity._track`, the monitor's
  `z.track`), a word that says nothing on its own; it holds the followed disparity, its rate and its anchor.
  To be renamed by what it holds, with the monitor's port.
- A soft limit on the z speed: a guard to try on the motion beside its finite guard
  ([MOTION.md](MOTION.md#the-guard)): z as measured up to a knee (about 1.5 m/s), compressed above it toward a
  maximum (about 3 m/s), the speed gate (6 m/s) behind it; a curve per reading, no time filter, checked first
  against the spread of z speeds on a recording.
- A finer depth would give the mask and z a finer edge at the source; the camera's sizes are a given
  ([RESOLUTION.md](RESOLUTION.md#the-rules)).

## Parked

Not worked on until asked for; each with what is known.

- The 4K upscale: depth in 3D at the grid and detail in 2D at 4K, a march at the grid's size, one
  edge-aware upscale of the flat result and a 4K detail layer advected by the grid's velocity; what
  was found is in [FLUID_3D.md](FLUID_3D.md#the-densitys-detail).
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

## Rules it touches

- A `Field` description is one short line of panel text: 1.10.
- Remove obsolete pathways; keep presets and settings schemas in sync: the depth read not chosen and the
  preset's `view.depth_read`, line 2.
- Choose a data path by its latency: the chain before the fluid timed as one, line 2.
- A built step leaves the plan; plans and commits: one commit per checked step, on the user's literal "go".

Not designed yet, so their rules are not named: the dye's brightness and no captures (line 2); the fade and the
evaporation, the temperature and the details (line 3); lines 4, 6 and 7; the motion's details.
