# The 3D fluid

The 3D fluid's document, as [FLOW.md](FLOW.md) is the flow's: what the fluid is, how it takes what
the bridge gives it, how it is measured, and what is open. "Code" is what the files show;
**(deduction)** marks what is inferred from them, **(site fact)** what the operator saw.

The 2D fluid (`fluid_2d/`, `Fluid2DLayer`, hd_trio) is apart: nothing here changes it. The 3D
fluid is `Fluid3D` (`fluid_3d/fluid_3d.py`), its layer `Fluid3DLayer`
(`modules/render/layers/flow/fluid_3d_layer.py`), run by deep_flow alone.

## The fluid

A step (`Fluid3D.update`), in order, at the preset (`apps/deep_flow/data/settings/settings.json`):

| Pass        | What                                                                                  | On                    | Depends on the input |
|-------------|---------------------------------------------------------------------------------------|-----------------------|----------------------|
| advect_vel  | self-advection by `advection`, `self_advection` 1, fade 60 s; on its result the dampen, a drag on the speed above `dampen_threshold` (50 widths/s) over `dampen_time` (2 s) | velocity 256×448×128 | the trace distance |
| viscosity   | Jacobi diffusion, off (`viscosity` 0)                                                 | velocity              | no                   |
| vorticity   | the curl, then the confinement force × dt added to the velocity in the same pass       | velocity, curl        | no                   |
| advect_temp | off: `buoyancy` 0 clears the temperature once, and nothing is injected into it  | temperature           | the trace distance   |
| buoyancy    | gravity × (sink × density − buoyancy × ΔT) × dt; gravity is 0 in the preset           | force                 | no                   |
| incompress  | divergence, 40 Jacobi iterations, gradient subtraction, on the walls ([The walls](#the-walls)) | pressure, velocity | no              |
| advect_den  | semi-Lagrangian advection, one trilinear read per texel, the evaporation subtracted   | density 256×448×128   | the trace distance   |
| composite   | front-to-back march, one step per layer                                               | output 2160×3840      | no                   |

Every field is a `GL_TEXTURE_3D` on one grid of cubic cells at the stage's size, 256 × 448 by
`volume.layers` 128: the velocity RGBA16F, the temperature R16F and the pressure R32F, the density
RGBA16F with its color baked in, each a swap pair, the obstacle R8. The simulation's cell is 1/256
width in x, y and z, so the slab is 128 / 256 = 0.5 widths deep (`_z_extent`). The grid is the one
that holds 60 fps in deep_flow with the pose on the same GPU ([TIMING.md](TIMING.md#the-grid)). The density is on the same
grid, or `density.size` times finer in x and y (1, 2 or 4): every pass that reads the velocity or
the obstacle against it samples by uv, so the shaders are the same (`advect_3d`, `inject_3d`,
`buoyancy_3d`). The composite reads the volume at the output's 2160 × 3840 and upsamples it
bilinearly. The monitors read every field as it is. The GPU is a 24 GB RTX 4090, so nothing pages.

| `layers` | `density.size` | density, one buffer | step, GPU ms |
|----------|----------------|---------------------|--------------|
| 64       | 1              | 448 × 800 × 64, 0.18 GB | 6.6      |
| 64       | 2              | 896 × 1600 × 64, 0.73 GB | 7.2     |
| 24       | 4              | 1792 × 3200 × 24, 1.1 GB | 4.8     |

The step's GPU time at 448 × 800, measured in a hidden window on a swirl at the preset's passes, the
composite at 2160 × 3840; every pass's time in the app is in [TIMING.md](TIMING.md). `layers` is the knob if it is too much; `density.size` 4 is the 4K look on the same
path, with the z lerp across coarse layers and the scattered reads of a big volume.

What the layer injects before the step (`Fluid3DLayer._inject`), each at the injection's `weight`,
into the gaussian slab at `injection_layer` 0.5, `injection_spread` 0.075 widths:

| Input                          | The fluid                                                            | Shader                       |
|--------------------------------|----------------------------------------------------------------------|------------------------------|
| `boundary_velocity`            | dragged toward, [The drag](#the-drag), w = −z                        | `boundary_velocity_3d`       |
| `body_force_acceleration`      | added × `body_force.gain` × dt, w = −z                               | `body_force_acceleration_3d` |
| `boundary_density` (RGBA)      | added × `density.boundary_gain` × dt                                 | `inject_3d`                  |
| `body_force_density` (RGBA)    | added × `density.body_force_gain` × dt                               | `inject_3d`                  |
| `boundary_temperature`         | added × `temperature.boundary_gain` × dt                             | `inject_3d`                  |
| `body_force_temperature`       | added × `temperature.body_force_gain` × dt                           | `inject_3d`                  |

The velocity is in image widths per second, as the bridge's; the advection moves a texel by
velocity × dt × `speed` (1) in uv, so 1 width/s crosses the width in 1 s.

The volume's settings: `layers`; `injection_layer` and `injection_spread`, where the body sits in
the volume; `emission`, `absorption`, `glow`, `tint` and `fog`, the composite's march; `wall`, the
z walls. Obstacles stay supported (`set_obstacle`, `add_obstacle`, `clear_obstacles`, no caller
today): one R8 volume on the simulation's grid, which every field's pass samples.

## The inputs

The rule, by what each input is. The trail is a velocity in the fluid's own unit; added × dt it
would act as an acceleration and its gain as a rate. In Navier–Stokes the body force is an
acceleration per unit mass; a body in the fluid is a boundary that moves.

| Input is              | The fluid                                                                       | Here                                             |
|-----------------------|---------------------------------------------------------------------------------|--------------------------------------------------|
| a body's velocity     | is dragged toward it where the body is, over a time, never past it              | the bridge's `boundary.velocity`                 |
| a velocity change, Δv | adds it once in total: as an acceleration, the pulse over its area, × dt        | the bandpass's readings as accelerations         |
| an acceleration       | adds it × dt                                                                    | buoyancy, the vorticity force, `body_force.acceleration` |
| a source              | adds it × dt, an amount per second                                              | the densities and the temperatures               |

### The chain

What the bridge gives and what the fluid does with each part, the one statement of it. The six
readings, each as a velocity and as an acceleration:

| Reading                        | Made by      | As a velocity, widths/s                                                                  | As an acceleration, widths/s²                                                                 |
|--------------------------------|--------------|------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------|
| `trail`                        | the lowpass  | the body's velocity, held and smoothed                                                   | (velocity − trail) / `attack` or `smooth`: the gap the lowpass closes, a spike at a start, a brake over the coast at a stop, advected and blurred as the trail is |
| `change`                       | the bandpass | the fast lowpass minus the slow, over `step_peak`: the signed change of speed, the Δv whole | the same over (`slow_smooth` − `fast_smooth`): the acceleration itself, signed, braking at a stop |
| `start`, `bow`, `stop`, `wake` | the bandpass | the change projected on a direction and split by the region, at its height: the speed that started or stopped | the same pulse over (`slow_smooth` − `fast_smooth`): × dt over the pulse gives the Δv |

The outputs, in two subpanels, one per unit, since a mix holds one unit. Each output is a mix of
the six readings in that unit at shares 0 to 1, one knob per reading, and the fluid has one gain
per output, each meaning one thing:

| Subpanel     | Output        | Unit                       | The fluid                                                                        | The fluid's setting        | 1 means                               |
|--------------|---------------|----------------------------|----------------------------------------------------------------------------------|----------------------------|---------------------------------------|
| `boundary`   | `velocity`    | widths/s                   | drags the water toward it where it is non-zero, over `drag_time`, never past it  | `boundary.gain`            | the water as fast as the body         |
| `boundary`   | `density`     | widths/s, times the color  | adds it × dt into the slab: the brush, per width of travel                       | `density.boundary_gain`    | one unit of density per width         |
| `boundary`   | `temperature` | widths/s, by the color     | adds it × dt into the slab                                                       | `temperature.boundary_gain` | the same, signed by the color        |
| `body_force` | `acceleration`| widths/s²                  | adds it × dt                                                                     | `body_force.gain`          | a Δv added once                       |
| `body_force` | `density`     | widths/s², times the color | adds it × dt into the slab: the pipette, per Δv                                  | `density.body_force_gain`  | one unit of density per width/s of Δv |
| `body_force` | `temperature` | widths/s², by the color    | adds it × dt into the slab                                                       | `temperature.body_force_gain` | the same, signed by the color      |

The two subpanels are the two systems every solver distinguishes: a moving boundary, whose
velocity the water takes, and the body force, which accelerates it. So a wake may drag (its share
in `boundary.velocity`: the water behind a leaving limb is pulled to the speed it left with, and
let go when the pulse dies), kick (its share in `body_force.acceleration`), shed density as a brush
or throw it as a pipette, or any of these at once; the same reading in two outputs is a choice, not
an error. The change beside its parts lets a preset take the physics (the true acceleration,
braking at stops) or the look (a stop that pushes). Every reading's acceleration integrates to the
Δv over a change; they differ in their window: the lowpass's `attack` and `smooth`, the bandpass's
`fast_smooth` and `slow_smooth`.

On `BridgeOutput` and `Injection` the six are `boundary_velocity`, `boundary_density`,
`boundary_temperature`, `body_force_acceleration`, `body_force_density`, `body_force_temperature`.
The bridge's accelerations are the discrete form of the table's: the bandpass divides by the band's
width at the frame's dt and the lowpass uses the Ema's real step, the same integral, exact per
frame; FLOW.md states the form, this table the meaning.

### The drag

A body moving through water drags the water toward its own speed: the force is the difference,
`a = (v_body × gain − v) / drag_time`, an acceleration, added × dt. Where the fluid already moves
with the body the force is 0; where the body stops and the trail dies, nothing pulls, and the water
coasts on its own inertia. In the shader (`boundary_velocity_3d.comp`):
`v ← v + (v_body × gain − v) × pull`, with `pull = share × g × weight × slab`. `share` is
`1 − exp(−dt / drag_time)`, `drag_time` in ms, and 0 sets the velocity. `g` says where the body is:
the trail's speed above a floor, `min(|v_body| / DRAG_FLOOR, 1)`, `DRAG_FLOOR` a constant 0.02
widths/s, so a dead trail drags nothing toward 0; `weight` is the injection's, `slab` the gaussian.
Every factor is at most 1, so the injection never passes its target, whatever `gain`, `weight` or
the frame rate: it leaves the fluid at most as fast as the body × `gain`; what the simulation then
does with the velocity, the projection and the confinement, is its own. w = −z. The arithmetic is
in numpy beside the fluid (`drag_share`, `drag_gate`, `drag_velocity_step`, `test_drag.py`).

`speed` is 1: the density moves with the water. `self_advection` is 1: the velocity carries itself
the full step, the physics; viscosity and the confinement are the calming knobs. The lowpass's
`smooth` (1000 ms in the preset) makes the trail coast after a stop; with the drag that is the body
still dragging after it stopped, and the coast is the fluid's own inertia now ([Measuring](#measuring)
shows it at `smooth` 300), so the drag wants a shorter `smooth`, tuned on the preset.

### The pulse's area

The bandpass's pulse is the fast lowpass minus the slow one, two unit-gain Emas of the velocity.
An Ema lags its input by its time constant times the change, so for any change of speed, sudden
or gradual, the integral of their difference is (`slow_smooth` − `fast_smooth`) × Δv, 0.125 s at
25 and 150 ms; the height reads Δv only for a sudden change (`step_peak`). Dividing the pulse by
that time makes it an acceleration whose integral is the Δv, so × dt in the fluid adds the Δv
exactly once at any frame rate, with no history and no number from the bridge in the fluid. The
motion's `smooth` on z is one more linear filter in the chain, so the areas hold for z too; the
heights read z low by `smooth` against `fast_smooth`.

### The walls

The domain is a box with solid walls, as the textbook's (Bridson): beyond a closed wall the water
cannot pass, v·n = 0, and the pressure has no gradient across it, ∂p/∂n = 0. The projection's three
passes take the walls the same way, by a ghost beyond the last cell (`uClosed`, per axis): the
divergence a ghost velocity whose normal component is the last cell's turned, so the flux through the
wall's face is 0; the Jacobi solve and the gradient the last cell's pressure as the ghost; and the
gradient keeps the velocity from pointing into the wall, as into an obstacle's face. An open wall
takes the last cell's velocity as the ghost and 0 as the pressure: the water leaves. x and y are
closed or open by `pressure.wall`, z by `volume.wall`.

Before, the walls were the texture's: the velocity read 0 beyond the border and the pressure its
edge, while the obstacle code that keeps water out of a face ran only with an obstacle set. The
divergence then saw a face at the wall the gradient could not empty, so water pushed into a wall
stayed and was squeezed along it. Measured in a hidden window at the preset's grid, a body 50 px
from the right wall at 1 width/s, held 5 s, its band 300 px tall, at 25 fps:

| The body                 | The water's highest speed, before | after |
|--------------------------|-----------------------------------|-------|
| pushing into the wall    | 2.34 widths/s, in the wall's cells | 0.93 |
| sliding along the wall   | 3.96, in the corner               | 1.07  |

Confinement off changed neither (2.34 and 4.11 before), so the confinement was not the cause. With
the walls right, the water stays at the body's speed or under, the confinement adding a few
percent, MacCormack too (0.90 and 1.23).

### The guard

The dampen, at the step's first pass, stays as it is: `dampen_threshold` and `dampen_time`. With
the drag the injection cannot exceed the body's own speed, so what the dampen guards is the
simulation's own growth from the vorticity force, and a soft decay above a threshold fits that. If
the burst check ([Open](#open-the-burst-slows-the-fluid-to-2-to-4-fps)) still shows a problem, a
hard top speed (`velocity.limit`, the dampen shader with its factor 0) comes back as its own change.

### Names

The rule, in `.claude/rules/flow.md` scoped to `modules/flow/**` and
`modules/render/layers/flow/**`, with the terms in the "Flow" section of the glossary: the fluid's
things take the textbook word of fluid simulation where one exists; a reading is named by what it
is, a module by what it does; everything else by the quantity it is, nothing by a mechanism or a
role.

| Kind                                  | Names                                                                                                   | Unit                |
|---------------------------------------|---------------------------------------------------------------------------------------------------------|---------------------|
| the bridge's modules, by what they do | `lowpass` (`VelocityLowpass`), `bandpass` (`VelocityBandpass`)                                          |                     |
| the readings, by what they are        | `trail`; `change`, `start`, `bow`, `stop`, `wake`; each as a velocity and as an acceleration            | widths/s; widths/s² |
| the bridge's subpanels, the two systems | readings; `boundary`, the body as a moving boundary; `body_force`, the Navier–Stokes force term       |                     |
| the bridge's outputs, the fluid's inputs | `boundary.velocity`, `boundary.density`, `boundary.temperature`; `body_force.acceleration`, `body_force.density`, `body_force.temperature`; on `BridgeOutput` and `Injection` with `_` | widths/s; widths/s² |
| the mixes' knobs                      | shares, 0 to 1, one per reading in each output, summed                                                  |                     |
| the fluid's input groups              | `boundary` (`gain`, `drag_time`), `body_force` (`gain`); in `density` and `temperature` a `boundary_gain` and a `body_force_gain`; gains in the motion and the fluid, shares in the bridge | |
| the fluid's input methods             | `drag_velocity`, `add_acceleration`, `add_boundary_density`, `add_body_force_density`, `add_boundary_temperature`, `add_body_force_temperature` | |
| the fluid's fields                    | `velocity` (its own settings: fade, dampen, vorticity, viscosity, self-advection), `density`, `temperature`, `pressure`, `volume` | |
| the fluid's edge condition            | `wall` (`WallMode`), open or closed, in every field's group and the volume's, in both fluids; not `boundary`, which is the body |  |

"Drag" is what the fluid does with `boundary.velocity`, seen as `drag_time`, not the name of a
thing; "impulse", "body" alone, "force" alone, "model" and "delta" name nothing; "brush" and
"pipette" are the physics of the two densities, not names. A reading is what a module of the
bridge makes from the motion; not "signal", "band", "component", "event".

## Measuring

Tests go through the harness, `scripts/fluid_compare.py` beside `bridge_compare.py`: nothing saves
the bridge's outputs, so it chains the saved frames of a recording (`depth_velocity_capture`)
through the flow, a `Bridge3DLayer` and a `Fluid3DLayer` per side, A against B, with deep_flow's
preset and one setting changed by its panel path (`fluid.boundary.drag_time=0`,
`bridge.readings.lowpass.smooth=300`), and reads the fluid's velocity and density on the injection
layer's slice (`VolumeCut`). `--every N` steps the bridge and the fluid once per N frames with the
summed dt, a frame-rate check. Numpy only where the harness cannot reach: the drag's share and
floor as a formula (`test_drag.py`).

On stillslowfast (353 frames, 43 ms each) and standmoves (1356 frames, 44 ms), the water on the
moving player: its mean in the moves' fast halves over the body's (the trail's), the highest water
over the loop over the highest body, and the coast, s after a move's end until the water is under
20% of its fast mean; the body's coast beside it. The preset is `drag_time` 50 ms, `gain` 1,
`smooth` 1000 ms:

| Clip          | Side                 | Water / body, fast | Peak / body | Coast s, body / water | GPU ms, mean / p95 |
|---------------|----------------------|--------------------|-------------|-----------------------|--------------------|
| stillslowfast | preset               | 0.55               | 0.74        | 0.85 / 0.94           | 9.7 / 12.8         |
| stillslowfast | `drag_time` 0        | 0.61               | 1.18        | 0.85 / 0.94           | 9.8 / 12.8         |
| stillslowfast | `smooth` 300         | 0.56               | 0.66        | 0.23 / 0.94           | 9.7 / 12.6         |
| stillslowfast | `gain` 2             | 0.86               | 1.41        | 0.85 / 0.94           | 9.7 / 12.8         |
| standmoves    | preset               | 0.73               | 0.79        | 1.15 / 3.34           | 9.2 / 13.1         |
| standmoves    | `drag_time` 0        | 0.78               | 0.93        | 1.15 / 2.68           | 9.3 / 13.2         |

The water stays under the body at `gain` 1 and under twice it at `gain` 2 in the mean; the peak
over the loop passes 1 at `drag_time` 0 (1.18), which is the simulation's own doing after the
injection, the projection and the confinement **(deduction)**. With `smooth` 300 the body's coast
falls from 0.85 to 0.23 s and the water's stays at 0.94: the coast is the fluid's. The GPU time in
the harness is the step with its composite at 2160 × 3840, timed by `glFinish` around it.

The body force, the stop and the change as accelerations alone (`boundary.velocity.trail` 0), the
highest water mean around the stops, at the frames' rate and at every second frame:

| Reading as acceleration | Stop, 23 fps | Stop, 11.5 fps |
|-------------------------|--------------|----------------|
| `stop`                  | 0.032        | 0.025          |
| `change`                | 0.046        | 0.031          |

At half the rate the bridge is stepped at half the rate too, the bandpass's `fast_smooth` 25 ms
under the step, so the fluid's share of the difference is not isolated **(deduction)**; a check of
the fluid alone wants the bridge at the frames' rate and the fluid at every second step.

## Defaults

Every field gets the default its meaning says, so a fresh settings file reads as the design, not
as a tuning:

| Field                                                                            | Default         | Means                                   |
|----------------------------------------------------------------------------------|-----------------|-----------------------------------------|
| `boundary.gain`                                                                  | 1               | the water as fast as the body           |
| `boundary.drag_time`                                                             | 50 ms           | how fast the water takes the body's speed |
| `body_force.gain`                                                                | 1               | a Δv added once                         |
| `density.boundary_gain`, `density.body_force_gain`, `temperature.boundary_gain`, `temperature.body_force_gain` | 1 | one unit per width, per Δv |
| `speed`                                                                          | 1               | the density moves with the water        |
| `velocity.self_advection`                                                        | 1               | the velocity carries itself the full step |
| `width`, `height`, `volume.layers`                                               | 256, 448, 128   | half a width deep, within 60 fps        |
| `density.size`                                                                   | 1               | the density on the simulation's grid    |
| `volume.injection_layer`, `volume.injection_spread`                              | 0.5, 0.075      | the body mid-volume, 0.3 widths thick   |
| `wall` (every field, the volume)                                                 | closed          | as his settings                         |

The fields' physics take the textbook's water, with no artificial decay or thickness, and the preset
holds them: the velocity's `fade_time` 60 s (it keeps its momentum), `viscosity` 0 (water at this
scale), `pressure.iterations` 40 (GPU Gems' 40 to 80 for a Jacobi solve), the density's `fade_time`
60 s and `evaporation` 0, `temperature.sink` 0 and `ambient` 0 (the density as heavy as the water,
no buoyancy), gravity down the screen, every wall closed; the dampen a guard at 50 widths/s; the
bridge's mix the boundary's trail alone and its lowpass's `smooth` its default 160 ms, since the
coast is the fluid's now; the confinement Fedkiw's ε 0.3 over the neighbors (`vorticity_radius` 1);
in z, `injection_layer` 0.5 and `injection_spread` 0.075 widths, a body's thickness of about 0.3
widths at the camera's crop (from its `view_span` of 0.76 m, **(deduction)**) as four sigma, three
sigma inside the 0.5-width slab. Kept as tuned: the composite's look (`emission`,
`absorption`, `glow`, `tint`, `fog`).

Measured with these on standmoves through the harness: the water 0.89 of the body's speed in the
fast halves, its peak 0.76 of the body's, the step 11.5 ms with the harness (mean; p95 12.8).

### One unit

Every length in the fluid is in image widths, the motion's and the bridge's unit, and every time in
seconds: the cell h = 1 / the grid's width (the cells are cubic). The solvers' differences are over h
(the divergence, the gradient, the pressure's and the viscosity's Jacobi solves, the curl); the
confinement's force is ε × h × (N × ω) (Fedkiw 2001) with the curl over `vorticity_radius` cells;
the buoyancy is an acceleration, widths/s² per unit of density or temperature; the viscosity a
kinematic viscosity in widths²/s; `injection_spread` a thickness in widths. The advection moves a
texel by v × dt in uv as (1, width / height, 1 / slab depth) × v, x, y and z each a width per width.
The pressure is 32-bit: in these units it is small, past the half float's precision.

The output's allocation is not in it. Before, the cell was the grid's width over the output's (448 /
2160), so the confinement, the buoyancy and the viscosity followed the output's size, and y moved
(height / width)², 3.2 times, too far. Measured in a hidden window:

| Check                                                      | Before          | After          |
|------------------------------------------------------------|-----------------|----------------|
| 0.02 widths/s along y for 1 s: the density moved           | 0.064 widths    | 0.020          |
| a swirl with confinement, the same at a 1080 and a 2160 wide output: the largest velocity difference | 0.50 widths/s | 0.000 |

## Tuning

What remains is the preset, saved by him on the clips: `layers`, the gains, `drag_time`, the
lowpass's `smooth`, the mixes; judged on deep_flow with the standmoves and stillslowfast
recordings. Judge on: slow against fast reads; a stop coasts; a flick leaves one drop at its end
with the body force's density; the burst's fps; the wall.

## Open: the burst slows the fluid to 2 to 4 fps

FLOW.md's item: with `WARM_UP_TIME` 0 the exposure ramp reads as x/y motion over the whole image,
and the fluid's frame rate drops to 2 to 4 fps **(site fact)**. Nothing in the fluid loops by the
velocity: every pass runs a fixed number of dispatches, the pressure solve its 40 iterations, the
composite its 64 steps. What is data-dependent is where the advections read: a texel's backward
trace lands velocity × dt × `speed` away, and noise points neighbors apart **(deduction, not
measured)**. The shape takes the density from 1.6 GB to 0.18 GB a buffer, and the drag bounds the
speed a burst can reach, 2 to 4 widths/s at the motion's reach, so the slowdown should go with
them. The check: one run with `WARM_UP_TIME` 0 (locally, not committed). Only if the frame rate
still drops, a `GpuTimer` per pass, in the `GPU fluid` series, at rest, during the burst and
during a fast arm move; the pass that grows on the burst and not on the arm is the one, and a hard
top speed ([The guard](#the-guard)) the first thing to try.

## Next

Four steps, discussed one by one before any is built; each gets its decisions here as they are
made.

| Step | What                                                                                     | Judge on                                              |
|------|------------------------------------------------------------------------------------------|-------------------------------------------------------|
| 6    | Built, [MacCormack](#maccormack) | a drop's ring keeps turning; a thin thread stays thin; the step's GPU time |
| 7    | Built, [the particles](#step-7-the-particles); the weight (the splat back for the buoyancy) to come | a drop reads as a drop next to the grid's; 4K without sparkle at 60 fps |
| 8    | A better 4K upscale, parked ([Step 8](#step-8-the-4k-upscale-parked)) | the edges at 2160 × 3840 |
| 9    | Faster, one change at a time ([Step 9](#step-9-faster)), to be reviewed | the pass's GPU time, the physics and the picture the same |

### MacCormack

`advection` carries the velocity, the density and the temperature: `SEMI_LAGRANGIAN`, the default,
or `MACCORMACK` (Selle 2008, `maccormack_3d.comp`). The semi-Lagrangian step writes a predictor into
a scratch volume per field (allocated at its first MacCormack step); traced forward again, it gives
back the source up to the step's error, half of which is added back, clamped to the source's eight
texels around the backward trace. Where that trace reads past a wall the predictor stands: the
wall's value is in no texel, and the correction kept there grew the velocity at the walls step
after step, to eight times its speed in a second at the preset's grid (measured, before the guard).

Measured in a hidden window at 448 × 800 × 64, a swirl dragged in and left, then a square of
density carried for 2 s, no confinement, no viscosity; and on stillslowfast through the harness at
the preset:

| Measure                                        | Semi-Lagrangian | MacCormack |
|------------------------------------------------|-----------------|------------|
| the swirl's mean speed after 2 s, of its start | 0.52            | 0.62       |
| the density's edge, its 99.5% gradient over its max, per texel | 0.071 | 0.140 |
| the step, GPU ms, hidden window                | 5.7             | 6.7        |
| stillslowfast: water / body, fast              | 0.55            | 0.60       |
| stillslowfast: peak / body                     | 0.74            | 0.82       |
| stillslowfast: the step with the harness, ms   | 9.5             | 11.0       |

### Step 7: the particles

Built: the solver owns the velocity, the pressure and the temperature; the density is its carrier's
(`DensityCarrier`, `density_carrier.py`), `density.carrier` picking `GRID` (`GridDensity`, the default)
or `PARTICLES` (`ParticleDensity`), the one picked allocated. The field advection, semi-Lagrangian or
MacCormack, is one part both use (`FieldAdvection`). The particles' group is `particles`; the buoyancy
reads no weight from them yet, so with particles the density does not sink. Where a sprite would cover
the frame its birth radius stops at twice `radius`, and the reference mass carries the boundary's
density gain, so the gain sets how much ink there is, not how big a drop is.

Measured in a hidden window, the output at 2160 × 3840, the grid 448 × 800 × 64, the step with the
carrier's draw; with the ring full, and on stillslowfast through the harness at the preset (`rate` 2
for the blobby looks):

| `draw`, `overlap`    | Particles | Ring full, GPU ms | stillslowfast, ms mean / p95 |
|----------------------|-----------|-------------------|------------------------------|
| the grid             |           | 6.6               | 7.7 / 8.4                    |
| `POINTS`, occluded   | 32 M      | 21.3              | 16.3 / 24.7                  |
| `POINTS`, additive   | 32 M      | 12.7              |                              |
| `STREAKS`, occluded  | 32 M      | 39.5              | 13.8 / 24.5                  |
| `BLOBS`, occluded    | 1 M       | 26.9              | 17.9 / 43.0                  |
| `SURFACE`            | 1 M       | 31.9              | 20.1 / 54.1                  |

The cost is the draw's fill at 4K: the count for points and streaks, the radius for blobs and the
surface. `count`, `radius`, `slices` and the additive overlap are the knobs; the count each look
fits in 60 fps, measured in the app, is in [TIMING.md](TIMING.md#the-budget): 2 M points, the
default, 1 M streaks, 100 k blobs or spheres.

Decided: the grid density stays beside the particles, one setting picks the carrier; the first
look is the particles alone, the weight (the splat back into the grid for the buoyancy) after;
drawn as points or as streaks, one setting; the count a setting, sized by the memory the old 4K
density took (3.2 GB holds 100 M particles at 32 bytes), its default by the 60 fps budget; the oldest fade out
before the ring overwrites them; overlap first, ink a bit opaque, additive as the other option.

Everything stays on the GPU: the CPU dispatches and draws a fixed count, never reads a particle
back. The design **(deduction, to measure)**:

| Part     | How                                                                                       | Why                                                        |
|----------|-------------------------------------------------------------------------------------------|------------------------------------------------------------|
| storage  | one SSBO, 32 bytes a particle: position and age (vec4), the previous position (vec3), color RGBA8 packed in a uint; 32 M is 1 GB, 100 M 3.2 GB | each particle updates itself in place, so no swap pair     |
| emission | a compute pass over the bridge's density at the bridge's size: each texel's count, amount × rate × dt, rounded by a hash so a small amount still spawns; an atomic head (`atomicAdd`) gives the slots, modulo the count | a ring buffer: no free list, no compaction, no count read back |
| the ring's fade | a particle's place in the ring, (head − slot) mod N over N, is how near it is to being overwritten; its alpha falls to 0 over the ring's last share (`ring_fade`, 0.1), so the oldest fade out at whatever rate the body emits; the age's own fade by `density.fade_time` on top | no particle vanishes in a frame, whatever the emission |
| advection | a compute pass, 256 a group, RK2 (midpoint) through the velocity's `sampler3D`, the same uv metric as `advect_3d` (`uRdx`, `speed`); the walls clamp | two trilinear fetches a particle, the hardware's; the grid sets the motion |
| drawing  | no copy: a vertex shader pulls the SSBO by `gl_VertexID`; points `glDrawArrays(GL_POINTS, 0, N)`, streaks `GL_LINES` with 2 N vertices, the previous and the current position; a dead or unborn particle leaves the clip space | the rasterizer splats millions faster than atomics in compute, and draws lines, which compute would not |
| overlap (`OCCLUDED`, the default) | the particles drawn additively into depth slices, a `GL_TEXTURE_2D_ARRAY` at the output's size, a layered draw (`gl_Layer` from the particle's z): 16 slices at 2160 × 3840 RGBA16F, 1 GB, the color and the optical depth; then a front-to-back march through the slices, Beer–Lambert, `absorption`, `tint`, `fog` as the grid's composite | near ink covers far ink between slices, exact for emission and absorption inside one; a bucket sort by depth, no particle sorted |
| overlap (`ADDITIVE`) | the same draw into one RGBA16F target, each particle's fog and tint by its z; one pass: exp(−τ) and the emission | order-independent, one target; the depth only as fog |
| weight (later) | the particles splatted into an `r32ui` grid by fixed-point `imageAtomicAdd`, converted to the density volume the buoyancy reads | `imageAtomicAdd` needs integers |
| weight (later) | the particles splatted into an `r32ui` grid by fixed-point `imageAtomicAdd`, converted to the density volume the buoyancy reads | `imageAtomicAdd` needs integers |

Two more ways to draw the same particles, decided as options: blobby, where each particle carries
a smooth kernel and the kernels' sum melts near particles into one shape (Blinn's blobby objects,
metaballs). Far fewer particles, each covering many pixels: the cost moves from the count to the
kernel.

| `draw`    | What                                                                                     | Count, 60 fps  |
|-----------|------------------------------------------------------------------------------------------|----------------|
| `POINTS`  | one pixel a particle, into the slices or additive, above                                 | 2 M            |
| `STREAKS` | a line from the previous to the current position                                         | 1 M            |
| `BLOBS`   | the soft melt: each particle a gaussian sprite of `radius` pixels, its sum the density, marched through the slices as `POINTS` is; ink clouds whose drops merge | 100 k          |
| `SURFACE` | the liquid: screen-space fluid rendering (van der Laan 2009); each particle a sphere sprite of `radius`, its nearest depth into a depth target and its thickness summed into another; the depth smoothed by a filter that keeps its edges (a bilateral or narrow-range filter); the normals from the smoothed depth; then the ink's color through its thickness (Beer–Lambert, `absorption`, `tint`), a Fresnel edge, a specular and the background refracted by the normal; drops and threads with a sharp edge that merge and pinch off | 100 k          |

Each particle carries a mass, the density it took at its birth; its kernel sums to its mass, so
the ink is kept whatever the size. A blob's or a sphere's radius follows the mass,
`radius × (mass / reference)^(1/3)`, the cube root since the mass fills a volume: a strong drop a
big blob, a trace a small one. As it ages a blob widens and its peak falls by the same mass, as a
drop of ink spreads in water (`spread`, the diffusion: σ² grows by 2 × `spread` × dt). The
widening is refined: a small default, and a blob grows to at most twice its birth radius, so it
softens over its life and never bloats. A point's
and a streak's alpha is its mass.

| Part      | How                                                                                     |
|-----------|-----------------------------------------------------------------------------------------|
| emission  | each tick, at the bridge's size, the boundary's density (the brush, per width of travel) and the body force's (the pipette, per Δv), each at its gain: a texel spawns `rate × amount × dt` particles, rounded by a hash so faint ink still spawns now and then; each takes an even share of the texel's amount as its mass, the density's color, a random place in the texel and a z drawn from the injection slab's gaussian; no velocity of its own, the grid carries it and the drag gave the grid the body's |
| one knob  | a high `rate` many light particles, the points' and the streaks' look; a low `rate` few heavy ones, the blobs'; the mass and the size follow |
| duration  | three things end a particle, the first that comes: `density.fade_time`, the mass's exponential decay, as the grid density's; `evaporation`, subtracted, which takes faint ones to 0; the ring, once more are alive than `count`, the oldest fading over `ring_fade` before they are overwritten |
| the budget | in a steady state about `rate` × the emission × the lifetime are alive; while that stays under `count` the fade decides how long ink lasts and the ring is only the cap; the panel shows the seconds the ring takes to wrap, one counter read asynchronously, no particle |

`BLOBS` is the slice target with a bigger, softer point; `SURFACE` is its own pass, a depth and a
thickness target at the output's size, a filter of a few passes and one shading pass. Both depend
on `radius` against the particles' spacing: too small and the blobs fall apart into dots, too big
and the shape goes soft **(deduction)**.

The settings, a `particles` group: `count` (`Field.INIT`, 2 M), `rate` (particles per unit of
density per second), `ring_fade` (the ring's last share the oldest fade over), `draw`
(`ParticleDraw.POINTS`, `STREAKS`, `BLOBS` or `SURFACE`), `radius` (a blob's or a sphere's pixels at the reference mass), `spread` (a blob's widening with
age),
`smooth` (the surface's filter passes), `size` (a point's pixels), `streak` (a streak's length as a
share of its step), `overlap` (`Overlap.OCCLUDED` or `ADDITIVE`), `slices` (`Field.INIT`, 16).
The cost grows with the count by the advection's reads and the draw's fill, not by the memory: the
advection 0.2 ms a million, the draw faster than the count once many share a pixel
([TIMING.md](TIMING.md#the-particles)). Judge on: a
drop reads as a drop next to the grid's; a thread stays one particle wide at 2160 × 3840; near ink
covers far ink; the step's GPU time.

### Step 8: the 4K upscale, parked

Parked after steps 6 and 7. What was found:

- Many layers at 4K is what killed the old shape, and a cubic volume at 4K cannot be had: 64 cells
  deep at 448 wide is 64 × 4.8 ≈ 307 layers at 2160 wide, about 20 GB. So the detail at 4K does
  not come from a volume.
- A tricubic read in today's composite is 8 hardware reads a step instead of 1, 64 steps over
  2160 × 3840 pixels at 64 layers, about 4 G reads a frame, 5 to 10 ms **(deduction)**.
- The direction: depth in 3D at the grid, detail in 2D at 4K. The march at the grid's 256 × 448
  (0.11 M pixels instead of 8.3 M, about 70 times cheaper **(deduction)**), one edge-aware upscale
  of the flat result, and one 4K detail layer advected by the grid's velocity seen from the front
  with a curl noise where the water moves (Kim 2008 in 2D), multiplying the upscaled image so
  detail shows where ink is. The particles stay the other way to 4K in depth.
- The options, cheapest first: tricubic reads; a finer `density.size` with MacCormack; detail below
  the grid (the curl noise); an edge curve, a look rather than detail.

The grid's depth: 256 × 448 × 128, half a width deep, so the body's z motion has room before a wall.
A coarser grid across buys the depth: the detail it gives up in x and y is what the 4K layer and
the particles bring back, the trade the ink roadmap made at 128 × 224 × 96. The deeper grids
measured, and why this one, are in [TIMING.md](TIMING.md#the-grid).

### Step 9: faster

The changes of [TIMING.md's Faster](TIMING.md#faster), one at a time. Each is built behind its own
switch, measured against the baseline, and then kept one of two ways or dropped:

| Outcome                                               | Kept as                                                                   |
|-------------------------------------------------------|---------------------------------------------------------------------------|
| faster, and the same result                           | a replacement: the old path removed, no setting                           |
| faster, and a different result (a solver, a look)     | an option on the panel, as `advection` is, the current way staying the default until he picks |
| not faster                                            | dropped, its row kept in TIMING.md                                        |

The test, the same for every item: three timing runs back to back, baseline, change, baseline
(`-timing 60 -warmup 10`, the pose on, standmoves), the pass's own GPU timer the measure: the two
baselines' difference is the noise, and the change counts as faster when it beats the pass it
replaces by at least 2.5 times that; then
`scripts/fluid_compare.py` on stillslowfast, A the baseline, B the change, its numbers and the
composited output side by side (a `--save` to add). The same numbers and the same picture: a
replacement. Different: an option, judged on the look in the app.

The order puts the sure and cheap ones first, each an hour or two, then the solvers, each a day, then
the structural one; the particles after the fluid, since their draw is already within the budget.
Each item commits on its own when it is in, and writes its measured row into TIMING.md.

| # | Change                                                      | Expected                        | Kept as, if faster                                         |
|---|-------------------------------------------------------------|---------------------------------|------------------------------------------------------------|
| 1 | the temperature cleared once, its injection skipped at 0    | 0.4 to 0.5 ms                   | replacement                                                |
| 2 | the composite's early exit and the occupancy volume         | 0.5 to 1.5 ms                   | replacement                                                |
| 3 | 3D workgroups for the stencil passes                        | 0.3 to 0.6 ms                   | replacement                                                |
| 4 | MacCormack in one dispatch                                  | 0.4 to 0.6 ms                   | replacement                                                |
| 5 | fused passes                                                | 1 to 1.5 ms                     | replacement                                                |
| 6 | red-black SOR                                               | 1.5 to 2.5 ms                   | option, decided: `pressure.solver`, `JACOBI` (the default), `SOR`, whatever it measures |
| 7 | multigrid, or MGPCG                                         | 2.5 to 3.5 ms                   | option, decided: the same option gains `MULTIGRID`          |
| 8 | the sparse grid                                             | 0 to half the step              | its plan written, not built: it touches every pass         |
| 9 | the sprites cut at 2 sigma                                  | half the blobs' fill            | replacement; the growth cap stays as it is, a look          |
| 10 | the draw by the alive count                                | the ring's empty share          | replacement                                                |
| 11 | weighted blended OIT                                       | the march, the clear, a third of the draw | option: `overlap` gains `WEIGHTED`                |
| 12 | points and streaks rasterized in compute                   | 2 to 5 × on the draw            | replacement for points; streaks if the lines match         |
| 13 | the narrow-range filter and the narrow band                | 0.3 to 1 ms                     | the band a replacement; the filter an option if the surface differs |
| 14 | the Morton sort                                            | under 0.5 ms at 2 M             | replacement; the ring's fade must then go by the particle's age, not its slot |

Beside 5 and 6, the memory a step: the bytes read and written per cell by every pass, counted from
the shaders before and after, since the passes are bandwidth-bound and the three force and curl
volumes (24 bytes a cell) exist only between passes. A winning solver becomes the preset's
`pressure.solver` too. Each kept item is one commit, none pushed.

The panel gains at most three options from it: `pressure.solver`, `overlap`'s `WEIGHTED`, and the
surface's filter if it comes. Everything else is the same picture faster, and leaves no trace but
its row in TIMING.md.

## Later

- The 2D fluid modeled after this one: the drag, the two systems, the one grid.
- hd_trio's `density.fade_time` written per tick from the pose's motion: the 2D fluid's, not here.
