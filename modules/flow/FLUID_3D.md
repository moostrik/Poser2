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
| dampen_vel  | drag on the speed above `dampen_threshold` (50 widths/s) over `dampen_time` (2 s)     | velocity 448×800×64   | no                   |
| advect_vel  | semi-Lagrangian self-advection, `self_advection` 1, fade 6 s                          | velocity              | the trace distance   |
| viscosity   | Jacobi diffusion, off (`viscosity` 0)                                                 | velocity              | no                   |
| vorticity   | curl, confinement force × dt, added to the velocity                                   | velocity, curl, force | no                   |
| advect_temp | off (`buoyancy` 0 clears the temperature)                                             | temperature           | the trace distance   |
| buoyancy    | gravity × (sink × density − buoyancy × ΔT) × dt; gravity is 0 in the preset           | force                 | no                   |
| incompress  | divergence, 20 Jacobi iterations, gradient subtraction                                | pressure, velocity    | no                   |
| advect_den  | semi-Lagrangian advection, one trilinear read per texel, the evaporation subtracted   | density 448×800×64    | the trace distance   |
| composite   | front-to-back march, one step per layer                                               | output 2160×3840      | no                   |

Every field is a `GL_TEXTURE_3D` on one grid of cubic cells at the stage's size, 448 × 800 by
`volume.layers` 64: the velocity RGBA16F, the temperature and the pressure R16F, the density RGBA16F
with its color baked in, each a swap pair, the obstacle R8. The simulation's cell is 1/448 width in
x, y and z, so the slab is 64 / 448 = 0.14 widths deep (`_z_extent`). The density is on the same
grid, or `density.size` times finer in x and y (1, 2 or 4): every pass that reads the velocity or
the obstacle against it samples by uv, so the shaders are the same (`advect_3d`, `inject_3d`,
`buoyancy_3d`). The composite reads the volume at the output's 2160 × 3840 and upsamples it
bilinearly. The monitors read every field as it is. The GPU is a 24 GB RTX 4090, so nothing pages.

| `layers` | `density.size` | density, one buffer | step, GPU ms |
|----------|----------------|---------------------|--------------|
| 64       | 1              | 448 × 800 × 64, 0.18 GB | 6.6      |
| 64       | 2              | 896 × 1600 × 64, 0.73 GB | 7.2     |
| 24       | 4              | 1792 × 3200 × 24, 1.1 GB | 4.8     |

The step's GPU time, measured in a hidden window on a swirl at the preset's passes, the composite at
2160 × 3840. `layers` is the knob if it is too much; `density.size` 4 is the 4K look on the same
path, with the z lerp across coarse layers and the scattered reads of a big volume.

What the layer injects before the step (`Fluid3DLayer._inject`), each at the injection's `weight`,
into the gaussian slab at `injection_layer` 0.54, `injection_spread` 0.1:

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
| `volume.layers`                                                                  | 64              | cubic cells over the slab               |
| `density.size`                                                                   | 1               | the density on the simulation's grid    |
| `volume.injection_layer`, `volume.injection_spread`                              | 0.5, 0.1        | the body mid-volume                     |
| `wall` (every field, the volume)                                                 | closed          | as his settings                         |

The field's own settings (`fade_time`, `dampen_*`, `vorticity*`, `viscosity*`, the composite's)
keep their defaults; they are looks, not meanings.

## Tuning

What remains is the preset, saved by him on the clips: `layers`, the gains, `drag_time`, the
lowpass's `smooth`, the mixes; judged on deep_flow with the standmoves and stillslowfast
recordings. Judge on: slow against fast reads; a stop coasts; a flick leaves one drop at its end
with the body force's density; the burst's fps; the wall.

## Open: the burst slows the fluid to 2 to 4 fps

FLOW.md's item: with `WARM_UP_TIME` 0 the exposure ramp reads as x/y motion over the whole image,
and the fluid's frame rate drops to 2 to 4 fps **(site fact)**. Nothing in the fluid loops by the
velocity: every pass runs a fixed number of dispatches, the pressure solve its 20 iterations, the
composite its 64 steps. What is data-dependent is where the advections read: a texel's backward
trace lands velocity × dt × `speed` away, and noise points neighbors apart **(deduction, not
measured)**. The shape takes the density from 1.6 GB to 0.18 GB a buffer, and the drag bounds the
speed a burst can reach, 2 to 4 widths/s at the motion's reach, so the slowdown should go with
them. The check: one run with `WARM_UP_TIME` 0 (locally, not committed). Only if the frame rate
still drops, a `GpuTimer` per pass, in the `GPU fluid` series, at rest, during the burst and
during a fast arm move; the pass that grows on the burst and not on the arm is the one, and a hard
top speed ([The guard](#the-guard)) the first thing to try.

## Open: the velocity explodes at a wall

Seen on 2026-11-06 **(site fact)**, before the shape and the drag: when the body crosses a wall of
the volume the fluid's velocity sometimes explodes, while the bridge's `boundary_velocity` on the
monitor stays calm, so the cause is the fluid's. Not checked **(deduction)**: the velocity is 0 at
the walls (`GL_CLAMP_TO_BORDER`), so a push into the cells right at a wall leaves a strong
divergence there; if the pressure solve's 20 iterations do not cancel it, it grows step after
step, and the vorticity confinement may amplify it at the wall. His walls are all closed **(site
fact)**: the velocity's wrap is `GL_CLAMP_TO_BORDER` 0, the other fields' `GL_CLAMP_TO_EDGE`, a
zero gradient, which the Jacobi solve reads the same way (`loadPressure` at clamped coordinates).
Whether it remains with the drag is his check on the app. If it does, the places to look are the
divergence and the pressure at the wall on the monitors and the confinement near the border.

## Later

- The 4K density by the particle carrier, beside `density.size`; judged on the same clips at the
  output's size once the fluid is right.
- The advection's order: MacCormack beside semi-Lagrangian for the velocity (the roadmap's B2);
  measured on how long a drop's ring keeps turning.
- The particle carrier (the roadmap's phase C), once the grid's drops read right; it needs the
  roadmap's B1 split first, the velocity solver apart from the density carrier, which is one
  field's move now that the density is a `Texture3D` on the grid.
- The 2D fluid modeled after this one: the drag, the two systems, the one grid.
- hd_trio's `density.fade_time` written per tick from the pose's motion: the 2D fluid's, not here.
