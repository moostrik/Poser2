# The 3D fluid

The plan for the fluid refactor: what the 3D fluid is today, how it takes what the bridge gives it, what
goes, and the steps in build order. It becomes the 3D fluid's document once built, as
[FLOW.md](FLOW.md) is the flow's. "Code" is what the files show; **(deduction)** marks what is
inferred from them, **(decision)** a choice to make before the step is built.

The 2D fluid (`fluid_2d/`, `Fluid2DLayer`, hd_trio) is out of scope: nothing here changes it, and
hd_trio looks as it does. The 3D fluid is `Fluid3D` (`fluid_3d/fluid_3d.py`), its layer
`Fluid3DLayer` (`modules/render/layers/flow/fluid_3d_layer.py`), run by deep_flow alone.

## Today

A step (`Fluid3D.update`), in order, at the preset of 2026-11-06 (`apps/deep_flow/data/settings/settings.json`):

| Pass        | What                                                                                  | On                    | Depends on the input |
|-------------|---------------------------------------------------------------------------------------|-----------------------|----------------------|
| dampen_vel  | drag on the speed above `dampen_threshold` (50 widths/s) over `dampen_time` (2 s)     | velocity 448×800×16   | no                   |
| advect_vel  | semi-Lagrangian self-advection, `self_advection` 0.51, fade 6 s                       | velocity              | the trace distance   |
| viscosity   | Jacobi diffusion, off (`viscosity` 0)                                                 | velocity              | no                   |
| vorticity   | curl, confinement force × dt, added to the velocity                                   | velocity, curl, force | no                   |
| advect_temp | off (`buoyancy` 0 clears the temperature)                                             | temperature           | the trace distance   |
| buoyancy    | gravity × (sink × density − buoyancy × ΔT) × dt; gravity is 0 in the preset           | force                 | no                   |
| advect_pres | pressure advection, `pressure.speed` 0.66                                             | pressure              | the trace distance   |
| incompress  | divergence, 20 Jacobi iterations, gradient subtraction                                | pressure, velocity    | no                   |
| advect_den  | semi-Lagrangian advection, two bilinear reads and a z lerp per texel                  | density 2160×3840×24  | the trace distance   |
| composite   | front-to-back march, 48 steps per pixel                                               | output 2160×3840      | no                   |

The simulation fields are `GL_TEXTURE_3D` at the stage's size, 448 × 800 by `sim_layers` 16: the
velocity RGBA16F, temperature and pressure R16F, each a swap pair. The density is a
`GL_TEXTURE_2D_ARRAY` at the layer's allocation, 2160 × 3840 (deep_flow allocates its large layers
at twice the output) by `layers` 24, RGBA16F, a swap pair: 1.6 GB each, 3.2 GB in all. The
density's monitor port is its reduction to 448 × 800 × 24 (`ReduceDensity`), refreshed when a view
asks. The GPU is a 24 GB RTX 4090, so nothing pages.

What the layer injects before the step (`Fluid3DLayer._inject`), each at the injection's `weight`:

| Input                  | Today                                                                                                                                                | Shader               |
|------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------|
| `velocity` (the trail) | added to the velocity: × `gain` × `input_x` / `input_y` on x and y × dt, w = −z, in the gaussian slab at `injection_layer` 0.54, `injection_spread` 0.1 | `inject_flow_3d`     |
| `impulse`              | the same call: added × `gain` × dt                                                                                                                   | `inject_flow_3d`     |
| `spin`                 | a 2D curl for the vorticity force; no bridge gives one, so never                                                                                     | `vorticity_force_3d` |
| `density` (RGBA)       | added × `density.gain` × dt into the slab                                                                                                            | `inject_3d_density`  |
| `temperature`          | added × `temperature.gain` × dt into the slab                                                                                                        | `inject_3d`          |

The velocity is in image widths per second, as the bridge's; the advection moves a texel by
velocity × dt × `speed` (0.4) in uv, so 1 width/s crosses the width in 2.5 s at the preset.

## The shape

Two grids today, neither cubic. The simulation's cell is 1/448 width in x and y and `spacing` 4
cells in z, so the slab is 16 × 4 / 448 = 0.14 widths deep. The density spans the same slab in 24
layers at 2160 × 3840: its z step is 12.9 of its own x texels, and each density layer covers 2.7
simulation layers, so the density's z lerp blurs across them every frame **(deduction)**. The ink
notes name this the one thing the particles were to replace: the velocity can be made isotropic,
the density array cannot without 133 MB a layer.

| Field                 | Today                                           | Cell x : z | One grid                        | Cell x : z |
|-----------------------|-------------------------------------------------|------------|---------------------------------|------------|
| velocity              | 448 × 800 × 16, `GL_TEXTURE_3D`                 | 1 : 4      | 448 × 800 × 64, `GL_TEXTURE_3D` | 1 : 1      |
| density               | 2160 × 3840 × 24, `GL_TEXTURE_2D_ARRAY`, 3.2 GB | 1 : 12.9   | the same grid, 0.37 GB          | 1 : 1      |
| temperature, pressure | 448 × 800 × 16                                  | 1 : 4      | 448 × 800 × 64                  | 1 : 1      |

One grid for every field, cubic cells: `spacing` 1, `layers` 64 keeps today's slab of 0.14 widths
(the velocity's z extent, `_z_extent`, unchanged). The density is a `GL_TEXTURE_3D` on it, so
every field takes the same shaders (`advect_3d`, `inject_3d`, `buoyancy_3d`), a trilinear read is
one fetch, and the 2D-array family (`advect_3d_density`, `inject_3d_density`,
`inject_channel_3d_density`, `buoyancy_3d_density`, `composite_3d_density`, `reduce_density`) goes.
`buoyancy_3d` keeps `buoyancy_3d_density`'s body (the density's max of r, g, b, the gravity vector,
the weight wherever density is); today's `buoyancy_3d.comp` has no caller and reads a scalar
density. The composite reads the volume at the output's 2160 × 3840 and upsamples 4.8 × bilinearly: soft, and
that is the test. The monitors read the density as it is; the reduced view goes. `volume.layers`
and `sim_layers` become one `layers`. The simulation grows from 5.7 M to 23 M cells, four times the
pressure solve; the `GPU fluid` series says what it costs **(to measure)**, and `layers` is the
knob if it is too much.

The 4K look stays as an option on the same path: a `GL_TEXTURE_3D` read by the same shaders can
be allocated at any x/y size, since the advection samples the velocity by normalized coordinates
and the injection and the composite by uv. One setting, the density's size as a multiple of the
simulation's, 1, 2 or 4, `layers` shared; 1 is the cubic grid and the default, 4 is today's 4K
look on a 3D texture, no second obstacle volume since the obstacle is sampled the same way. Its
costs come with it: at 4 and 64 layers 4.2 GB a buffer, so 4K wants fewer layers (24 as today,
1.6 GB), the z lerp across coarse layers, and the burst's scattered reads over a big volume. The
particle carrier stays the other way to 4K ([Later](#later-not-in-this-plan)).

The volume's settings with the shape, eighteen today, nine after: deep_flow is the app that makes
the fluid, not a show, so what stays is what the fluid needs and the look he uses, emission and
absorption with a small glow, a light water tint and fog.

| Setting                                                  | Becomes                                                      |
|----------------------------------------------------------|--------------------------------------------------------------|
| `layers`, `sim_layers`                                   | one `layers`, 64                                             |
| `spacing`                                                | goes: the cells are cubic                                    |
| `debug_layers`                                           | goes: the monitors cut the volume in 3D                      |
| `gamma`, `layer_blur`, `perspective`, `parallax`, `eye_tracking`, `drift_period` | go, with deep_flow's head-tracked eye (`set_eye`, `_eye`) and the layer legend |
| `injection_layer`, `injection_spread`                    | stay: where the body sits in the volume                      |
| `emission`, `absorption`, `glow`, `tint`, `fog`          | stay: the composite's march, with the glow, tint and fog he uses |
| `boundary`                                               | `wall`                                                       |

## The inputs

The rule, by what each input is. Today both fluids add the injected velocity × dt, as Stam adds a
source; but the trail is a velocity in the fluid's own unit, so added × dt it acts as an
acceleration and `gain` as a rate. In Navier–Stokes the body force is an acceleration per unit
mass; a body in the fluid is a boundary that moves.

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
`boundary_temperature`, `body_force_acceleration`, `body_force_density`, `body_force_temperature`,
built as 76a500c5. The bridge's accelerations are the discrete form of the table's: the bandpass
divides by the band's width at the frame's dt and the lowpass uses the Ema's real step, the same
integral, exact per frame; FLOW.md states the form, this table the meaning.

### The drag

A body moving through water drags the water toward its own speed: the force is the difference,
`a = (v_body × gain − v) / drag_time`, an acceleration, added × dt. Where the fluid already moves
with the body the force is 0; where the body stops and the trail dies, nothing pulls, and the water
coasts on its own inertia. In the shader: `v ← v + (v_body × gain − v) × pull`, with
`pull = share × g × weight × slab`. `share` is `1 − exp(−dt / drag_time)`, `drag_time` in ms, and 0
sets the velocity. `g` says where the body is: the trail's speed above a floor,
`min(|v_body| / DRAG_FLOOR, 1)`, a constant floor of 0.02 widths/s **(decision: constant or
setting)**, so a dead trail drags nothing toward 0; `weight` is the injection's, `slab` the
gaussian. The drag can never pass its target, whatever `gain`, `weight` or the frame rate: the
fluid is at most as fast as the body × `gain`. w = −z as today.

The drag replaces today's add, with no toggle: the recordings and git compare the two looks. The
preset's `gain` of 10 was a rate and does not carry over; 1 is as fast as the body. Two
settings on the advection stay: `speed`, how far the density and the temperature travel per step
against the water, 1 = with the water, 0.4 in the preset a look where they lag it; and
`self_advection`, how far the velocity carries itself per step, 1 = the full step, the physics;
0.51 in the preset was a calming factor, which viscosity and the confinement do honestly. `speed` goes to
1.0 with the drag, so the density moves with the water. The drag's formula, the share and the
floor gate, gets a numpy reference tested without GL, as the trail has (`velocity_trail`); the
rest is measured on the harness ([Steps](#steps)). The lowpass's
`smooth` (1000 ms in the preset) makes the trail coast after a stop; with the drag that is the
body still dragging after it stopped, and the coast is the fluid's own inertia now, so the drag
wants a shorter `smooth` **(deduction)**, tuned on the preset. A burst of false velocity ([Open](#open-the-burst-slows-the-fluid-to-2-to-4-fps))
cannot run away through the drag: the fluid never exceeds the burst's own speed, 2 to 4 widths/s
at the motion's reach, where the add at `gain` 10 integrates it to tens of widths per second
**(deduction)**.

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
the burst check after the drag still shows a problem, a hard top speed (`velocity.limit`, the
dampen shader with its factor 0) comes back as its own change.

### Names

The rule, for `.claude/rules/flow.md` scoped to `modules/flow/**` and
`modules/render/layers/flow/**`, with the terms in a "Flow" section of the glossary: the fluid's
things take the textbook word of fluid simulation where one exists; a reading is named by what it
is, a module by what it does; everything else by the quantity it is, nothing by a mechanism or a
role.

| Kind                                  | Names                                                                                                   | Unit                |
|---------------------------------------|---------------------------------------------------------------------------------------------------------|---------------------|
| the bridge's modules, by what they do | `lowpass` (`VelocityLowpass`, once the trail bridge), `bandpass` (`VelocityBandpass`, once the impulse bridge) | |
| the readings, by what they are        | `trail`; `change`, `start`, `bow`, `stop`, `wake`; each as a velocity and as an acceleration            | widths/s; widths/s² |
| the bridge's subpanels, the two systems | readings; `boundary`, the body as a moving boundary; `body_force`, the Navier–Stokes force term       |                     |
| the bridge's outputs, the fluid's inputs | `boundary.velocity`, `boundary.density`, `boundary.temperature`; `body_force.acceleration`, `body_force.density`, `body_force.temperature`; on `BridgeOutput` and `Injection` with `_` | widths/s; widths/s² |
| the mixes' knobs                      | shares, 0 to 1, one per reading in each output, summed                                                  |                     |
| the fluid's input groups              | `boundary` (`gain`, `drag_time`), `body_force` (`gain`); in `density` and `temperature` a `boundary_gain` and a `body_force_gain`; gains in the motion and the fluid, shares in the bridge | |
| the fluid's fields                    | `velocity` (its own settings: fade, dampen, vorticity, viscosity, self-advection), `density`, `temperature`, `pressure`, `volume` | |
| the fluid's edge condition            | `wall`, open or closed, in every field's group and the volume's; not `boundary`, which is the body    |                     |

"Drag" is what the fluid does with `boundary.velocity`, seen as `drag_time`, not the name of a
thing; "impulse", "body" alone, "force" alone, "model" and "delta" name nothing; "brush" and
"pipette" are the physics of the two densities, not names. A reading is what a module of the
bridge makes from the motion; not "signal", "band", "component", "event".

## Open: the burst slows the fluid to 2 to 4 fps

FLOW.md's item: with `WARM_UP_TIME` 0 the exposure ramp reads as x/y motion over the whole image,
and the fluid's frame rate drops to 2 to 4 fps **(site fact)**. Nothing in the fluid loops by the
velocity: every pass runs a fixed number of dispatches, the pressure solve its 20 iterations, the
composite its 48 steps. What is data-dependent is where the advections read: a texel's backward
trace lands velocity × dt × `speed` away, noise points neighbors apart, and the density's advection
over 1.6 GB then reads scattered **(deduction, not measured)**. At the preset the burst, integrated
at `gain` 10 for a second, reaches tens of widths per second: traces of hundreds of texels over the
whole volume.

The shape takes the density from 1.6 GB to 0.18 GB a buffer, and the drag bounds the speed a
burst can reach, so the slowdown should go with them. The check comes after step 3: one run with
`WARM_UP_TIME` 0 (locally, not committed). Only if the frame rate still drops, a `GpuTimer` per
pass, in the `GPU fluid` series, at rest, during the burst and during a fast arm move; the pass
that grows on the burst and not on the arm is the one, and a hard top speed
([The guard](#the-guard)) the first thing to try.

## Open: the velocity explodes at a wall

Seen on 2026-11-06 **(site fact)**: when the body crosses a wall of the volume the fluid's velocity
sometimes explodes, while the bridge's `boundary_velocity` on the monitor stays calm, so the cause
is the fluid's. Not checked **(deduction, the bridge thread's)**: the velocity is 0 at the walls
(`GL_CLAMP_TO_BORDER`), so a push into the cells right at a wall leaves a strong divergence there;
if the pressure solve's 20 iterations do not cancel it, it grows step after step, and the vorticity
confinement may amplify it at the wall. His walls are all closed **(site fact)**: the velocity's
wrap is `GL_CLAMP_TO_BORDER` 0, the other fields' `GL_CLAMP_TO_EDGE`, a zero gradient, which the
Jacobi solve reads the same way (`loadPressure` at clamped coordinates). Not a step: the fluid
changes first, and the check after the drag (step 3) says whether this remains. If it does, the
places to look are the divergence and the pressure at the wall on the monitors and the
confinement near the border.

## Cleanup

Form only, no change to the look, before the injection work so it lands on a clean fluid:

- Inputs nobody calls: `set_velocity`, `set_density`, `set_temperature`, `add_pressure`,
  `add_density_channel` with `InjectChannel3D` and `InjectChannel3DDensity`; the spin: `set_spin`,
  `_spin`, `_spin_strength`, `Injection.spin`, `Fluid3DLayer`'s use, and the spin function and
  uniforms of `vorticity_force_3d.comp`.
- Obstacles stay, though `set_obstacle`, `add_obstacle` and `clear_obstacles` have no caller
  today: the fluid supports them, and the shaders keep their obstacle branches. With one grid
  ([The shape](#the-shape)) the density's own R8 obstacle volume (200 MB) goes, since the
  simulation's serves every field.
- `input_x` and `input_y` go: the motion's `x_gain` and `y_gain` balance the axes. The preset holds
  both at 1, so nothing changes.
- The edge condition `boundary` (`BoundaryMode`, in `density`, `temperature`, `pressure` and
  `volume`) becomes `wall`, with the preset's four keys; `boundary` is the body's. The enum is the
  2D fluid's too, so its three `boundary` fields and hd_trio's presets' keys (`studio.json`,
  `umu.json`) rename with it; hd_trio looks the same.
- Properties nobody reads: `velocity_volume`, `velocity`, `density_volume`, `temperature_volume`,
  `pressure_volume`, `obstacle_volume`, `density_obstacle_volume`, `divergence_volume`,
  `curl_volume`, `depth`, `sim_width`, `sim_height`, `allocated`; the ports are the way out.
- Not part of a fluid simulation: the pressure advection (`pressure.speed`, `pressure.fade_time`,
  the `advect_pres` pass, "optional, non-physical" in its own words; the pressure is solved fresh
  each step) and the `speed_offset` of the density and the temperature, an extra speed per field
  on top of `speed`, 0 in his settings. `self_advection` stays as a setting, at 1: the velocity carries its own
  momentum the full step, as the physics has it; viscosity and the confinement are the calming
  knobs.
- `profile` and the `Profiler` go: a `glFinish` after every pass and a wall clock, which
  serializes the GPU; the `GpuTimer` around the step (timer queries, the `GPU fluid` series,
  `timing.bat`) is the way to measure, and a timer per pass is added only when a check needs it.
- "ink" in docstrings and shader comments becomes density.
- `.claude/rules/flow.md` and the glossary's "Flow" section, from [Names](#names).

Each removed field leaves the preset in the same change (`test_flow_systems.py` fails on an
unknown key); the preset is his to save, so a step that removes a field names its keys. deep_flow's
second preset, `event.json`, goes in step 1, so `settings.json` is the one preset to keep. FLOW.md's
Open items on `input_x` and `input_y`, the spin and the reduced view close in the step that
settles them, and MONITOR.md's reduced-density section goes with the reduction.

## Defaults

Every field a step touches gets the default its meaning says, so a fresh settings file reads as
the design, not as a tuning:

| Field                                                                            | Default         | Means                                   |
|----------------------------------------------------------------------------------|-----------------|-----------------------------------------|
| `boundary.gain`                                                                  | 1               | the water as fast as the body           |
| `boundary.drag_time`                                                             | 50 ms           | how fast the water takes the body's speed |
| `body_force.gain`                                                                | 1               | a Δv added once                         |
| `density.boundary_gain`, `density.body_force_gain`, `temperature.boundary_gain`, `temperature.body_force_gain` | 1 | one unit per width, per Δv |
| `speed`                                                                          | 1               | the density moves with the water        |
| `velocity.self_advection`                                                        | 1               | the velocity carries itself the full step |
| `volume.layers`                                                                  | 64              | cubic cells over today's slab           |
| the density's size against the simulation's                                     | 1               | the density on the simulation's grid    |
| `volume.injection_layer`, `volume.injection_spread`                              | 0.5, 0.1        | the body mid-volume                     |
| `wall` (every field, the volume)                                                 | closed          | as his settings                         |

The field's own settings (`fade_time`, `dampen_*`, `vorticity*`, `viscosity*`, `self_advection`,
the composite's) keep their defaults; they are looks, not meanings.

## Steps

In build order, each one change, each judged on deep_flow with the standmoves and stillslowfast
recordings; `go` per step.

| Step | What                                                                                       | Touches                                                                                                        | Judge on                                              |
|------|--------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------|-------------------------------------------------------|
| 1    | Cleanup: the dead inputs, spin, properties, `input_x` and `input_y`, `boundary` to `wall` in both fluids, the pressure advection, the speed offsets, the profiler, the words, the rules file | `fluid_3d.py`, `fluid_settings.py`, `boundary_mode.py`, the shaders, `fluid_layer.py`, `fluid_3d_layer.py`, `shaders/__init__.py`, the preset's eleven keys, `event.json` (goes), `fluid_2d`'s settings and hd_trio's presets' `boundary` keys, FLOW.md's Open items, `.claude/rules/flow.md`, the glossary | looks the same; tests pass |
| 2    | The shape: one cubic `GL_TEXTURE_3D` grid, the density on it, the 2D-array shaders go, the density's size a multiple of the simulation's (1, 2, 4), the volume's settings cut to nine plus it, the composite to emission, extinction, glow, tint and fog | `fluid_3d.py`, `fluid_settings.py`, the shaders, `composite_3d`, the monitors' density port, `fluid_3d_layer.py` (the legend, `set_eye`), deep_flow's `render.py` (`_eye`), the preset's `volume` keys, MONITOR.md | a drop keeps its shape in z on the monitors; the `GPU fluid` time |
| 3    | The boundary side: the harness first; the drag replaces the add; the group `boundary` (`gain`, `drag_time`), the floor; `speed` and `self_advection` to 1; `velocity` keeps the field's settings | `scripts/fluid_compare.py`, `inject_flow_3d.comp`, `fluid_3d.py`, `fluid_settings.py`, `fluid_3d_layer.py`, the preset's `velocity` keys and `speed` | a first setting of `boundary.gain`, `drag_time` and the lowpass's `smooth`, then before and after on a clip: slow against fast reads; a stop coasts; the burst's fps; the wall |
| 4    | The body force side: `add_acceleration` × dt with `body_force.gain`; the second density and temperature with `boundary_gain` and `body_force_gain` | `fluid_3d.py`, `fluid_3d_layer.py`, `fluid_settings.py`, the preset's `density` and `temperature` keys | a flick leaves one drop at its end; a stop one |
| 5    | Tune the preset: `layers`, the gains, `drag_time`, the lowpass's `smooth`, the mixes      | the preset, saved by him                                                                                       | the clips                                             |

Step 1 changes no look; 2 trades the density's 4K for its shape and is the one step with a cost
risk, four times the pressure solve, read off the `GPU fluid` series first; 3 and 4 change how
the fluid takes the bridge's two sides, judged on the clips before and after. Every step edits
the preset's keys, so the preset is saved before each. After 5 this document loses "plan" and
keeps the rules and the numbers.

Tests go through the harness: a `scripts/fluid_compare.py` beside `bridge_compare.py`. Nothing
saves the bridge's outputs, so it chains as `bridge_compare.py` does, the saved frames of a
recording through the flow and the bridge into the fluid, A against B, with a read-back of a 3D
texture (`read_back` reads 2D), measuring what a step claims: for the drag the water's peak speed against the body's (1 at `boundary.gain` 1, never
above) and the coast after a stop; for the body force a stop's Δv added once, the same at 30 and
60 fps; for the densities the amount per width of travel and per Δv; for the shape the `GPU fluid`
time. Numpy only where the harness cannot reach: the drag's share and floor as a formula.

## Later, not in this plan

- The 4K density by the particle carrier, beside the density's size option; judged on the same
  clips at the output's size once the fluid is right.
- The advection's order: MacCormack beside semi-Lagrangian for the velocity (the roadmap's B2);
  measured on how long a drop's ring keeps turning.
- The particle carrier (the roadmap's phase C), once the grid's drops read right; it needs the
  roadmap's B1 split first, the velocity solver apart from the density carrier, which is one
  field's move once the density is a `Texture3D` on the grid (step 2).
- hd_trio's `density.fade_time` written per tick from the pose's motion: the 2D fluid's, not here.
