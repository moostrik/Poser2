# The 3D fluid

Every quantity and setting here is in a meaningful unit ([FLOW.md](FLOW.md#the-2d-and-3d-flow-systems)).

The 3D fluid's document, as [FLOW.md](FLOW.md) is the flow's: what the fluid is, how it takes what
the bridge gives it ([BRIDGE.md](BRIDGE.md)), its density and its look, and how it is measured; what
is next is its [Plan](#plan). "Code" is what the files show;
**(deduction)** marks what is inferred from them, **(site fact)** what the operator saw.

The 2D fluid (`fluid_2d/`, `Fluid2DLayer`, hd_trio) is apart: nothing here changes it. The 3D
fluid is `Fluid3D` (`fluid_3d/fluid_3d.py`), its layer `Fluid3DLayer`
(`modules/render/layers/flow/fluid_3d_layer.py`), run by deep_flow alone.

## The goal

A visualization inspired by ink in water: high detail at 4K, sharp, translucent ink patterns; smoke
patterns a second goal **(site fact)**. Two ways draw it, the particles and the upscaled density; the
upscaled density is the one worked on now, the particles and probably a combination of both later. Its
way there: layers of water with sharp, well-defined motion of their own, that influence each other and
stay distinct. Ink's thin sheets folding into tendrils are shear, water at one depth moving otherwise than
the water beside it, their boundary rolling up **(deduction)**. The light is a means to it, not a goal of
its own, and so far not a good one **(site fact)**.

What decides it in the upscale now (the code and deep_flow's preset):

| What                     | Now                                                                                   | Against the goal                                                                          |
|--------------------------|---------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------|
| the density's own layers | the 4K amount 24 layers deep (`upscaled.depth`), the velocity's grid 64               | a sheet thinner than a 24th of the box blurs into its neighbors: the layers it can hold |
| its edges as it moves    | `density.advection`: semi-Lagrangian 1.37 ms on the 4K amount at 24 layers, MacCormack 4.07 | the plain step blurs a little each step, MacCormack keeps edges sharp                      |
| the layers kept apart    | the body at its own depth ([The body's depth](#the-bodys-depth)), the confinement at a small radius, viscosity | the body's shape in depth lets the water go around it; the confinement and viscosity cross one layer's motion into the others'; viscosity 0 the sharpest |

## Plan

What is next for the fluid, in order; what is open elsewhere in the flow is
[FLOW.md's Open](FLOW.md#open).

Weighed by [the goal](#the-goal): sharp, translucent ink patterns, layers distinct.

### 1. Swirls at more than one size

The confinement keeps swirls alive at one size, `vorticity_radius`: fine wisps or big eddies, not both.
More than one radius, each with its own strength (multi-scale vorticity confinement), keeps fine and large
at once. Starting with two: the curl at both radii in the same pass (six more velocity reads a cell), the
force from both curls in one pass (a curl volume more), their sum added to the faces once. About a third
to a half more of the confinement's 0.57 ms at the preset, 0.2 to 0.3 ms **(deduction)**, measured as it
lands. The upscaled density's turbulence already adds swirls below the grid to the dye, so what a second
radius adds to the look is judged against it, and by whether the layers stay apart: the confinement
also carries one layer's motion into the next.

### 2. The smaller side as the unit

Every length in the flow is in widths: the motion's velocity (widths/s of the camera's image), the bridge's
readings, the fluid's inputs (`spread`), its look (`opacity`, `fog`), the box's depth, the monitor's scale.
In portrait the width is the smaller side; in landscape it is the longer, and the same settings would
then mean less against what is seen. One unit across the stages, the smaller side, keeps a preset's look
and feel when the box turns: in portrait nothing changes. It reaches past the fluid (the motion, the
bridge, the monitor), so it is one rule for all of them, made where each length is made.

### After the plan

- The masking in the bridge: the mask (`inputs.mask`, R16F 0 to 1, the depth's or the segmentation's) gates the velocity in the motion (`use_mask`) and the density's color in the bridge; in the bridge alone it is one gate in one stage.

- The body force's acceleration: tested, tweaked and made useful.
- The temperature: whether a field of its own is needed. Buoyancy needs only a scalar the water is lighter
  or heavier by; the density alone can be it (ink sinks), and the temperature adds currents apart from the
  dye (warm clear water rising). Off in the preset (`buoyancy` 0).
- The particles: worked on, and how they combine with the upscaled density.

## The fluid

A step (`Fluid3D.update`), in order, at the code's defaults:

| Pass        | What                                                                                                                                                                                                                                                                    | On                  | Depends on the input |
|-------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------|----------------------|
| advect_vel  | self-advection on the faces by `velocity.advection`, each face traced on its own ([The staggered velocity](#the-staggered-velocity)), `velocity.advection_scale` 1, fade 60 s; on its result the guard ([The guard](#the-guard)) and the slip | velocity's faces    | the trace distance   |
| viscosity   | Jacobi diffusion on the faces, off (`viscosity` 0)                                                                                                                                                                                                                      | velocity's faces    | no                   |
| vorticity   | the confinements (`confinement_1` to `confinement_3`, each a `strength` and a `radius`): every running one's curl at the cells from the faces in one pass, their forces summed at the cells in one, each face given its two cells' mean | faces, curl         | no                   |
| advect_temp | off: `buoyancy` 0 clears the temperature once, and nothing is injected into it                                                                                                                                                                                          | temperature         | the trace distance   |
| buoyancy    | gravity × (sink × density − buoyancy × ΔT), kept as an acceleration at the cells, each face given its two cells' mean × dt; gravity down the screen (`gravity_y` 1)                                                                                                     | acceleration        | no                   |
| incompress  | the divergence per cell from its six faces, the pressure by `pressure.solver` ([The pressure](#the-pressure)), the gradient per face from its two cells, on the walls ([The walls](#the-walls))                                                                         | pressure, faces     | no                   |
| centers     | the faces averaged to the cells' centers, which the particles, the upscaled density and the monitors read                                                                                                                                                               | velocity            | no                   |
| advect_den  | semi-Lagrangian advection, one trilinear read per texel, the evaporation taking the alpha down, the color with it                                                                                                                                                       | density 256×448×128 | the trace distance   |
| composite   | front-to-back march, one step per layer                                                                                                                                                                                                                                 | output 2160×3840    | no                   |

Every field is a `GL_TEXTURE_3D` on one grid of cubic cells at the stage's size, 256 × 448 by
`depth` 128: the velocity on its faces RGBA16F a face longer on every axis and at the centers RGBA16F, the
temperature R16F and the pressure R32F, the density
RGBA16F with its color baked in, each a swap pair, the obstacle R8. The simulation's cell is 1/256
width in x, y and z, so the slab is 128 / 256 = 0.5 widths deep (`_z_extent`). The grid is the one
that holds 60 fps in deep_flow with the pose on the same GPU ([TIMING.md](TIMING.md#the-grid)). The density is carried
on the same grid always (the simulation density), and where enabled by the upscaled density, its amount at
`upscaled.width` × `upscaled.height` × `upscaled.depth` with its color the simulation density's
([The density's detail](#the-densitys-detail)), and by the particles, drawn over the volume. Every pass
that reads the velocity or the obstacle against the density samples by uv, so the shaders are the same
(`advect_3d`, `inject_3d`, `buoyancy_3d`). The composite reads the volume at the output's 2160 × 3840. The monitors
read every field as it is. The GPU is a 24 GB RTX 4090, so nothing pages. Every pass's time is in
[TIMING.md](TIMING.md).

The layer injects the bridge's six outputs before the step (`Fluid3DLayer._inject`), each at the
injection's `weight`, into the gaussian slab at `input.layer` (0.5) `input.spread` thick (0.2 widths,
its width at half its height), each cell the body's share between its faces, each velocity's z as the
volume's w = −z, in widths/s as x and y (the motion's `z_gain` its share); what the fluid does with each is
[What the fluid takes](#what-the-fluid-takes).

The depth scale is the box's depth over the smaller of its width and height (`depth` / 448 at 448 × 800,
0.07 at 32 layers, 0.5 at 256 × 448 × 128): the box's depth stands for as much real depth as its width does, so
a body moving toward the camera crosses the box's depth in the time the same speed crosses its width. At one
scale with x and y, a near hand's real 2–3 m/s pushed water the box could not hold, and the projection turned
it sideways: on standmoves at 448 × 800 × 32 the water reached 5.2 widths/s, scaled 1.7, no step above 2.

`time_scale` is the fluid's time as a share of real time: every pass steps by dt × it, the sources and the drag
included, so at 0.5 the whole fluid runs at half speed and keeps its shapes. A viscosity costs 0.7 ms a step.

The velocity is in image widths per second, as the bridge's; the advection moves a texel by
velocity × dt × `density.advection_scale` (1) in uv, so 1 width/s crosses the width in 1 s.

The volume's settings: `depth_ratio`, the box's depth over its width, its z cells stretched to it (`z_stretch`
how many times longer a z cell is than an xy cell); `input.layer` and `input.spread`, where the body sits in
the volume and how thick it is;
`emission`, `opacity`, `tint` and `fog`, the composite's march; `wall` and `slip_time`, every
side of the box ([The walls](#the-walls)). Obstacles stay supported (`set_obstacle`, `add_obstacle`,
`clear_obstacles`, no caller today): one R8 volume on the simulation's grid, which every field's pass
samples.

## The inputs

The rule, by what each input is. The trail is a velocity in the fluid's own unit; added × dt it
would act as an acceleration and its gain as a rate. In Navier–Stokes the body force is an
acceleration per unit mass; a body in the fluid is a boundary that moves.

| Input is              | The fluid                                                                | Here                                                     |
|-----------------------|--------------------------------------------------------------------------|----------------------------------------------------------|
| a body's velocity     | is dragged toward it where the body is, over a time, never past it       | the bridge's `boundary.velocity`                         |
| a velocity change, Δv | adds it once in total: as an acceleration, the pulse over its area, × dt | the bandpass's readings as accelerations                 |
| an acceleration       | adds it × dt                                                             | buoyancy, the vorticity force, `body_force.acceleration` |
| a source              | adds it × dt, an amount per second                                       | the densities and the temperatures                       |

### What the fluid takes

What the fluid does with each of the bridge's outputs; the readings they are mixed from, and what
each output says of the body, are in [BRIDGE.md](BRIDGE.md#the-mix).

The outputs, in two subpanels, one per system, each a mix of the six readings; the fluid's settings
say how the water follows each; the densities and the temperatures go in by `inject_3d`, `FILL` for
the boundary, `RATE` for the body force:

| Subpanel     | Output         | Unit                              | The fluid                                                                                                                      | The fluid's setting                  | 1 means                               |
|--------------|----------------|-----------------------------------|--------------------------------------------------------------------------------------------------------------------------------|--------------------------------------|---------------------------------------|
| `boundary`   | `velocity`     | widths/s                          | drags the water toward it where it is non-zero, over `velocity_drag`, never past it (`add_to_faces_3d`, [The drag](#the-drag)) | `input.boundary.velocity_gain`       | the water as fast as the body         |
| `boundary`   | `density`      | the dye: color × presence, 0 to 1 | fills the water toward it × `density_gain` (at most 1) in the slab, never past it, over `density_drag`                         | `input.boundary.density_gain`        | fully dyed at 1 width/s               |
| `boundary`   | `temperature`  | warmth, -1 to 1                   | drags the water toward it × `temperature_gain` (at most ±1) in the slab where the body is, over `temperature_drag`             | `input.boundary.temperature_gain`    | fully warm or cold at 1 width/s       |
| `body_force` | `acceleration` | widths/s²                         | adds it × dt (`add_to_faces_3d`)                                                                                               | `input.body_force.acceleration_gain` | a Δv added once                       |
| `body_force` | `density`      | a push's dye, presence 0 to 1     | adds it at a rate × dt into the slab, capped at fully dyed: the pipette                                                        | `input.body_force.density_rate`      | fully dyed per second at a full push  |
| `body_force` | `temperature`  | a push's warmth, -1 to 1          | adds it at a rate × dt into the slab, capped at ±1                                                                             | `input.body_force.temperature_rate`  | full warmth per second at a full push |

The two subpanels are the two systems every solver distinguishes: a moving boundary, whose values
the water takes, and the body force, which adds to it. The boundary's density and temperature are
dragged toward as its velocity is, each over its own time and never past the body's value; the body
force's are added at a rate, capped at fully dyed and ±1.

### The body's depth

The camera sees the body's front surface, a plane of x, y and depth, not a volume, so each pixel goes in
as a thin skin centered on its own z position, `spread` its thickness: an arm reaching toward the camera
pushes the water near the front and the torso further back. The box's depth is the player's depth range,
the depth source's near to far, 0 front to 1 back, with no setting of its own. Each injection reads its
output's z position ([BRIDGE.md](BRIDGE.md#the-z-positions)), red over green, at the pixel it reads
(`slab.glsl`'s `slab_center`): the drag and the body force at their readings' depths, the densities and the
temperatures, the upscaled amount and the particles at the body's now. A pixel without one (no depth, out
of the range) goes in at `layer`, and so does every pixel without depth. The densities and the
temperature go in a column a thread (`inject_3d`, `inject_amount_3d`): the source and its z position read
once, then the cells the slab reaches, out to 4 sigmas. The particles' mass is the slab's share inside
the box per pixel, from erf at its two ends (`slab_column`).

z stays in widths/s, as the motion measures it, so the water's z and the skin's place move at different
scales: a hand's skin moves by its share of the range, the water it pushes by its widths at that distance;
`z_gain` scales the push. If that shows, the fluid can convert each pixel's z by its own depth into shares
of the box (× `view_span` × depth / (far − near)), exact per pixel.

### The drag

A body moving through water drags the water toward its own speed: the force is the difference,
`a = (v_body × gain − v) / boundary_drag`, an acceleration, added × dt. Where the fluid already moves
with the body the force is 0; where the body stops and the trail dies, nothing pulls, and the water
coasts on its own inertia. In the shader (`add_to_faces_3d.comp`, per face):
`v ← v + (v_body × gain − v) × pull`, with `pull = share × g × weight × slab`. `share` is
`1 − exp(−dt / velocity_drag)`, `velocity_drag` in ms, and 0 sets the velocity. `g` says where the body is:
the trail's speed above a floor, `min(|v_body| / DRAG_FLOOR, 1)`, `DRAG_FLOOR` a constant 0.02
widths/s, so a dead trail drags nothing toward 0; `weight` is the injection's, `slab` the gaussian.
Every factor is at most 1, so the injection never passes its target, whatever `gain`, `weight` or
the frame rate: it leaves the fluid at most as fast as the body × `gain`; what the simulation then
does with the velocity, the projection and the confinement, is its own. w = −z. The arithmetic is
in numpy beside the fluid (`drag_share`, `drag_gate`, `drag_velocity_step`, `test_drag.py`).

`density.advection_scale` and `temperature.advection_scale` are 1: the density moves with the water.
`velocity.advection_scale` is 1: the velocity carries itself the full step, the physics; viscosity and
the confinement are the calming knobs. The lowpass's `smooth` makes the trail coast after a stop; with
the drag that is the body still dragging after it stopped, while the coast is the fluid's own inertia
([Measuring](#measuring) shows it at `smooth` 300), so the drag wants a short `smooth`: its default is
160 ms.

### The walls

The domain is a box with solid walls, as the textbook's (Bridson): beyond a closed wall the water
cannot pass, v·n = 0, it slides along freely (free-slip), and the pressure has no gradient across
it, ∂p/∂n = 0. One rule says what lies beyond the box, `walls.glsl`, included by every pass that
reads there: the divergence, the gradient, the Jacobi and SOR solves, the multigrid's residual and
prolongation, the curl and the confinement (`uClosed`, per axis, from `wall`). A ghost cell beyond
the last mirrors it:

|                                      | closed                                           | open                                               |
|--------------------------------------|--------------------------------------------------|----------------------------------------------------|
| the velocity                         | through the wall turned; along it (2k − 1) of it | the same: it leaves as it came                     |
| the pressure                         | the same: no push across                         | its negative: 0 on the wall's face, on every level |
| the curl's size, for the confinement | the same: no slope toward the wall               | the same                                           |

On the faces (`faces.glsl`) the wall's own face is stored: closed, it is 0 and never written; open, the
gradient sets it from the ghost pressure. Beyond a closed wall a read takes the face at the edge, and the
curl's ghost carries (2k − 1) of it. Beyond an open wall a read takes the water outside, at rest, 0,
faded in over the last texel as a border of 0 is: a trace from outside carries still water in. Read as
the face at the edge instead, the boundary fed its own inflow back in and the water grew (0.84 widths/s
within 6 s on the swing at ε 1.93), the divergence 0 throughout. An obstacle's cell is a closed wall inside the box: the faces on
its sides 0, its curl's ghost the closed wall's.

A closed wall slips (Navier slip): `slip_time`, the seconds the water along it takes to slow to 1%,
0 free-slip (the default). It models the layer along the wall a coarse grid cannot resolve, where
no-slip would need that layer resolved. k = 0.01^(dt / `slip_time`) is the share of the velocity
along the wall kept a step: the velocity's advection multiplies the tangential velocity of the cells
against a closed wall by it, and the curl's ghost carries (2k − 1) of it, so the curl reads the shear
the braking makes. Only the tangential velocity changes; the pressure's condition stays, so the
projection, the curl and the confinement still agree. A slipping wall makes a curl in proportion to
its braking, which the confinement amplifies.

Every pass reads the same ghosts, and none reads the texture's border instead:

- A curl or a confinement reading the border's 0 sees a wall the water slides along as a shear, a
  curl no wall has, and the confinement's push along the wall makes it bigger the next step.
- A divergence reading the border sees a face at the wall the gradient cannot empty, so water pushed
  into a wall stays and is squeezed along it.
- An open wall's pressure as a ghost of 0 puts that 0 half a cell further out on each multigrid
  level, a different place on each, and the coarse correction spreads to the edge as at a closed
  wall, so it diverges.

Measured with `scripts/maccormack_check.py` at deep_flow's preset (the multigrid, 256 × 448 × 128), a
disc swinging for 3 s, then still; the mean speed (widths/s) in the two cells next to the walls and
inside them, at 4 s and 12 s:

| Run                                     | wall band, the border | inside, the border | wall band, the ghosts | inside, the ghosts  |
|-----------------------------------------|-----------------------|--------------------|-----------------------|---------------------|
| closed, vorticity 1.93, semi-Lagrangian | 0.060 → 0.212         | 0.024 → 0.069      | 0.032 → 0.011         | 0.022 → 0.008       |
| closed, vorticity 1.93, MacCormack      | 0.049 → 0.227         | 0.025 → 0.061      | 0.031 → 0.009         | 0.025 → 0.007       |
| closed, vorticity 5, MacCormack         |                       |                    | 0.027 → 0.015         | 0.025 → 0.021       |
| open, both advections                   | non-finite in 1.5 s   |                    | 0.004 → 0.002 (6 s)   | 0.007 → 0.004 (6 s) |

Jacobi and SOR, closed and open, decay as well. With the ghosts, the slip, MacCormack, the wall band
and the inside at 12 s, no run non-finite:

| Run                 | free-slip     | `slip_time` 2 | `slip_time` 0.5 |
|---------------------|---------------|---------------|-----------------|
| vorticity 1.93 (SL) | 0.023 / 0.011 | 0.004 / 0.007 |                 |
| vorticity 1.93 (MC) | 0.018 / 0.010 | 0.011 / 0.010 |                 |
| vorticity 5 (MC)    | 0.015 / 0.021 |               | 0.008 / 0.021   |

Measured in a hidden window at 448 × 800, a body 50 px from the right wall at 1 width/s, held 5 s,
its band 300 px tall, at 25 fps:

| The body               | The water's highest speed, the border as wall | the ghosts |
|------------------------|-----------------------------------------------|------------|
| pushing into the wall  | 2.34 widths/s, in the wall's cells            | 0.93       |
| sliding along the wall | 3.96, in the corner                           | 1.07       |

The confinement is not the cause: off, the border gives 2.34 and 4.11. With the ghosts the water
stays at the body's speed or under, the confinement adding a few percent, MacCormack too (0.90 and
1.23).

### The pressure

`pressure.solver` picks how the Poisson equation is solved each step, each warm from the last step's
pressure, each on the walls as [The walls](#the-walls) states:

| `solver`    | How                                                                                                                                                                                                                                                                                                                      | Its count                |
|-------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------|
| `JACOBI`    | every cell from its neighbors' last values, five iterations a dispatch in shared memory (`jacobi_pressure_3d`)                                                                                                                                                                                                           | `iterations`, 40         |
| `SOR`       | red-black Gauss–Seidel over-relaxed by `relaxation`: each color from the other's newest values (`sor_pressure_3d`); 1.95 diverged, so its range stops at 1.85                                                                                                                                                            | `iterations`, its sweeps |
| `MULTIGRID` | V-cycles (`multigrid_pressure.py`): on each level two red-black Gauss–Seidel sweeps, the residual restricted to the level halved, its correction solved from 0 and added back trilinearly, two sweeps again; down to 8 × 14 × 4 at the default grid, 16 sweeps there in one dispatch; obstacles on the finest level only | `cycles`, 1              |

Measured through the harness on stillslowfast, the step's time with `glFinish` around it, the solve's
residual |div − L p| over |div| (the mean and the p99, at four quarters of the loop), and the projected
velocity's divergence, central differences, a cell's share:

| Solver              | Residual, mean | p99    | Divergence after | Step, ms |
|---------------------|----------------|--------|------------------|----------|
| Jacobi 40           | 0.372          | 2.92   | 0.0079           | 8.66     |
| Jacobi 20           | 0.454          | 3.76   | 0.0090           | 7.20     |
| SOR 10, 1.8         | 1.570          | 12.0   | 0.0074           | 8.94     |
| SOR 20, 1.8         | 0.898          | 6.87   | 0.0078           | 12.33    |
| multigrid, 1 cycle  | 0.024          | 0.42   | 0.0078           | 7.89     |
| multigrid, 2 cycles | 0.002          | 0.03   | 0.0077           | 10.68    |
| multigrid, 4 cycles | 0.0001         | 0.0004 | 0.0073           | 16.24    |

The residual is worked out where the restriction reads it (`ResidualRestrict3D`), never written at the fine
level, and the coarsest level's sweeps run in shared memory in one dispatch (`CoarsePressure3D`, up to 4096
cells): a level of a few thousand cells cannot fill the GPU, so its 32 half-sweep dispatches were their
latency. At the preset (288 × 512 × 64) a cycle alone 0.98 → 0.77 ms, the pressure in the app 1.61 →
1.44 ms; the fused residual is the same to the bit, the coarse solve the same to a float's last bit (the
compiler orders its multiply-adds its own way).

One V-cycle leaves a sixteenth of Jacobi's residual in less time; each further cycle divides it by
about thirteen. SOR's sweeps cost 0.34 ms each, two dispatches from global memory, and over-relaxed it
leaves the high frequencies, which the mean residual counts, worse than Jacobi does. These were measured
on the centered velocity, whose central differences' Laplacian is not the 7-point one the solvers solve;
on the faces it is ([The staggered velocity](#the-staggered-velocity)), and the divergence the multigrid
leaves is the half float's rounding. With the solve exact the water keeps more of its speed, 0.150
widths/s in the mean against Jacobi's 0.134.

### The guard

A guard on the advection of each field, at the step's first pass, so a bad value stops where it was made
and is never hidden:

| Guard     | What                                                                                                                                                                                                                                                                                                                                                                                                                                             | Setting                                                                                             |
|-----------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------|
| finite    | a value whose exponent bits are all ones, an infinity or a NaN, becomes 0: a cell that held one before the step, or that the step made one; each counted per field into the stage's `FiniteGuard` (`finite_guard.py`), read once a second from the other of two buffers once a fence says the GPU is done, so nothing waits, and logged as a warning naming the stage, the field and the count; on the velocity, the density and the temperature | `guard`, beside `reset`, on by default; off once the fluid is stable                                |

The bit test, not `isnan`: GLSL may assume no NaN exists and drop `isnan`. A NaN slips past every comparison
(`speed > threshold` is false), so a threshold could not catch it, and it spreads to every value computed from
it. An infinite velocity traces off the grid and reads the border's 0, so it is caught where it is held. Where all
eight texels MacCormack's limiter reads are non-finite it has no bounds, and the predictor stands for the guard
to take. With the drag the injection cannot exceed the body's own speed; the simulation's own growth, from the
vorticity force above all, is the walls' and the advection's to hold, with no top speed over it.

Measured in a hidden window at 64 × 128 × 32, the velocity filled with each value and stepped, semi-Lagrangian
and MacCormack alike:

| Filled with                | After one step                                | Warning within 2 s                                 |
|----------------------------|-----------------------------------------------|----------------------------------------------------|
| infinity                   | 0 everywhere, the output finite               | 262144 non-finite velocity values, every cell once |
| NaN                        | 0 everywhere, the output finite               | 262144, every cell once                            |
| 50 widths/s                | top speed 44.7 (the walls and the projection) | none                                               |

### Names

The rule is `.claude/rules/flow.md`: the textbook word where one exists, a reading by what it is, a
module by what it does, everything else by its quantity.

| Kind                      | Names                                                                                                                                           |
|---------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------|
| the mixes' knobs          | shares, 0 to 1, one per reading in each output, summed                                                                                          |
| the fluid's input methods | `drag_velocity`, `add_acceleration`, `add_boundary_density`, `add_body_force_density`, `add_boundary_temperature`, `add_body_force_temperature` |
| the fluid's fields        | `velocity` (its own settings: fade, evaporation, top speed, vorticity, viscosity, self-advection), `density`, `temperature`, `pressure`         |

The readings, the filters, the two systems, the input groups and the wall are the glossary's
("Flow" and "Apps"). "Drag" is what the fluid does with `boundary.velocity`, seen as `velocity_drag`, not the name of a
thing; "impulse", "body" alone, "force" alone, "model" and "delta" name nothing; "brush" and
"pipette" are the physics of the two densities, not names. A reading is what a module of the
bridge makes from the motion; not "signal", "band", "component", "event".

## Measuring

Tests go through the harness, `scripts/fluid_compare.py` beside `bridge_compare.py`: nothing saves
the bridge's outputs, so it chains the saved frames of a recording (`depth_velocity_capture`)
through the flow, a `Bridge3DLayer` and a `Fluid3DLayer` per side, A against B, with deep_flow's
preset and one setting changed by its panel path (`fluid.input.boundary.velocity_drag=0`,
`bridge.readings.lowpass.smooth=300`), and reads the fluid's velocity and density on the injection
layer's slice (`VolumeCut`). `--every N` steps the bridge and the fluid once per N frames with the
summed dt, a frame-rate check. Numpy only where the harness cannot reach: the drag's share and
floor as a formula (`test_drag.py`).

On stillslowfast (353 frames, 43 ms each) and standmoves (1356 frames, 44 ms), the water on the
moving player: its mean in the moves' fast halves over the body's (the trail's), the highest water
over the loop over the highest body, and the coast, s after a move's end until the water is under
20% of its fast mean; the body's coast beside it. The preset is `velocity_drag` 50 ms, `gain` 1,
`smooth` 1000 ms:

| Clip          | Side              | Water / body, fast | Peak / body | Coast s, body / water | GPU ms, mean / p95 |
|---------------|-------------------|--------------------|-------------|-----------------------|--------------------|
| stillslowfast | preset            | 0.55               | 0.74        | 0.85 / 0.94           | 9.7 / 12.8         |
| stillslowfast | `velocity_drag` 0 | 0.61               | 1.18        | 0.85 / 0.94           | 9.8 / 12.8         |
| stillslowfast | `smooth` 300      | 0.56               | 0.66        | 0.23 / 0.94           | 9.7 / 12.6         |
| stillslowfast | `gain` 2          | 0.86               | 1.41        | 0.85 / 0.94           | 9.7 / 12.8         |
| standmoves    | preset            | 0.73               | 0.79        | 1.15 / 3.34           | 9.2 / 13.1         |
| standmoves    | `velocity_drag` 0 | 0.78               | 0.93        | 1.15 / 2.68           | 9.3 / 13.2         |

The water stays under the body at `gain` 1 and under twice it at `gain` 2 in the mean; the peak
over the loop passes 1 at `velocity_drag` 0 (1.18), which is the simulation's own doing after the
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

| Field                                                                                                                                      | Default         | Means                                                                                                                            |
|--------------------------------------------------------------------------------------------------------------------------------------------|-----------------|----------------------------------------------------------------------------------------------------------------------------------|
| `input.boundary.velocity_gain`                                                                                                             | 1               | the water as fast as the body                                                                                                    |
| `input.boundary.velocity_drag`                                                                                                             | 50 ms           | how fast the water takes the body's speed                                                                                        |
| `input.body_force.acceleration_gain`                                                                                                       | 1               | a Δv added once                                                                                                                  |
| `input.boundary.density_drag`, `temperature_drag`; `density_gain`, `temperature_gain`; `input.body_force.density_rate`, `temperature_rate` | 200 ms; 1; 1 /s | the water takes the dye in a fifth of a second, fully dyed where the body moves at 1 width/s; a full push dyes fully in a second |
| `density.advection_scale`, `temperature.advection_scale`                                                                                   | 1               | the density moves with the water                                                                                                 |
| `velocity.advection_scale`                                                                                                                 | 1               | the velocity carries itself the full step                                                                                        |
| `width`, `height`, `depth`                                                                                                                 | 256, 448, 128   | half a width deep, within 60 fps                                                                                                 |
| `upscaled.enabled`, `particles.enabled`                                                                                                    | off, off        | the density on the simulation's grid                                                                                             |
| `input.layer`, `input.spread`                                                                                                              | 0.5, 0.2        | the body mid-volume, 0.2 widths thick at half its height                                                                         |
| `wall` (every field, the volume)                                                                                                           | closed          | the operator's settings                                                                                                          |

The fields' physics take the textbook's water, with no artificial decay or thickness, as defaults: the
velocity's `fade_time` 60 s (it keeps its momentum), `viscosity` 0 (water at this scale),
`pressure.iterations` 40 (GPU Gems' 40 to 80 for a Jacobi solve), the density's `fade_time` and
`evaporation_time` 0 (off: it stays until the water carries it away), `temperature.sink` 0 and `ambient`
0 (the density as heavy as the water, no buoyancy), gravity down the screen, every wall closed; the top
speed off, a failsafe; the bridge's mix the boundary's trail alone and its lowpass's `smooth` its default
160 ms, since the coast is the fluid's; the three confinements off (`strength` 0), each over its neighbors
(`radius` one cell) where it runs, Fedkiw's ε 0.3 the textbook's start; in z, `input.layer` 0.5 and `input.spread` 0.2 widths, a body's thickness at
the camera's crop (from its `view_span` of 0.76 m, **(deduction)**) where it is at least half there. Kept as tuned: the composite's look (`emission`, `opacity`,
`tint`, `fog`).

Measured with these on standmoves through the harness: the water 0.89 of the body's speed in the
fast halves, its peak 0.76 of the body's, the step 11.5 ms with the harness (mean; p95 12.8).

### One unit

Every length in the fluid is in image widths, the motion's and the bridge's unit, and every time in
seconds: a cell's sides 1 / the grid's width across and `depth_ratio` / `depth` in z (the z cells stretched),
one `Cell` the fluid hands every face pass through `bind_faces`. The solvers' differences are over each
axis's side (the divergence, the gradient, the pressure's solves, the viscosity, the curl, the velocity's own
advection); the multigrid halves an axis only while its cells are shorter than the longest by more than 1.5,
all of them once near cubic (semi-coarsening, `coarsening`), so a level's cells grow toward cubic before the
stretched axis is halved; the confinement's force is ε × h × (N × ω) (Fedkiw 2001), h the xy cell, with the
curl over each confinement's `radius` widths, at least a cell on each axis, and the confinements' forces
summed;
the buoyancy is an acceleration, widths/s² per unit of density or temperature; the viscosity a
kinematic viscosity in widths²/s; `input.spread` a thickness in widths. The advection moves a
texel by v × dt in uv as (1, width / height, 1 / `depth_ratio`) × v, x, y and z each a width per width.
The pressure is 32-bit: in these units it is small, past the half float's precision.

The output's allocation is not in it. A cell taken as the grid's width over the output's (448 /
2160) makes the confinement, the buoyancy and the viscosity follow the output's size, and moves y
(height / width)², 3.2 times, too far. Measured in a hidden window:

| Check                                                                                                | Cell by the output | One unit |
|------------------------------------------------------------------------------------------------------|--------------------|----------|
| 0.02 widths/s along y for 1 s: the density moved                                                     | 0.064 widths       | 0.020    |
| a swirl with confinement, the same at a 1080 and a 2160 wide output: the largest velocity difference | 0.50 widths/s      | 0.000    |

## Tuning

What remains is the preset, saved by the operator on the clips: `depth`, the gains, `velocity_drag`, the
lowpass's `smooth`, the mixes; judged on deep_flow with the standmoves and stillslowfast
recordings. Judge on: slow against fast reads; a stop coasts; a flick leaves one drop at its end
with the body force's density; the burst's fps; the wall.

## The advection

Each field's `advection` (`velocity.advection`, `density.advection` for the simulation's and the upscaled
density, `temperature.advection`) carries it: `SEMI_LAGRANGIAN`, the default,
or `MACCORMACK` (Selle 2008, `maccormack_3d.comp`). The semi-Lagrangian step writes a predictor into
a scratch volume per field (allocated at its first MacCormack step); traced forward again, it gives
back the source up to the step's error, half of which is added back, bounded by the source's eight
texels around the backward trace: where the correction leaves them, the predictor stands (Selle's
revert), as it does where that trace reads past a wall: the wall's value is in no texel, and a
correction kept there grows the velocity at the walls step after step, to eight times its speed in a
second at the preset's grid (measured without the guard).

Measured in a hidden window at 448 × 800 × 64, a swirl dragged in and left, then a square of
density carried for 2 s, no confinement, no viscosity; and on stillslowfast through the harness at
the preset:

| Measure                                                        | Semi-Lagrangian | MacCormack |
|----------------------------------------------------------------|-----------------|------------|
| the swirl's mean speed after 2 s, of its start                 | 0.52            | 0.62       |
| the density's edge, its 99.5% gradient over its max, per texel | 0.071           | 0.140      |
| the step, GPU ms, hidden window                                | 5.7             | 6.7        |
| stillslowfast: water / body, fast                              | 0.55            | 0.60       |
| stillslowfast: peak / body                                     | 0.74            | 0.82       |
| stillslowfast: the step with the harness, ms                   | 9.5             | 11.0       |

The revert, not a clamp to the bounds: a clamp keeps the corrected value at the bound, energy the
step lost, and with the vorticity confinement adding more the water grows after the body stops.
Measured with `scripts/maccormack_check.py` at deep_flow's preset (256 × 448 × 128), a disc swinging
through the water for 3 s, the water's mean speed (widths/s) after the body stopped, 4 s and 20 s, and a
square of density carried 2 s by a steady swirl at 128 × 224 × 32, no confinement:

|                                                  | Semi-Lagrangian | MacCormack, clamp             | MacCormack, revert |
|--------------------------------------------------|-----------------|-------------------------------|--------------------|
| vorticity 0.3: mean speed at 4 s → 20 s          | 0.0233 → 0.0082 | 0.0286 → 0.0285, not decaying | 0.0256 → 0.0080    |
| vorticity 1.93: mean speed at 4 s → 20 s         | 0.0271 → 0.0968 | 0.0303 → 0.2458               | 0.0268 → 0.0884    |
| the density's edge (99.5% gradient over the max) | 0.099           | 0.173                         | 0.172              |
| the swirl's mean speed after 2 s, of its start   | 0.703           | 0.742                         | 0.723              |

The revert keeps MacCormack's detail and decays as the plain step does. Above about 1 the confinement
grows the water in both advections, after the body stopped: the confinement adds energy every step,
which the step's own loss balances only at a small epsilon.

## The density

The solver owns the velocity, the pressure and the temperature; the density is its carriers'
(`DensityCarrier`, `density_carrier.py`), each taking the bridge's dye in the same units, fully dyed
water at 1: `SimulationDensity` on the simulation's grid, always on; `UpscaledDensity`, its amount
finer than the grid, and `ParticleDensity`, each switched by `enabled` and allocated only then. The
field advection, semi-Lagrangian or MacCormack, is one part they share (`FieldAdvection`).

### The particles

The particles draw into their own image, laid over the volume
(`CompositeOver`, Porter–Duff over); the buoyancy weighs the simulation density. Where a sprite would cover
the frame its birth radius stops at twice `radius`, and the reference mass is a texel of fully dyed water, so
how much dye there is sets how many drops there are, not how big a drop is.

Measured in a hidden window, the output at 2160 × 3840, the grid 448 × 800 × 64, the step with the
carrier's draw; with the ring full, and on stillslowfast through the harness at the preset (`rate` 2
for the blobby looks):

| `draw`, `overlap`   | Particles | Ring full, GPU ms | stillslowfast, ms mean / p95 |
|---------------------|-----------|-------------------|------------------------------|
| the grid            |           | 6.6               | 7.7 / 8.4                    |
| `POINTS`, occluded  | 32 M      | 21.3              | 16.3 / 24.7                  |
| `POINTS`, additive  | 32 M      | 12.7              |                              |
| `STREAKS`, occluded | 32 M      | 39.5              | 13.8 / 24.5                  |
| `BLOBS`, occluded   | 1 M       | 26.9              | 17.9 / 43.0                  |
| `SURFACE`           | 1 M       | 31.9              | 20.1 / 54.1                  |

The cost is the draw's fill at 4K: the count for points and streaks, the radius for blobs and the
surface. `count`, `radius`, `slices` and the additive overlap are the knobs; the count each look
fits in 60 fps, measured in the app, is in [TIMING.md](TIMING.md#the-budget): 2 M points, the
default, 1 M streaks, 100 k blobs or spheres.

The grid density stays beside the particles, which are switched on over it;
drawn as points or as streaks, one setting; the count a setting, sized by memory (3.2 GB, the 4K RGBA
density's, holds 100 M particles at 32 bytes), its default by the 60 fps budget; the oldest fade out
before the ring overwrites them; overlap first, the dye a bit opaque, additive as the other option.

Everything stays on the GPU: the CPU dispatches and draws a fixed count, never reads a particle
back:

| Part                              | How                                                                                                                                                                                                                                                                                                                                          | Why                                                                                                                              |
|-----------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------|
| storage                           | three R16F volumes, one per component, or one RGBA16F volume, each a face longer along an axis: the faster at step 3a                                                                                                                                                                                                                        |                                                                                                                                  |
| emission                          | a compute pass over the bridge's density at the bridge's size: each texel's count, amount × rate × dt, rounded by a hash so a small amount still spawns; an atomic head (`atomicAdd`) gives the slots, modulo the count                                                                                                                      | a ring buffer: no free list, no compaction, no count read back                                                                   |
| the ring's fade                   | a particle's place in the ring, (head − slot) mod N over N, is how near it is to being overwritten; its alpha falls to 0 over the ring's last share (`ring_fade`, 0.1), so the oldest fade out at whatever rate the body emits; the age's own fade by `density.fade_time` on top                                                             | no particle vanishes in a frame, whatever the emission                                                                           |
| advection                         | a compute pass, 256 a group, RK2 (midpoint) through the velocity's `sampler3D`, the same uv metric as `advect_3d` (`uRdx`, `advection_scale`); the walls clamp                                                                                                                                                                               | two trilinear fetches a particle, the hardware's; the grid sets the motion                                                       |
| drawing                           | no copy: a vertex shader pulls the SSBO by `gl_VertexID`; points `glDrawArrays(GL_POINTS, 0, N)`, streaks `GL_LINES` with 2 N vertices, the previous and the current position; a dead or unborn particle leaves the clip space                                                                                                               | the rasterizer splats millions faster than atomics in compute, and draws lines, which compute would not                          |
| overlap (`OCCLUDED`, the default) | the particles drawn additively into depth slices, a `GL_TEXTURE_2D_ARRAY` at the output's size, a layered draw (`gl_Layer` from the particle's z): 16 slices at 2160 × 3840 RGBA16F, 1 GB, the color and the optical depth; then a front-to-back march through the slices, Beer–Lambert, `absorption`, `tint`, `fog` as the grid's composite | near dye covers far dye between slices, exact for emission and absorption inside one; a bucket sort by depth, no particle sorted |
| overlap (`ADDITIVE`)              | the same draw into one RGBA16F target, each particle's fog by its z; one pass: exp(−τ) and the emission                                                                                                                                                                                                                             | order-independent, one target; the depth only as fog                                                                             |
| overlap (`WEIGHTED`)              | weighted blended order-independent transparency (McGuire and Bavoil 2013): one pass into two targets at the output's size, the emission and the opacity weighted by depth (RGBA32F) and the revealage per channel (×(1 − α)); one pass then divides                                                                             | one pass, no slices to clear or march; the order approximate, by a weight                                                        |

Two more ways to draw the same particles, as options: blobby, where each particle carries
a smooth kernel and the kernels' sum melts near particles into one shape (Blinn's blobby objects,
metaballs). Far fewer particles, each covering many pixels: the cost moves from the count to the
kernel.

| `draw`    | What                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      | Count, 60 fps |
|-----------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------|
| `POINTS`  | one pixel a particle, into the slices or additive, above                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  | 2 M           |
| `STREAKS` | a line from the previous to the current position                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          | 1 M           |
| `BLOBS`   | the soft melt: each particle a gaussian sprite of `radius` pixels, its sum the density, marched through the slices as `POINTS` is; dye clouds whose drops merge                                                                                                                                                                                                                                                                                                                                                                           | 100 k         |
| `SURFACE` | the liquid: screen-space fluid rendering (van der Laan 2009); each particle a sphere sprite of `radius`, its nearest depth into a depth target and its thickness summed into another; the depth smoothed by a filter that keeps its edges (a bilateral or narrow-range filter); the normals from the smoothed depth; then the dye's color through its thickness (Beer–Lambert, `absorption`, `tint`), a Fresnel edge, a specular and the background refracted by the normal; drops and threads with a sharp edge that merge and pinch off | 100 k         |

Each particle carries a mass, the density it took at its birth; its kernel sums to its mass, so
the dye is kept whatever the size. A blob's or a sphere's radius follows the mass,
`radius × (mass / reference)^(1/3)`, the cube root since the mass fills a volume: a strong drop a
big blob, a trace a small one. As it ages a blob widens and its peak falls by the same mass, as a
drop of dye spreads in water (`spread`, the diffusion: σ² grows by 2 × `spread` × dt). The
widening is refined: a small default, and a blob grows to at most twice its birth radius, so it
softens over its life and never bloats. A point's
and a streak's alpha is its mass.

| Part       | How                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
|------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| emission   | each tick, at the bridge's size, the boundary's density and the body force's, each by its setting in `input`: a texel spawns `rate × amount × dt` particles, rounded by a hash so faint dye still spawns now and then; each takes an even share of the texel's amount as its mass, the density's color, a random place in the texel and a z drawn from the injection slab's gaussian; no velocity of its own, the grid carries it and the drag gave the grid the body's |
| one knob   | a high `rate` many light particles, the points' and the streaks' look; a low `rate` few heavy ones, the blobs'; the mass and the size follow                                                                                                                                                                                                                                                                                                                            |
| duration   | three things end a particle, the first that comes: `density.fade_time`, the mass's exponential decay, as the grid density's; `evaporation_time`, a fixed amount taken off, which takes faint ones to 0; the ring, once more are alive than `count`, the oldest fading over `ring_fade` before they are overwritten                                                                                                                                                      |
| the budget | in a steady state about `rate` × the emission × the lifetime are alive; while that stays under `count` the fade decides how long dye lasts and the ring is only the cap; the panel shows the seconds the ring takes to wrap, one counter read asynchronously, no particle                                                                                                                                                                                               |

`BLOBS` is the slice target with a bigger, softer point; `SURFACE` is its own pass, a depth and a
thickness target at the output's size, a filter of a few passes and one shading pass. Both depend
on `radius` against the particles' spacing: too small and the blobs fall apart into dots, too big
and the shape goes soft **(deduction)**.

The settings, a `particles` group: `count` (`Field.INIT`, 2 M), `rate` (particles per unit of density per
second), `ring_fade` (the ring's last share the oldest fade over), `draw` (`ParticleDraw.POINTS`,
`STREAKS`, `BLOBS` or `SURFACE`), `radius` (a blob's or a sphere's pixels at the reference mass),
`spread` (a blob's widening with age), `smooth` (the surface's filter passes), `size` (a point's pixels),
`streak` (a streak's length as a share of its step), `overlap` (`Overlap.OCCLUDED` or `ADDITIVE`),
`slices` (`Field.INIT`, 16). The cost grows with the count by the advection's reads and the draw's fill,
not by the memory: the advection 0.2 ms a million, the draw faster than the count once many share a pixel
([TIMING.md](TIMING.md#the-particles)). Judge on: a drop reads as a drop next to the grid's; a thread
stays one particle wide at 2160 × 3840; near dye covers far dye; the step's GPU time.

### The density's detail

The goal: at the 4K output the density is as fine as the pixels, thin threads and smooth gradients drawn
out by the motion, never grainy; near density covers far; all of it at 60 fps with the pose running.

A 2160 × 3840 × 16 RGBA density carried by a coarse velocity, the ofxFlowTools recipe in 3D, costs 2 GB
of reads and writes a step. The upscaled density is that look with the color split off, as video splits its
color (4:2:0): the eye resolves the brightness much finer than the color, and the color never has more
detail than the bridge's 448 × 800 it comes from.

| Part        | How                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
|-------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| two volumes | the upscaled density (`UpscaledDensity`): the color, the simulation density's RGBA volume (color × amount, the amount); the amount alone (R16F) at `upscaled.width` × `upscaled.height` × `upscaled.depth` (2160 × 3840 × 16, the output's size), over the same box; at 16 layers 265 MB a buffer                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| injection   | the bridge's density into both: the RGBA into the color, its amount into the detail's gaussian slab                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| a step      | both carried by the velocity, by `density.advection`; where the detail has fewer layers than the grid, one pass first averages the velocity over each detail layer's thickness and the detail moves with that; the fade and the evaporation as on the grid                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| composite   | the detail's layers marched at the output: the brightness and the opacity from the detail's amount, the color the color volume's rgb over its amount at the same place                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| off         | `upscaled.enabled` off: the detail is not allocated, the simulation density drawn                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| settings    | `upscaled.enabled`, its size, and `upscaled.sample` (`MIDDLE` or `AVERAGE`, the velocity a layer spanning several of the grid's moves with); smaller than the output, the composite enlarges it by uv; on the monitor the amount is `upscaled_density`, the color `density`; timed in [TIMING.md](TIMING.md#the-density-at-the-outputs-4k)                                                                                                                                                                                                                                                                                                                                                                                                       |
| turbulence  | the 4K amount moved semi-Lagrangian by the grid's velocity plus swirls below the grid (wavelet turbulence, Kim et al. 2008): a periodic 64³ tile of divergence-free noise (`swirl_noise`, the curl of random vectors band-passed to 8–16 texels, the tile 8 grid cells, so the swirls are 1 to 2 cells), read where the water started (two displacements of the water on the grid, reset a second apart half a cycle from each other and cross-faded), scaled by the grid's curl × its cell and `upscaled.turbulence`; the curl computed for it when the confinement is off. On stillslowfast with the calm preset, 1 to 3 change the picture by under 0.2 of 255; 10 to 30 give visible fine frizz along the density's edges, where the curl is |
| storage     | 3D textures, as every field: the advection reads a traced point in one trilinear fetch, z included, and the driver tiles the texels in x, y and z; a texture array filters within a layer only, two fetches and a mix for z, and its layers are tiled apart, which suits drawing into layers (the particles' slices), not tracing through them                                                                                                                                                                                                                                                                                                                                                                                                   |

Measured: the step at 4K × 16, 32 and 64 layers, semi-Lagrangian and MacCormack, against the budget,
in TIMING.md; judged by eye against the RGBA 4K look on standmoves. A second preset, `flat.json`, for this
look with a velocity grid still to pick (448 × 800 × 64 or 128 × 224 × 16).

What a 4K volume cannot do, and so where the detail comes from:

- Many layers at 4K do not fit, and a cubic volume at 4K cannot be had: 64 cells
  deep at 448 wide is 64 × 4.8 ≈ 307 layers at 2160 wide, about 20 GB. So the detail at 4K does
  not come from a cubic volume, but from the upscaled density's few layers, the turbulence below the
  grid and the particles.
- A tricubic read in the composite is 8 hardware reads a step instead of 1, 64 steps over
  2160 × 3840 pixels at 64 layers, about 4 G reads a frame, 5 to 10 ms **(deduction)**: not worth
  it for a smoother upscale.
- A march at the grid's 256 × 448 (0.11 M pixels instead of 8.3 M, about 70 times cheaper
  **(deduction)**) with an edge-aware upscale of the flat result is parked
  ([Parked](#parked)).

The grid's depth: 256 × 448 × 128, half a width deep, so the body's z motion has room before a wall.
A coarser grid across buys the depth: the detail it gives up in x and y is what the upscaled density
and the particles bring back, the trade the density roadmap made at 128 × 224 × 96. The deeper grids
measured, and why this one, are in [TIMING.md](TIMING.md#the-grid).

## The fade and the evaporation

Every field (velocity, density, temperature) has a `fade_time` and an `evaporation_time`, both seconds,
0 to 300 in steps of 1, 0 off. The fade is the seconds to 1%, a share at a time
(0.01^(dt / `fade_time`)), so it never reaches zero. The evaporation is the seconds a full value
takes to reach 0, a fixed amount at a time (dt / `evaporation_time` of it), so faint values go first:
fully dyed, ±1 warm, 1 width/s; on the velocity a constant braking that stops slow water. It takes a
field's size down, the alpha of a density, the length of a velocity, |r| of the temperature, and
scales every channel with it, so a color keeps its hue, a velocity its direction and a warmth its
sign (`evaporate` in `advect_3d.comp` and `maccormack_3d.comp`). `temperature.buoyancy` is per unit
of warmth.

## The look

How the density is drawn, using the depth the coarse fluid already simulates (its 128 layers). The look
is set in widths: `composite.opacity` and `fog` are the distances that hide half and dim half, and the
monitor's `scale` 1 shows a full value for the density and the temperature as for a velocity of 1 width/s.

A sun in water, dye in it lit from above: a sun a little right and behind (`light.x` 0.2, `light.y` −1,
`light.z` 0.3) at `light.intensity` 18, nearly opaque dye (`opacity` 0.029 widths)
with a faint glow (`emission` 0.1), the water turning blue with depth and dark below (`tint` 0.58, 0.84, 1,
`fog` 0.32 widths), its shafts from `fog_scattering` 0.15 and broad caustics (`caustics` 0.6, `caustic_size` 0.25,
`caustic_depth` 0.8), the dye glowing within by octaves (`scattering` 0.4), dark-rimmed by `powder` 0.6,
its shadows lifted by `sky_fill` 0.15, the light a little forward (`anisotropy` 0.3, `back_scattering`
0.2). Judged on stillslowfast through `fluid_compare.py` at 1080 × 1920, from the fluid's output alone:
the app's grade after it (its composite's LUT and blend) and the camera's colors on the dye, white on the
recording, are not in that picture.

| Part       | How                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
|------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| shadows    | `Fluid3D._lit`: a light grid at `light.grid`'s share of the simulation's cells each way (`FULL`, all of them, the default; `HALF`, `QUARTER`), each step, each cell's RGB transmittance toward the light: for a sun one sweep of parallel lines (`LightSweep3D`), one thread a line stepping a cell at a time along the axis the light mostly travels and sliding across by its slope, only over the layers where it is in the box (the water before them crossed at once), the density read where the line passes, the grid stored sheared, a line a column (`sweep_size`), and read through `sweep_map`, so the texture's filtering mixes the four lines around a place and every place gets the light of straight lines (the monitor's port `light` shows a sun's grid sheared); for a lamp each cell marched to it (`LightVolume3D`, at most 256 steps); the simulation density's amount summed along the way (the upscaled amount has 16 layers to the simulation's 64, and its detail is finer than a light cell), times the extinction `opacity` gives over the view, per width of the box's depth, and the water's `fog` over the whole distance, both per channel by `tint`, so deep light turns bluish and a lamp falls off; the composite's each sample lit beside its own emission (the glow): its color × the light there × `light.intensity`, made per view as `emission` is (`view_march`); off and released while `light.enabled` is off; a step's GPU time with it on within the noise of one without it, under 0.5 ms, in a smoke run at 448 × 800 × 64 with the upscaled density (not yet in TIMING.md) |
| the light  | `composite.light`, a subpanel, `enabled` on top: off, the composite is the plain march, every light pass skipped and its memory released, its settings kept; `source`, `DIRECTIONAL` (a sun: one direction, parallel shadows) or `POINT` (a lamp, shadows fanning out); `x`, `y`, `z` toward the sun or the lamp's place (±1 the box's sides; x right, y down the screen, z away from the viewer), first above and a little toward the front; `intensity` how bright the light shows fully dyed water filling the view lit straight on, 1 its color; the rest below are the subpanel's too. `Fluid3D.set_light` overrides the settings, the hook for a light the app derives from the pose (a lamp at the nose), the pose pipeline's value handed over; the monitor shows the light grid as the port `light`.                                                                                                                                                                 |
| per layer  | `LayerLight3D`, each step, what the composite reads, per column of the light grid and layer of the march (froxels: the view looks straight down z, so the light grid's columns are the view's), once a light cell and a layer instead of at every 4K sample: the light on the dye, the sun's (its phase, octaves and powder in) and the sky's added, its alpha the caustic net's contrast times the sun's share of the brightness, so the composite reads the net at its own size and moves the sun's light alone; and the rays |
| rays       | while `fog_scattering` and `fog` are on: the water's own light per column and layer, the fog's in-scattering integrated exactly over each layer from its front, two sub-steps a light cell, over each the light taken as even (exp(−fog × from the front) − exp(−fog × to the back)); the water scatters every color alike and takes red out first, so the scattering's share is the grey `fog` over the tinted extinction; × the phase, the caustic net and `fog_scattering` × `light.intensity`; the march adds a layer's value under what lies in front, so the density covers the rays behind it and a shadow's edge falls between layers without banding; the light grid's fog goes on past the box's sides (the water does), a sun's from the plane it enters by, a lamp's over its whole way |
| caustics   | `CausticNet`, while `caustics` is on: the bright net sunlight makes under a wavy surface, a 256² tileable texture each step, two Voronoi patterns of drifting points (their cells' edges the net's lines, the second finer), its mean 1 so it moves light without adding any; read per sample of the lit dye and per sub-step of the rays (`light.glsl`'s `caustic`), the same all along each of the light's lines so it is carried down them as shafts: a sun's net laid flat across its way, a lamp's spread from it; `caustic_size` its cell (widths), `caustic_depth` the widths down the light's way over which its contrast fades to 37%, `caustic_speed` its points' turns per second |
| a lamp     | falls off as 1 / its distance², 1 at the box's center, at most 16 times that near it |
| scattering | `scattering`, the light bounced inside the density, made as `scattering_model` says: `BLUR` (the default), the light grid blurred (`LightBlur3D`, a separable 5-tap gaussian) and mixed into the direct light, soft, but it spreads light into empty water and through thin walls of density; `OCTAVES` (Wrenninge), two octaves of the light, each half the extinction and half the light, evenly spread, worked out per sample from the light grid's transmittance (T^½ and T^¼), no pass, so thick parts glow from within as the density shapes them; `powder` (Schneider), Beer–powder over Beer, 2 (1 − T²) on the density's own transmittance (the light grid's alpha, grey, without the water), the edges facing the light darker and thin dye glowing less than thick; `sky_fill`, the light from straight above, swept at a quarter of the simulation's cells each way, soft for it, a share of `light.intensity`, so no shadow is black; `anisotropy`, the Henyey–Greenstein g of each lit sample toward the viewer, 1 at g 0, a silver lining against a light from behind, and `back_scattering` the share of a backward lobe as strong, for a light from the front; a sun's phase is the same everywhere and worked out once on the CPU; all 0 off, at no cost |
| the march  | `march_look`: the look in widths, the same at any number of layers and any depth of the box: `opacity` the widths of fully dyed water that hide half of what is behind it (its half-thickness: its extinction ln 2 / `opacity` per width), `fog` the widths of water that dim white light to half its brightness, turning it toward `tint`'s hue (the water's color, white none: each channel kept 0.5 × `tint` / its brightness over `fog`), `emission` the glow of thick density, 1 its own color, 0 none; each layer sends the share of its light that is the density's, exact for an even layer (its part of the layer's extinction × what the layer takes out), so thick density is seen at its color × `emission` however thick a layer is; the density covers grey by its amount, the water per channel; a particle look takes the density's glow and extinction over its `slices` |
| occlusion  | what covers what measured through the coarse grid's layers, the upscaled density laid over it: the order in depth as fine as the fluid's, the detail at `upscaled.width` × `upscaled.height` × `upscaled.depth`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| highlights | `Knee`, one pass on the output after the particles: the composite's color through a knee, straight up to 0.8 and bent softly toward 1 above it, each channel on its own, so the image below 0.8 keeps its contrast, thick cores do not clip flat and the brightest drift a little toward white, as film does; no setting, `composite.emission` keeps its meaning                                                                                                                                                                                                                                                                                                                                                                        |
| blend      | the composite writes its color premultiplied (times its coverage) and the coverage as alpha, so the fluid layer draws it with the premultiplied form of `composite.blend_mode` (`BlendMode.premultiplied`: `ADD` as `GL_ONE, GL_ONE`, `ALPHA` as `GL_ONE, GL_ONE_MINUS_SRC_ALPHA`), not multiplied by its alpha a second time                                                                                                                                                                                                                                                                                                                                                                                                           |

## Faster

The changes of [TIMING.md's Faster](TIMING.md#faster), one at a time. Each is built behind its own
switch, measured against the baseline, and then kept one of two ways or dropped:

| Outcome                                           | Kept as                                                                                                 |
|---------------------------------------------------|---------------------------------------------------------------------------------------------------------|
| faster, and the same result                       | a replacement: the old path removed, no setting                                                         |
| faster, and a different result (a solver, a look) | an option on the panel, as a field's `advection` is, the current way staying the default until the operator picks |
| not faster                                        | dropped, its row kept in TIMING.md                                                                      |

The test, the same for every item: three timing runs back to back, baseline, change, baseline
(`-timing 60 -warmup 10`, the pose on, standmoves), the pass's own GPU timer the measure: the two
baselines' difference is the noise, and the change counts as faster when it beats the pass it
replaces by at least 2.5 times that; then
`scripts/fluid_compare.py` on stillslowfast, A the baseline, B the change, its numbers and the
composited output side by side (a `--save` to add). The same numbers and the same picture: a
replacement. Different: an option, judged on the look in the app.

| #   | Change                                                   | Expected                                  | Kept as, if faster                                                                      |
|-----|----------------------------------------------------------|-------------------------------------------|-----------------------------------------------------------------------------------------|
| 1   | the temperature cleared once, its injection skipped at 0 | 0.4 to 0.5 ms                             | replacement                                                                             |
| 2   | the composite's early exit and the occupancy volume      | 0.5 to 1.5 ms                             | replacement                                                                             |
| 3   | 3D workgroups for the stencil passes                     | 0.3 to 0.6 ms                             | replacement                                                                             |
| 4   | MacCormack in one dispatch                               | 0.4 to 0.6 ms                             | replacement                                                                             |
| 5   | fused passes                                             | 1 to 1.5 ms                               | replacement                                                                             |
| 6   | red-black SOR                                            | 1.5 to 2.5 ms                             | option, decided: `pressure.solver`, `JACOBI` (the default), `SOR`, whatever it measures |
| 7   | multigrid, or MGPCG                                      | 2.5 to 3.5 ms                             | option, decided: the same option gains `MULTIGRID`                                      |
| 8   | the sparse grid                                          | 0 to half the step                        | its plan written, not built: it touches every pass                                      |
| 9   | the sprites cut at 2 sigma                               | half the blobs' fill                      | replacement; the growth cap stays as it is, a look                                      |
| 10  | the draw by the alive count                              | the ring's empty share                    | replacement                                                                             |
| 11  | weighted blended OIT                                     | the march, the clear, a third of the draw | option: `overlap` gains `WEIGHTED`                                                      |
| 12  | points and streaks rasterized in compute                 | 2 to 5 × on the draw                      | replacement for points; streaks if the lines match                                      |
| 13  | the narrow-range filter and the narrow band              | 0.3 to 1 ms                               | the band a replacement; the filter an option if the surface differs                     |
| 14  | the Morton sort                                          | under 0.5 ms at 2 M                       | replacement; the ring's fade must then go by the particle's age, not its slot           |

Beside 5 and 6, the memory a step: the bytes read and written per cell by every pass, counted from
the shaders before and after, since the passes are bandwidth-bound and the three force and curl
volumes (24 bytes a cell) exist only between passes. A winning solver becomes the preset's
`pressure.solver` too.

What each came to, measured, is [TIMING.md's Measured](TIMING.md#measured): kept as replacements 1, 3, 5,
9 and 10; options 6 (`SOR`), 7 (`MULTIGRID`, the preset's) and 11 (`WEIGHTED`); dropped 2, 12 and 13,
slower; not built 4 and 14, for the reasons there; 8 parked ([Parked](#parked)).

The panel gains at most three options from it: `pressure.solver`, `overlap`'s `WEIGHTED`, and the
surface's filter if it comes. Everything else is the same picture faster, and leaves no trace but
its row in TIMING.md.

## The staggered velocity

Built (2026-11-09). The velocity is on the cells' faces, the MAC grid (Harlow and Welch 1965; Bridson, *Fluid Simulation for Computer Graphics*,
2nd ed. 2015, ch. 2): each component on the faces across its own axis, x between left and right
neighbors, y between upper and lower, z between front and back. The divergence and the gradient become
one-cell differences, whose product is the 7-point Laplacian the solvers solve, so an exact solve leaves
the water exactly incompressible, with no pattern alternating cell by cell. A collocated grid patched by
Rhie–Chow interpolation, or a solve of the wide Laplacian the central differences make, are the
alternatives: the first is engineering CFD's fix and keeps a velocity the projection did not make
divergence-free, the second decouples the cells into a checkerboard.

| Part                         | How                                                                                                     |
|------------------------------|---------------------------------------------------------------------------------------------------------|
| storage                      | three R16F volumes, one per component, each one face longer along its own axis                          |
| the step                     | each pass reads and writes the faces; the pressure, the density and the temperature stay at the centers |
| the centers                  | one pass at the step's end averaging the faces to an RGBA16F volume at the cells' centers               |
| the density, the temperature | advected by the faces; the particles, the upscaled density and the monitors read the centers            |
| unchanged                    | the pressure solvers, the multigrid, the obstacle's cells, the bridge, the motion                       |

The storage, `FaceVelocity`: one RGBA16F volume at 257 × 449 × 129 for 256 × 448 × 128 cells, a texel's
x, y and z the faces on its cell's left, bottom and front, w unused. Three R16F volumes, one a component
(x's faces 257 × 448 × 128, y's 256 × 449 × 128, z's 256 × 448 × 129), were timed against it in the app
and were slower in every pass, so the packed volume is the one kept: a row of 256 cells has 257 faces, the box's two walls included. The far wall's face is
stored, since at an open wall it is an unknown the projection sets; at a closed wall it is 0. The grid,
every dispatch and the multigrid's levels stay multiples of 16: one thread per cell, the last cell's
thread along an axis also writing that far face. Any size is a valid texture, and the driver pads to its
own tiles, so 257 costs what a hand-padded 272 would **(deduction)**. One component per volume, not an
RGBA texel per cell: an interpolated read of one component fetches only that component, 2 bytes a
texel instead of 8, and nothing is stored unused. R16F takes image load and store and trilinear
filtering as the temperature does. Each volume sits half a cell off along its own axis, so each read
takes its own uv. One volume: RGBA16F at 257 × 449 × 129, a texel's x, y and z the faces on its cell's
left, bottom and front, w unused; one fetch reads all three where a pass reads them together (the
divergence, the gradient), but an interpolated read of one component fetches all four.

The step, pass by pass:

| Pass                      | On the faces                                                                                                                                                                                                                                                            |
|---------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| the velocity's advection  | each face traced back on its own from the velocity at the face (its own component, the other two averaged from their four nearest faces), its component read at the departure point from its own volume (Bridson ch. 3)                                                 |
| MacCormack                | the same per component, its bounds the eight texels of that component's volume                                                                                                                                                                                          |
| the top speed             | on the full vector at the face, the other two components averaged as in the trace                                                                                                                                                                                       |
| the slip                  | the tangential faces along a closed wall braked by k                                                                                                                                                                                                                    |
| the viscosity             | each component's Laplacian on its own faces                                                                                                                                                                                                                             |
| the curl, the confinement | the curl at the centers straight from the faces, each derivative a difference of two face averages, so the step needs no centers' volume midway; the force averaged to the faces (Fedkiw, Stam and Jensen 2001)                                                         |
| the buoyancy              | the temperature and the density averaged to each face, x, y and z, since gravity has all three (`gravity_x`, `gravity_y`, `gravity_z`)                                                                                                                                  |
| the drag, the body force  | the bridge's textures read at each face's place                                                                                                                                                                                                                         |
| the divergence            | per cell, its six faces' one-cell differences                                                                                                                                                                                                                           |
| the gradient              | per face, its two cells' pressures                                                                                                                                                                                                                                      |
| the obstacles             | a face with a solid cell on either side solid, 0 and never written; see below                                                                                                                                                                                           |
| the walls                 | closed: the faces on the wall 0, never written; open: the face on the wall set by the gradient from the ghost pressure −p; the curl's tangential ghost (2k − 1); the face volumes sampled clamped to their edge, so a face along the wall reads its own value beyond it |

The order in a step. The velocity is on the faces from the advection to the projection; the curl reads
the faces directly, so nothing midway needs the centers. After the projection one pass averages the faces
to the centers, the velocity the projection made, which the particles, the upscaled density and the
monitors read.

The density and the temperature. Their advection reads each component from its own faces at the traced
point: the field the projection made divergence-free, interpolated once. The centers' copy is the faces
averaged, a two-tap blur along each axis, then interpolated again, so it smooths more and is not exactly
divergence-free. The 4K upscaled density moves to the faces too only if its timing allows: three reads a
texel instead of one.

The obstacles: a solid cell is one whose obstacle value (R8, at the cells, extruded from the 2D mask
`set_obstacle`) is above 0.5, static, so a solid face's velocity is 0. A face between two fluid cells is
fluid; a face with a solid cell on either side is solid, held at 0 like a closed wall's face, so the
divergence reads it as no flow through, the gradient skips it and every pass that writes the velocity
leaves it. The pressure's solvers keep their rule, a solid neighbor's pressure the cell's own, the
Neumann condition the solid face makes. The faces along an obstacle are fluid, so the water slides past
it; the curl's ghost inside a solid cell is the closed wall's, (2k − 1) of the face along it, so
`slip_time` acts on obstacles as on the walls: an obstacle is a closed wall inside the box. A trace
ending inside a solid cell reads the faces there, 0 on its solid sides, as today's advection reads the
cell's own value **(deduction)**. Cut cells (Batty, Bertails and Bridson 2007), the obstacle as a
fraction of a cell, are for later.

Cost and gain **(deduction)**, against today's RGBA16F at the centers:

| Pass                     | Expected                                                                                            |
|--------------------------|-----------------------------------------------------------------------------------------------------|
| the velocity's advection | slower: three traces a cell instead of one, each a velocity at its face and a read at its departure |
| the centers' average     | new: one pass reading the faces, writing the RGBA volume                                            |
| the divergence           | faster: six 2-byte faces a cell instead of six 8-byte neighbors                                     |
| the gradient             | faster: the velocity read and written at 6 bytes a cell instead of 16                               |
| the viscosity            | faster where on: 6 bytes a cell an iteration instead of 8                                           |
| the step                 | about even to 1 ms slower at 256 × 448 × 128                                                        |

The look **(deduction)**: the density's brightness kept (no bunching or thinning by compression), less
cell-by-cell jitter for the confinement to spin up, swirls holding longer, pushes carrying further,
sharper density edges; subtle at the calm preset.

Measured before and after, the same runs on the code before (the baseline, taken first) and with the
staggered velocity:

| Test        | Run                                                                                                                                                                                                                                      |
|-------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| timing      | `trial_runs.py`, baseline, staggered, baseline (`-timing 60 -warmup 10`, the pose on, standmoves): the step and each pass, the noise the two baselines' difference ([TIMING.md](TIMING.md#measured))                                     |
| the floor   | the velocity's divergence after the projection, a cell's share, per solver, each grid by its own stencil: what the solve leaves on the grid it solves                                                                                    |
| compression | a square of density in a closed box, a steady swirl, no fade and no evaporation: its total and its peak over 10 s, the same measure on both grids, since the water squeezed or stretched gathers or thins the density as the eye sees it |
| the walls   | `maccormack_check.py`: the wall band and the inside, closed and open, both advections, ε 1.93 and 5, `slip_time` 0 and 2                                                                                                                 |
| the detail  | `maccormack_check.py`: the density's edge and the swirl kept after 2 s                                                                                                                                                                   |
| the water   | `fluid_compare.py` on stillslowfast and standmoves: water over body, peak, coast                                                                                                                                                         |
| the look    | `fluid_compare.py --save` on stillslowfast: the composited output at four steps, side by side                                                                                                                                            |

Sources: Harlow and Welch 1965, the MAC grid; Bridson 2015, ch. 2 (the grid), 3 (advecting it), 5
(the pressure on it); Fedkiw, Stam and Jensen 2001, the confinement on it; the walls folded into the
operators on the GPU (Zaspel).

### Measured

The same runs on the centered velocity (the baseline, `28655856`) and on the faces, deep_flow's preset. In the app (`trial_runs.py`, standmoves, the pose on), ms mean, two runs each within 0.03:

| Pass                     | Centered | Faces | Faces, three R16F |
|--------------------------|----------|-------|-------------------|
| advect velocity          | 0.18     | 0.35  | 0.46              |
| pressure                 | 1.80     | 1.61  |                   |
| density                  | 1.48     | 1.59  |                   |
| faces to centers         |          | 0.16  | 0.24              |
| the step                 | 4.44     | 4.61  |                   |
| viscosity, 20 iterations | 1.16     | 0.95  | 4.88              |

The viscosity works out each face's weights (the walls', the slip's, the obstacles') once a dispatch;
worked out for every face they cost more than the iterations (3.6 to 4.0 ms), so only the faces next to a
wall or an obstacle do, the rest summing their six neighbors plainly, and the test whether a face is held
reads all three components at once.

The swirl (`maccormack_check.py --test swirl`, multigrid), at 10 s:

| Advection       | Swirl kept, centered | faces | Density total | The floor, centered | faces   |
|-----------------|----------------------|-------|---------------|---------------------|---------|
| semi-Lagrangian | 0.602                | 0.651 | the same      | 0.00009             | 0.00010 |
| MacCormack      | 0.573                | 0.625 | the same      | 0.00050             | 0.00010 |

The floor at the multigrid is the half float's rounding on both grids: 4 cycles leave the same 0.0001.
Jacobi's 40 iterations leave 0.002 on both, and the density's total in the swirl grows by a third, the
solve unconverged. On the swing at ε 1.93, closed, the water keeps more of its speed (inside at 12 s
0.0103 against 0.0080, semi-Lagrangian), and at ε 5 MacCormack's inside holds at 0.033 where the centered
held at 0.021; `slip_time` 2 decays as before. Through `fluid_compare.py` the water against the body is
the same within 0.01 on stillslowfast and standmoves. An obstacle's cells read 0 (`--obstacle`). Open
walls decay as the centered velocity's did (the swing at ε 1.93, the mean at 4 s and 6 s: 0.0073 and
0.0032 against 0.0077 and 0.0031); the open swirl drains alike, its floor 0.0001 against the centered
0.0005 to 0.0015.

## Parked

Not worked on until asked for; each with what is known.

- The 4K upscale: depth in 3D at the grid and detail in 2D at 4K, a march at the grid's size, one
  edge-aware upscale of the flat result and a 4K detail layer advected by the grid's velocity; what
  was found is in [The density's detail](#the-densitys-detail).
- The particles' weight: the particles splatted into an `r32ui` grid by fixed-point
  `imageAtomicAdd` (it needs integers), converted to the density volume the buoyancy reads.
- The sparse grid: each step a pass marks the 8 × 8 × 8 bricks whose velocity or density is above
  a floor, grown by one brick, into a list with an atomic counter; every pass but the pressure then
  runs `glDispatchComputeIndirect` over the list, one workgroup a brick, and a brick off the list
  keeps its values. The volumes stay dense: 24 GB holds them, so GVDB's atlas, which saves memory,
  buys nothing here. The multigrid's coarse levels stay whole, an eighth and less of the work. Its
  gain is the share of bricks off the list; first one cheap pass counts the active share on
  standmoves and stillslowfast, and the grid is built only if it is under half.
- MacCormack's cost at 4K: a 4K texel reads 41 texels a step against semi-Lagrangian's 16 (the
  predictor, the velocity again, the forward trace, the cell, the limiter's eight), 3.7 ms for the
  4K × 16 density. Two ways down: MacCormack on the velocity and semi-Lagrangian on the 4K density,
  a setting; or the predictor reading its eight texels itself and keeping their min and max for the
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
