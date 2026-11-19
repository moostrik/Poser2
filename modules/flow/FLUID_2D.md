# The 2D fluid

Every quantity and setting here is in a meaningful unit ([FLOW.md](FLOW.md#the-2d-and-3d-flow-systems)).

The 2D fluid is the 3D fluid on a plane: the same inputs, settings, names and laws for what a plane has, on its own
fragment shaders. What each law is, and why, is in [FLUID_3D.md](FLUID_3D.md); this document holds what the 2D does
differently. How the stages fit together is in [FLOW.md](FLOW.md), what is next in superficial_flow's plan
([SUPERFICIAL_FLOW_PLAN.md](../../apps/superficial_flow/SUPERFICIAL_FLOW_PLAN.md)). The fluid is `Fluid2D`
(`fluid_2d/fluid_2d.py`), its layer `Fluid2DLayer` (`modules/render/modules/flow/fluid_2d_layer.py`), run by
superficial_flow; hd_trio runs its own copy (`apps/hd_trio/flow/`).

## What it is

| Part         | The 2D                                                                                                            |
|--------------|-------------------------------------------------------------------------------------------------------------------|
| the grid     | `width` × `height` cells, each rounded up to a multiple of 8; 1080 × 1920 by default, half the 4K output each way |
| the velocity | at the cells' centers (RG16F), widths/s, x right and y as the 3D's; no faces                                      |
| the density  | at the output's size, apart from the grid (RGBA16F, premultiplied), carried by the grid's velocity                |
| the cell     | in widths: 1 / `width` across, 1 / (aspect × `height`) high, the aspect the density's width over its height       |
| the passes   | each a fragment shader; the pressure's solvers and the viscosity's compute shaders                                |

A step runs in the 3D's order ([FLUID_3D.md](FLUID_3D.md#the-fluid)): the velocity carried along itself, its
viscosity, its confinement, the buoyancy, the projection with the expansion, then the temperature and the density
carried by the projected velocity. The inputs come before the step, as the 3D takes them
([FLUID_3D.md](FLUID_3D.md#the-inputs)), each its own 2D fragment shader (`InjectVelocity2D`, `InjectDensity2D`,
`InjectScalar2D`).

## The settings

The panel is the 3D's for what a plane has: each setting sits at the 3D's path, under its name, in its unit and with
its default, the sizes apart (`modules/flow/tests/test_fluid_2d_settings.py`). What the 2D has of each group:

| Group         | The 2D's settings                                                                              | The 3D's law                                                                                     |
|---------------|------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------|
| root          | `guard`, `wall`, `slip_time`, `width`, `height`                                                | [the guard](FLUID_3D.md#the-guard), [the walls](FLUID_3D.md#the-walls)                           |
| `time`        | `time_step`, `fixed_fps`, `fps_smoothing`, `time_scale`                                        | every pass and input steps by the fluid's time                                                   |
| `input`       | the drags, the gains, the density's blend and release, gravity's `gravity_x` and `gravity_y`   | [the inputs](FLUID_3D.md#the-inputs)                                                             |
| `velocity`    | `advection`, `limiter`, `advection_scale`, `fade_time`, `evaporation_time`, `evaporation_knee` | [the advection](FLUID_3D.md#the-advection), [the fade](FLUID_3D.md#the-fade-and-the-evaporation) |
| `viscosity`   | `enabled`, `viscosity` (widths²/s), `iterations`                                               | implicit diffusion, Jacobi                                                                       |
| `vorticity`   | `enabled`, `strength_1` to `_3`, `radius_1` to `_3` (widths)                                   | the force ε h (N × ω) dt, h the cell's width (Fedkiw 2001)                                       |
| `temperature` | `enabled`, `buoyancy`, `fade_time`, `evaporation_time`, `evaporation_knee`                     | [the buoyancy](FLUID_3D.md#the-buoyancy)                                                         |
| `expansion`   | `enabled`, `rate`                                                                              | [the expansion](FLUID_3D.md#the-expansion)                                                       |
| `pressure`    | `solver`, `iterations`, `relaxation`, `cycles`                                                 | [the pressure](FLUID_3D.md#the-pressure)                                                         |
| `density`     | `color_space`, `sink`, `advection`, `limiter`, `advection_scale`, `fade_time`, the evaporation | [the density's blend](FLUID_3D.md#the-densitys-blend), the buoyancy                              |

superficial_flow's preset holds deep_flow's value for every setting the two share, the sizes a textbook default
(`apps/superficial_flow/tests/test_fluid_preset.py`). deep_flow's gravity points along z, so on the plane it is
(0, 0) and the density's `sink` moves nothing.

## What it does differently

- The walls are the 3D's ([FLUID_3D.md](FLUID_3D.md#the-walls)), written once (`walls_2d.glsl`) and read by every
  pass that reads a neighbor beyond the field: the divergence, the gradient, every pressure solver and multigrid
  level, the curl and the viscosity. Beyond a closed wall the velocity's component through it is turned and along
  it (2k − 1) of it, the pressure the cell's own; beyond an open wall the velocity the same, the pressure its
  negative, 0 on the wall's face. The velocity sits at the cells' centers, so the wall's face is the mean of the
  edge cell and its ghost: closed, nothing through it. A trace that reads beyond a wall reads the field's
  texture: closed, the edge cell; open, still water, 0. The density and the temperature are read the same way.
- A closed wall slips as the 3D's: k = 0.01^(dt / `slip_time`), the velocity along the wall of the cells against
  it multiplied by k in the advection, 1 free-slip.

  Checked in the GPU harness, 2026-11-19, superficial_flow plan D5, at 64 × 64: a flow along a closed wall shows
  no curl at it and the viscosity takes nothing; at `slip_time` 2 the wall's cells keep 0.4773 of 0.5, the
  law's 0.4775, the inside 0.5; a flow of 1 width/s into the walls, projected, keeps a mean of 0.016 closed and
  1.000 open.
- The confinements are the 3D's three ([FLUID_3D.md](FLUID_3D.md#the-fluid), its passes), each
  its strength and its radius in widths, each running while its strength is above 0. On the plane the curl is a
  scalar, ω along z, so the three fit one texture, a channel each: `VorticityCurl` writes them in one pass,
  `VorticityForce` sums their forces ε h (N × ω) dt, N × ω = (N.y ω, −N.x ω), in one, added to the velocity once.
  Beyond the walls the curl's magnitude is the edge cell's, as the 3D's ghost: no slope toward a wall for the
  force to push along.

  Checked in the GPU harness, 2026-11-19, superficial_flow plan D6, at 128 × 128: a rigid rotation's curl is 2Ω in
  each channel at radii of 2, 5 and 9 cells; on a vortex, confinement 2 alone gives confinement 1's force to the
  bit, two alike twice it.
- The solvers are the 3D's on the plane, each its own 2D kernel ([FLUID_3D.md](FLUID_3D.md#the-pressure)):
  `JACOBI` five iterations a dispatch in shared memory (`JacobiPressureCompute`), `SOR` red-black over-relaxed
  (`SorPressure2D`), `MULTIGRID` V-cycles (`MultigridPressure2D`: `SorPressure2D` at omega 1 smoothing,
  `ResidualRestrict2D`, `Prolong2D`, the coarsest level in one workgroup by `CoarsePressure2D`), the pressure in R32F.
  At 1080 × 1920 the multigrid has 7 levels below the finest. A coarse cell is two fine cells from 0 along an axis
  the level halves; where the fine level is odd the last reaches past the wall, its missing fine cells a residual of
  0, and the prolongation reads it at half the fine position. The restriction so keeps the residual's sum: with every
  wall closed the coarse right-hand side sums to 0 as the fine one does. Averaged over the fine cells there only, the
  edge cells counted double, a coarse level's sum was not 0, the coarsest level's sweeps grew the pressure's constant
  step after step and the closed walls blew up (JACOBI stayed calm) **(deduction)**.

  Measured in the GPU harness, 2026-11-19, superficial_flow plan D3, a result: one centered velocity (three
  swirling sources, 0 at the walls) projected once from a pressure of 0 at 270 × 480, the residual |div − L p| over
  |div| by `ResidualRestrict2D` at (1, 1), its low frequencies the residual's 8 × 8 means, and the divergence after by
  the fluid's own pass; one run, closed walls (open alike, the multigrid's a little lower):

  | Solver           | Residual | Low frequencies | Divergence after |
  |------------------|----------|-----------------|------------------|
  | Jacobi 40        | 0.966    | 0.972           | 0.959            |
  | Gauss–Seidel 40  | 0.918    | 0.926           | 0.911            |
  | SOR 40, 1.8      | 4.875    | 0.534           | 0.522            |
  | multigrid, 1     | 0.093    | 0.094           | 0.093            |
  | multigrid, 2     | 0.015    | 0.016           | 0.017            |
  | multigrid, 4     | 0.0008   | 0.0007          | 0.0047           |

  From rest the error is the swirls' size, which Jacobi and Gauss–Seidel barely reach in 40 sweeps; SOR's
  over-relaxation reaches it and leaves the high frequencies worse **(deduction)**. Each V-cycle divides the
  residual by six or more.
- MacCormack is the 3D's on the plane, its own kernel (`MacCormack2D`, [FLUID_3D.md](FLUID_3D.md#the-advection)):
  the plain step into a predictor per field (`_velocity_scratch`, `_temperature_scratch`, `_density_scratch`, each
  allocated at its field's first MacCormack step at its size), then half its error traced forward added back,
  limited by the source's four texels around the backward trace. The temperature takes the density's `advection`
  and `limiter`.

  Measured in the GPU harness, 2026-11-19, superficial_flow plan D4, a result: a 128 × 128 grid under a 256 × 256
  density, 50 steps of 0.02 s; one run each:

  | Measure                                                     | Semi-Lagrangian | MacCormack, clamp | MacCormack, revert |
  |-------------------------------------------------------------|-----------------|-------------------|--------------------|
  | a square at (0.3, 0.2) widths/s: its edge, 99.5% gradient   | 0.153           | 0.184             | 0.184              |
  | a swirl carried and projected: its mean speed, of its start | 0.947           |                   | 0.977              |

- The guard counts the velocity, the density and the temperature in the advection, the one pass that writes each.
- One GPU timer, `GPU fluid`, holds the step; the layer's `GPU fluid inject` the inputs.
