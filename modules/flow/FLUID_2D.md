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
| the passes   | each a fragment shader, the two solvers compute shaders with 5 Jacobi iterations a dispatch in shared memory      |

A step runs in the 3D's order ([FLUID_3D.md](FLUID_3D.md#the-fluid)): the velocity carried along itself, its
viscosity, its confinement, the buoyancy, the projection with the expansion, then the temperature and the density
carried by the projected velocity. The inputs come before the step, as the 3D takes them
([FLUID_3D.md](FLUID_3D.md#the-inputs)), each its own 2D fragment shader (`InjectVelocity2D`, `InjectDensity2D`,
`InjectScalar2D`).

## The settings

The panel is the 3D's for what a plane has: each setting sits at the 3D's path, under its name, in its unit and with
its default, the sizes apart (`modules/flow/tests/test_fluid_2d_settings.py`). What the 2D has of each group:

| Group         | The 2D's settings                                                                             | The 3D's law                                                             |
|---------------|-----------------------------------------------------------------------------------------------|--------------------------------------------------------------------------|
| root          | `guard`, `wall`, `width`, `height`                                                            | [the guard](FLUID_3D.md#the-guard), [the walls](FLUID_3D.md#the-walls)   |
| `time`        | `time_step`, `fixed_fps`, `fps_smoothing`, `time_scale`                                       | every pass and input steps by the fluid's time                           |
| `input`       | the drags, the gains, the density's blend and release, gravity's `gravity_x` and `gravity_y`  | [the inputs](FLUID_3D.md#the-inputs)                                     |
| `velocity`    | `advection_scale`, `fade_time`, `evaporation_time`, `evaporation_knee`                        | [the fade and the evaporation](FLUID_3D.md#the-fade-and-the-evaporation) |
| `viscosity`   | `enabled`, `viscosity` (widths²/s), `iterations`                                              | implicit diffusion, Jacobi                                               |
| `vorticity`   | `enabled`, `strength_1`, `radius_1` (widths)                                                  | the force ε h (N × ω) dt, h the cell's width (Fedkiw 2001)               |
| `temperature` | `enabled`, `buoyancy`, `fade_time`, `evaporation_time`, `evaporation_knee`                    | [the buoyancy](FLUID_3D.md#the-buoyancy)                                 |
| `expansion`   | `enabled`, `rate`                                                                             | [the expansion](FLUID_3D.md#the-expansion)                               |
| `pressure`    | `iterations`                                                                                  | [the pressure](FLUID_3D.md#the-pressure)                                 |
| `density`     | `color_space`, `sink`, `advection_scale`, `fade_time`, `evaporation_time`, `evaporation_knee` | [the density's blend](FLUID_3D.md#the-densitys-blend), the buoyancy      |

superficial_flow's preset holds deep_flow's value for every setting the two share, the sizes a textbook default
(`apps/superficial_flow/tests/test_fluid_preset.py`). deep_flow's gravity points along z, so on the plane it is
(0, 0) and the density's `sink` moves nothing.

## What it does differently

- The velocity is 0 beyond every wall, open or closed (its texture's border): on a plane the cells carry no faces
  to hold a wall's normal apart from its tangent. The pressure, the density and the temperature follow `wall`:
  closed, beyond it the edge cell's value (Neumann); open, 0 (Dirichlet). The two compute solvers write the same
  ghosts beyond the field as the fragment passes' wraps, at any grid size.
- The confinement's curl is a scalar, ω along z, and N × ω is (N.y ω, −N.x ω). Beyond the walls the curl's
  magnitude is the edge cell's, as the 3D's ghost: no slope toward a wall for the force to push along.
- The guard counts the velocity, the density and the temperature in the advection, the one pass that writes each.
- One GPU timer, `GPU fluid`, holds the step; the layer's `GPU fluid inject` the inputs.
