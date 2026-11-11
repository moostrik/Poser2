# The 2D and 3D flow systems

**Meaningful units.** Every quantity and every setting in the flow, the motion, the bridge and the
fluid alike, is in a unit that means something on its own: widths/s, widths/s², seconds, ms, a dye
from 0 (clear) to 1 (fully dyed), a warmth from −1 to 1, a share of the box's depth. A setting's
value is chosen for what that unit says, never derived from an old look or from another setting by
a factor; a value that would need such a factor means a unit is wrong, and the unit is fixed where
the quantity is made.

Two app-independent flow systems, each a pipeline of three stages, motion, bridge and fluid: the
2D flow, a velocity on the image plane, and the 3D flow, a velocity with a component toward the
camera. Either system takes either kind of camera. An app picks a system and wires its cameras
to it.

"Code" is what the files show; **(deduction)** marks what is inferred from them. This document
holds what the stages share and what is open outside the 3D fluid; each stage's design is its own:

| Document                   | Holds                                                                                       |
|----------------------------|---------------------------------------------------------------------------------------------|
| [MOTION.md](MOTION.md)     | the motion: when it measures, the Lucas–Kanade solve and the depth velocity, their settings |
| [BRIDGE.md](BRIDGE.md)     | the 3D bridge: the filters, the readings, the mix and its outputs' units                    |
| [FLUID_3D.md](FLUID_3D.md) | the 3D fluid: its plan, how it takes the bridge, its fields, density and look               |
| [RESOLUTION.md](RESOLUTION.md) | every size the picture and the motion pass through, what each needs, and its plan       |
| [MONITOR.md](MONITOR.md)   | the monitors' views                                                                         |
| [TIMING.md](TIMING.md)     | every stage's GPU time in deep_flow                                                         |

## The two systems

| System  | Velocity | Motion                                          | Bridge      | Fluid            | Without depth                   | With depth                                         |
|---------|----------|-------------------------------------------------|-------------|------------------|---------------------------------|----------------------------------------------------|
| 2D flow | x, y     | motion on the image plane                       | `bridge_2d` | 2D Navier–Stokes | its normal case                 | depth is not wired; the system never reads it      |
| 3D flow | x, y, z  | motion on the image plane and toward the camera | `bridge_3d` | 3D Navier–Stokes | z is 0, and so is the fluid's w | z from the depth velocity; the fluid takes it as w |

The camera is the other axis. A 2D camera gives an image and a player mask; a 3D camera gives
those and a depth image on the same view. Every combination is a wiring, not a code path:

| Camera | 2D flow                 | 3D flow                              |
|--------|-------------------------|--------------------------------------|
| 2D     | image, mask             | image, mask; z and w are 0           |
| 3D     | image, mask; depth idle | image, mask, depth; z and w measured |

## Terms

| Term                 | Means                                                                                                                                                                                                               | In code                                                                                       |
|----------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------|
| flow                 | the domain: everything from the camera's motion to the fluid                                                                                                                                                        | `modules/flow`                                                                                |
| motion               | the stage that measures a velocity field from two images: the optical flow                                                                                                                                          | `motion/`, `GradientFlow`, `LucasKanadeFlow`, `LucasKanadeFlow3D`                             |
| bridge               | the stage that turns the measured velocity into what the fluid injects: the boundary, and in 3D the body force                                                                                                      | `bridge_2d/`, `bridge_3d/`                                                                    |
| lowpass, bandpass    | the 3D bridge's two filters on the motion: one trail, and the band between a fast and a slow trail                                                                                                                  | `VelocityLowpass`, `VelocityBandpass`, groups `lowpass`, `bandpass`                           |
| reading              | what a bridge filter makes from the motion, as a velocity (widths/s) and as an acceleration (widths/s²): trail, change, start, bow, stop, wake                                                                      | `Change`, `MixSettings`                                                                       |
| boundary             | the body in the water: what the water is dragged toward, a velocity, with the density and the temperature it brings                                                                                                 | `boundary_velocity`, `boundary_density`, `boundary_temperature`, group `boundary`             |
| body force           | the Navier–Stokes force term: an acceleration added × dt, with the density and the temperature it brings                                                                                                            | `body_force_acceleration`, `body_force_density`, `body_force_temperature`, group `body_force` |
| region               | the pixels around one, which tell a part of the body changing speed from a limb arriving or leaving                                                                                                                 | `start_region_size`, `stop_region_size`, `*_region_weight`                                    |
| injection            | one bridge output as one fluid takes it, at its weight                                                                                                                                                              | `Injection`                                                                                   |
| fluid                | the Navier–Stokes simulation; its density is the show                                                                                                                                                               | `fluid_2d/`, `fluid_3d/`, `Fluid2D`, `Fluid3D`                                                |
| composite            | the 3D fluid's look: how its density volume is marched into the output                                                                                                                                              | `CompositeSettings`, group `composite`                                                        |
| carrier              | what carries the 3D fluid's density through its velocity: the simulation density always, the upscaled density and particles where enabled                                                                           | `DensityCarrier`, `SimulationDensity`, `UpscaledDensity`, `ParticleDensity`                   |
| depth                | the camera's depth in metres, on the flow's view                                                                                                                                                                    | `DepthView`                                                                                   |
| depth velocity       | the z of the 3D flow: the depth change along the x/y displacement, in view widths per second                                                                                                                        | `DepthVelocity`, `DepthVelocitySettings`, group `depth_velocity`                              |
| z position           | where a pixel is in the box's depth, a share of it, 0 front to 1 back, from the player's depth range (the depth source's near to far); carried as itself times its weight, and the weight, so none is not the front | `z_position`, `TextureKind.Z_POSITION`                                                        |
| 2D flow, 3D flow     | a system: a motion layer and a bridge layer per camera, a fluid layer per show output                                                                                                                               | `Motion2DLayer` + `Bridge2DLayer` + `Fluid2DLayer`; `…3D…`                                    |
| 2D camera, 3D camera | an image and a mask; those and a depth image on the same view                                                                                                                                                       | `MotionInput` without / with `depth`                                                          |
| port                 | a stage's named texture with its kind (image, velocity, acceleration, scalar, z position) and the channels that hold its values; its input and output fields declare theirs, every component its own ([MONITOR.md](MONITOR.md)) | `Port`, `TextureKind`, `HasPorts`, `ports_of`, in `modules/flow`                              |
| monitor              | a layer that shows one port of one stage, picked on the panel, drawn only while it is shown; its views in [MONITOR.md](MONITOR.md)                                                                                  | `MonitorLayer`, `monitor_a`, `monitor_b`                                                      |

"bridge" is ofxFlowTools' word (`ftVelocityBridge`, `ftDensityBridge`, `ftTemperatureBridge`);
it says where the stage sits, and the glossary defines what it does.

A thing that exists in 2D and 3D carries `2D` or `3D` in its class name, right after the noun it
qualifies, and `_2d` or `_3d` in its package name; the packages lie flat in `modules/flow`, one
per stage and dimension, each its layer's name in lower case (`Bridge3DLayer`, `bridge_3d`), and
`modules.flow` exports every class by name.
A stage's key in an app's settings and its `Layers` entry are the stage's name: `motion`,
`bridge`, `fluid`. The 2D bridge's classes (`SmoothTrail`, `DensityBridge`) keep their names until
the 2D bridge is rebuilt on the Lucas–Kanade flow.

## The stages and their layers

The design follows the light renderer (`apps/white_space/light/`): a stage's input is one
`@dataclass` built for it each tick, its output one `@dataclass` of its own textures, an abstract
layer with a template method holds what the two kinds share, and each kind is a subclass. One
layer per stage, so the layers are the packages. Above the three abstract layers, `StageLayer`
states what a stage is: an input, an output that carries its ports, and a reset. Only the
interface is shared: the templates differ, the motion running per camera frame and the bridge and
fluid per tick. Inputs have no common base, since each stage reads its own fields and nothing takes
any stage's input.

### Inputs and outputs

Each stage takes one input dataclass and gives one output dataclass. Both hold values only: textures and
numbers, never an output; the fluid's input holds its injections as a list of `Injection`, each values.
The app's wiring pass builds each input new every tick, copying into it what the stage reads from the
output of the stage before and from the camera layers, and sets it on the layer before the render loop.
The output holds the stage's textures, allocated once, and the layer hands it out as `output`; an output
is a stage's only way out, and the next stage reads its fields by name. A texture field declares its kind
on the field (`velocity: Annotated[Texture, TextureKind.VELOCITY]`), in inputs and outputs alike
(`ports_of`), and every component that owns textures names them once with their kinds in its `ports`
(`HasPorts`): `FlowBase` its `input` and `output`, a subclass its others. A stage names its components
once (`_components`), by their settings group, and its `ports`, what the monitors show, is one list with
the data running bottom to top, the prefix its group: `output.*` on top, each component's from the last
the stage runs to the first (`z.track`, `xy.solve`, `bandpass.fast`, `lowpass.trail`), `input.*` at the
bottom. A texture not allocated is left out, so a part the stage does not run gives none. Not by
reflection over attributes: it would find the FBOs, not their kinds, and a swap's back buffer would be a
port too. A layer's `texture` stays what it draws, read by the composite. The layers hold no reference to
each other, and no input knows an output's type. A field is named with the same noun for the same kind of
thing, in every input.

A texture field holds the `Fbo`, `SwapFbo` or `Tensor` object itself; a swap is its current
buffer, and its `back` is the other one, read inside the pass that writes the current and never
kept, since the two trade places at the next swap. `FlowBase` hands out its `SwapFbo`s for the
same reason. That is what lets an input be built before the render loop: the object named in it
is the one the stage before writes into during the loop, so the next stage reads this tick's
result. Values that are not textures (`timestamp`, `dt`, the weight) come from the app, never from
an output, or they would be the tick before's.

| Stage   | Input `@dataclass`, built by the app each tick                                                                                                                                                                                                                                                                                                                                                     | Output `@dataclass`, the stage's textures                                                                                                                                                                                                                                                                                                                                                                                                                                  |
|---------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| motion  | `MotionInput`: `image`, the picture the velocity is measured on by its brightness; `timestamp`, the capture time of the camera frame `image` holds, the frame's identity; `dt`, the render tick's; `depth`, a `DepthView` or `None`; `mask`, or `None`; `depth_mask`, where the depth is the player's, or `None` | `MotionOutput`: `velocity` (x, y, z), gated by `mask` with the 3D flow's `use_mask`, z also by `depth_mask`; z is 0 without depth; `z_position`, the body's now, the 3D motion's, or `None`                                                                                                                                                                                                                                                                                                       |
| bridge  | `BridgeInput`: `velocity`, from the motion output; `dt`; `image`, or `None`, the picture whose RGB colors the density; `color`, an RGB tint on it, white by default; `mask`, or `None`, which gates the image; `z_position`, the motion's, or `None`                                                                                                                                                  | `BridgeOutput`, what the fluid injects in two systems ([BRIDGE.md](BRIDGE.md#the-mix)): the boundary, `boundary_velocity`; `boundary_density`, in 2D the amount alone, the trail's speed (R), in 3D RGBA, the amount times `color`, times the image's RGB where there is one, with the amount as alpha; `boundary_temperature`; and the 3D bridge's body force, `body_force_acceleration`, `body_force_density`, `body_force_temperature`, each `None` while it gives none; where each goes in depth, the 3D bridge's: `boundary_velocity_z_position`, `body_force_acceleration_z_position` and `z_position`, the body's now, for the densities and the temperatures ([BRIDGE.md](BRIDGE.md#the-z-positions)) |
| fluid   | `FluidInput`: `dt`, and `injections`, a list of `Injection`: the six outputs of one bridge output and their z positions, with `weight`, 0 to 1, the injection's share of the fluid's velocity, density and temperature, as a layer's weight in the light renderer's mix, and `color`, the color the 2D fluid gives the injection's density at its output; the app builds one injection per bridge it wants in that fluid | `FluidOutput`: `density`, the show, which the composite draws as the layer's `texture`                                                                                                                                                                                                                                                                                                                                                                                     |
| monitor | `MonitorInput`: `ports`, every stage's ports keyed by stage; the monitor picks by its own `stage` and `texture`                                                                                                                                                                                                                                                                                    | none: it draws                                                                                                                                                                                                                                                                                                                                                                                                                                                             |

Two clocks. The motion layer measures once per camera frame ([MOTION.md](MOTION.md#when-it-measures)),
between two camera frames, so its interval comes from `timestamp` (`frame_interval`, `motion_layer.py`),
held to half to twice the expected frame interval, 1 / the camera's set frame rate (the motion settings'
`fps`, shared from the app's `input_fps`). The range is fixed, not learned from the gaps, so a burst of
frames a few ms apart while the camera comes up to speed is held once and does not carry into the frames
after it; `dt` is the fallback for a source without stamps or frame rate. The bridge and the fluid run
per render frame, so they step on `dt`, the tick's measured length, which the app holds to at most twice
the mean tick (`hold_tick`, `time_step.py`): a stall counts as one long frame and the rest is dropped, as
a capture gap is held to twice the expected frame interval. The fluid's `TimeStep` smooths it over
`fps_smoothing`; a raw per-frame dt looks jittery in the sim **(site fact)**.

A tick without a new camera frame measures nothing: the motion output holds what the last pair
of frames gave, the tick's `dt` adds to the interval the next measurement covers, and the bridge
and the fluid run as on any tick, on the held velocity. A velocity held between samples and
integrated at the render rate is the right reading of it: the motion continues until the next
pair says otherwise, and the stamps keep the magnitude per second right whether a frame took one
tick or two. The bridges are filters in time (their trails decay by `decay_weight(dt, smooth)`,
the lowpass's advect carries it by velocity × dt), so a held velocity
stepped every tick on `dt` is the same filter sampled finer. Between two 50 fps frames the render
takes one or two 60 Hz ticks, so one frame's integral of the held velocity is 17 or 33 ms worth
instead of 20; the mean is right, and the trails and the fluid smooth it **(deduction)**.

The render runs on the display's vsync (`WindowManager._render_loop`, `swap_interval(1)`); the
camera runs at 50 fps, the mains rate **(site fact)**, against a 60 Hz display. The board holds
one latest capture per camera (`CaptureStoreMixin.set_capture`), which `CaptureImageLayer.update`
takes once per tick. At these rates every frame gets a tick of its own and none is lost; a frame
waits for the next vsync, up to one 60 Hz tick, before it is measured. That wait is a limit of
the design: every stage runs on the render tick, so measuring on arrival, on a thread and GL
context of its own, would move the wait to the bridge, and the screen still shows at vsync.

### The layers

| Class                                                              | Role                                                                                  |
|--------------------------------------------------------------------|---------------------------------------------------------------------------------------|
| `StageLayer[I, O]` (ABC)                                           | what a stage is: an input, an output carrying its ports, its parts, a reset           |
| `MotionLayer` (ABC)                                                | `StageLayer[MotionInput, MotionOutput]`: measures on a new camera frame               |
| `Motion2DLayer`, `Motion3DLayer`                                   | on `GradientFlow`; on `LucasKanadeFlow3D`                                             |
| `BridgeLayer` (ABC)                                                | `StageLayer[BridgeInput, BridgeOutput]`: the density's color, then the bridges        |
| `Bridge2DLayer`, `Bridge3DLayer`                                   | on `bridge_2d`; on `bridge_3d`, the velocity lowpass beside the bandpass              |
| `FluidLayer` (ABC)                                                 | `StageLayer[FluidInput, FluidOutput]`: injects, then steps; its `texture` is the show |
| `Fluid2DLayer`, `Fluid3DLayer`                                     | on `Fluid2D`; on `Fluid3D`                                                            |
| `MonitorLayer`                                                     | shows one port of one stage, while shown; not a stage                                 |
| `MotionLayerSettings`, `BridgeLayerSettings`, `FluidLayerSettings` | the stages' base settings, each a `StageLayerSettings`                                |

`StageLayer`: `set_input(I)` before the render loop; `output: O`, a `StageOutput`, whose fields with
a kind are its ports; `_components()`, the parts by group name; `ports`, input, parts, output;
`reset()`, from rest; `update()` takes a reset the panel's `reset` button asked for, on the render
thread, then runs `_update()`. It unbinds the button in `deallocate()`.

`MotionLayer`'s `_update()` is the template method: `_measure` on a new frame. It holds the allocate
and resize pattern; the flow runs at `settings.width` by `settings.height`, and it draws nothing. The
2D flow's velocity is per frame, not per second, and it reads no mask, until the 2D flow moves to the
Lucas–Kanade solve.

`BridgeLayer`'s `_update()`: the density's color (`DensityColor`: the image, or white, times the tint,
gated by the mask), then `_bridge` the input's velocity into velocity, density and temperature
([BRIDGE.md](BRIDGE.md)). The bridges' filters run at the velocity's size and their outputs at
`settings.width` by `settings.height`; the layer follows both. It draws nothing.

`FluidLayer`'s `_update()`: `_inject` each injection's velocity, density and temperature at its
weight, then `_step(dt)`. It knows injections, not cameras, and reads no poses; the stage's reset is
the fluid's only one. `Fluid2DLayer` keeps its injections apart: injection i's amount goes into
channel i of the density, four at most, and the output colors each channel with its injection's
color (`DensityColorize`), the brightness capped at white and the opacity the strongest channel's.
`Fluid3DLayer` drags the fluid toward the boundary's velocity, adds the body force's acceleration ×
dt and injects both systems' RGBA density and temperature, each velocity's z as the volume's w (its
sign turned: w points away from the viewer); how the fluid takes each is in
[FLUID_3D.md](FLUID_3D.md).

`MonitorLayer`: `set_input(MonitorInput)`; `update()` keeps the panel's list of the chosen stage's
port names (`names`, the options of `texture`); `draw()` renders the port once per tick, however
many panels show it, by its kind or by `view`, and nothing while not shown; its views and settings
are in [MONITOR.md](MONITOR.md). It is not a stage: it has no output and no `reset`.

The stages' settings: `reset`, first on every stage's panel; then each stage's own switches (the 3D
motion's and the 3D fluid's `guard`, [FLUID_3D.md](FLUID_3D.md#the-guard); the fluid's `advection`,
`wall` and `slip_time`), then its working size, `width` and `height` (the 3D fluid's `depth` after
them), which each stage declares itself. The motion's is the size its flow runs at, the bridge's the
size it writes its outputs at (its filters follow the velocity's size), the fluid's its simulation
grid (its density is at the size the app allocates). On the fluid layer `blend_mode`. A layer whose
stage runs one component is that component's settings too (`Motion2DLayerSettings` is a
`GradientFlowSettings`, `Motion3DLayerSettings` a `LucasKanadeFlow3DSettings`, the fluid layers' a
`Fluid2DSettings` and a `Fluid3DSettings`), so the component's fields and groups sit on the stage's
panel; the bridge layers, running several bridges, hold them in groups (the 3D bridge's three
subpanels: `readings`, `lowpass` and `bandpass`; `boundary`, `velocity`, `density`, `temperature`;
`body_force`, `acceleration`, `density`, `temperature`, each with its mix). No `draw_mode`,
`visualization`, `stage`, `max_players` or `fps`: the stages draw nothing, read no poses and take
`dt` from their input.

The apps' render is a layer graph of `LayerBase` layers updated by one loop calling `update()`
on each (`apps/deep_flow/render.py`, `apps/hd_trio/render.py`), with a wiring pass before it
(`_wire_flow`). The wiring pass builds a motion and a bridge input per camera and a fluid input
per fluid layer, with one injection per bridge that fluid takes, and sets them; the loop, which
`UPDATE_LAYERS` orders motion, bridge, fluid, stays uniform. Which layers feed the inputs is app
policy: hd_trio wires fixed layers, measures on its centered mask, gives each bridge the centered
foreground as the picture behind its density and temperature, each injection the camera's track
color, and weights the own camera's injection by the player's motion and another's by the
cross-camera similarity times `similarity_scale`; deep_flow picks the layers on
its panel, live, one `Layers` field per picture (`motion_image`, `bridge_image`, `mask`) in its
`inputs` group with `scale_by_motion` (`apps/deep_flow/settings.py`) and reads the fields each
tick. Both apps give each camera two monitors, `monitor_a`
and `monitor_b`, each handed every stage's `ports`; they go into the panel's select lists like
any layer, and two show A beside B.

## Packages

| Package                                                  | Holds                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
|----------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `modules/flow/motion/`                                   | `GradientFlow`; `LucasKanadeFlow`, the x/y solve with `LucasKanadeFlowSettings`; `LucasKanadeFlow3D`, composing it with `DepthVelocity`, with `LucasKanadeFlow3DSettings` (`use_mask`, the gains, `xy`, `z`); `DepthVelocitySettings`; `DepthView`; its document [MOTION.md](MOTION.md)                                                                                                                                                                                                                                          |
| `modules/flow/bridge_2d/`                                | the 2D bridge                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| `modules/flow/bridge_3d/`                                | the 3D bridge; its document [BRIDGE.md](BRIDGE.md)                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| `modules/flow/fluid_2d/`                                 | `Fluid2D`, `Fluid2DSettings` with its own `VelocitySettings`, `DensitySettings`, `TemperatureSettings`, `PressureSettings`; `shaders/`                                                                                                                                                                                                                                                                                                                                                                                           |
| `modules/flow/fluid_3d/`                                 | `Fluid3D`, `Fluid3DSettings` with the box and one group per concern (`TimeSettings`, `CompositeSettings`, `ParticleSettings`, ...); the density's carriers (`DensityCarrier`: `SimulationDensity`, `UpscaledDensity`, `ParticleDensity`), `FieldAdvection`; `shaders/`; its document [FLUID_3D.md](FLUID_3D.md)                                                                                                                                                                                                                  |
| `modules/flow/shaders/`                                  | the shared fragment shaders, `AddBoolean` among them                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| `modules/flow/time_step.py`, `modules/flow/wall_mode.py` | shared                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| `modules/flow/port.py`                                   | `Port`, `Ports`, `TextureKind`, `HasPorts`, `OnDemand`, `prefixed`, `ports_of`                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| `modules/flow/visualization/`                            | the monitors' drawing: `Visualizer` (image, colormap, glyphs and histogram into one FBO), `Strip`, `Legend` with `log_axis.py`, `DrawnSize`, `ALL_CHANNELS`; a volume's `VolumeCube` with `CubeColor`, `VolumeGlyphs` and their `Orbit`, and `Cut`, `VolumeCut` for the strip and the scripts                                                                                                                                                                                                                                    |
| `modules/render/layers/flow/`                            | `motion_layer.py` (`MotionInput`, `MotionOutput`, `MotionLayer`, `frame_interval`), `motion_2d_layer.py`, `motion_3d_layer.py`; `bridge_layer.py` (`BridgeInput`, `BridgeOutput`, `BridgeLayer`), `bridge_2d_layer.py`, `bridge_3d_layer.py`; `fluid_layer.py` (`Injection`, `FluidInput`, `FluidOutput`, `FluidLayer`), `fluid_2d_layer.py`, `fluid_3d_layer.py`; `monitor_layer.py` (`MonitorInput`, `MonitorLayer`, `MonitorLayerSettings`, `MonitorAdvancedSettings`, `FlowStage`, `MonitorView`, `Channel`, `GlyphSpacing`) |

Neither `fluid_2d` nor `fluid_3d` imports the other. The settings group of each fluid is `fluid`.
`motion/` has no `_2d` and `_3d` packages because its 3D flow is its 2D flow plus z:
`LucasKanadeFlow3D` composes `LucasKanadeFlow` and `DepthVelocity`, where the bridges and the
fluids are two implementations. The visualization is shared because it is not a stage: a monitor
draws the port it is given by its channels, whatever the stage: arrows of a 2-channel velocity
are 2D and of a 3- or 4-channel one 3D (`Visualizer.process_glyphs`).
The shader loader finds a shader's source by its `.py` module name beside it
(`modules/gl/shader.py`), so moved shaders need no edit.

## The input's cadence

Every camera frame is to reach the motion at the camera's cadence, the motion measuring one capture per
frame. How a frame gets there, thread by thread, what costs frames and the plan are in
[INPUT_THREADS.md](INPUT_THREADS.md).

## Constraints

- Each pipeline stage is one package and one layer; a 2D and a 3D version of a stage sit side by
  side inside the package. Nothing is grouped by app or by camera.
- The 2D system's measurement stays as on `main` (`GradientFlow`, `bridge_2d`, `Fluid2D`) until
  the 2D optical flow moves to the Lucas–Kanade solve: its velocity is per frame and it has no
  velocity gate. hd_trio measures on the mask itself (`apps/hd_trio/render.py`), so a gate would
  change nothing there **(deduction)**.
- The fluid knows injections, not cameras. The 2D fluid keeps its injections apart until its
  output, one density channel each, so a 2D fluid takes four injections at most, and colors them
  there with each injection's color, as `main` colored its cameras: where two meet the brightness
  is capped at white and the opacity is the stronger one's. hd_trio looks as it did on `main`
  **(deduction)**: the fluid receives the same velocity at the same weight, the same amount per
  channel, the same temperature and the same colors. The 3D fluid takes the density colored by
  the bridge, any number of injections.
- The inputs and outputs are the contract and the base classes hold the two kinds to one shape;
  there are no `Protocol`s for the boundaries. Shared code lives in the base, never in a shared
  concrete class.
- The optical flow classes are GL but no `LayerBase`: the motion layer holds one, and the
  comparison scripts run one on saved frames with no render loop
  (`scripts/depth_velocity_compare.py`).
- The image, the mask and the depth come pushed in the inputs: a GL texture lives on the render
  thread and is not board data. The layers read nothing from the board; the app's wiring pass
  reads the pose frames there for the weights.
- A layer has no change flag: a source layer's `timestamp`, the capture time of the frame it holds, says
  whether it holds a new one, and `None` that it holds none yet: a stage that compares frames over time
  counts only frames with a capture time, and one that only draws may draw black. `MotionInput.timestamp`
  tells the motion layer, and `DepthMaskLayer` reads its depth source's the same way.
- The time step has one transport: the app hands the tick's `dt` in the inputs, and the fluid's
  `TimeStep` smooths it over `fps_smoothing`, or steps fixed at `fixed_fps`. No `fps` or
  `step_fps` field: runtime data stays off the settings.
- A slow frame never gives the fluid more work. Its step is at most 1/20 s (`MAX_STEP`), so a
  slower frame runs the fluid slower than real time and nothing moves further in one step; the
  pressure solve takes its `iterations` at any step, since the grid sets what it needs; the
  viscosity takes its `viscosity_iter` at a 60 fps step and fewer at a shorter one, never more
  (`viscosity_iterations`). Iterations growing with the step would keep a dropped frame rate down:
  at 4 fps, 15 times the work per frame **(deduction)**.
- The stages draw nothing; the monitors draw, only while shown, once per tick. A port's kind says
  how it is drawn when nothing else is asked; the monitor's `view` overrides it.
- The 2D bridge keeps three classes the 3D bridge also has (`DensityBridge`, `TemperatureBridge`,
  `VelocityMagnitude`) as copies, until the 2D bridge is rebuilt.
- The 3D fluid layer runs the 3D fluid only. Comparing the two fluids on one input means both
  systems wired in one app.
- An outside source gates an input through its weight: every injection carries `weight`, 1 by
  default, and the app sets it from whatever it reads (hd_trio: the pose's motion and the
  cross-camera similarity). The fluid never knows why.
- A multiplier on an input is a `gain` (the 2D fluid's `velocity.gain`, `density.gain`,
  `temperature.gain`; the 3D fluid's gains in `input.boundary` and `input.body_force`,
  one per bridge output); a 0 to 1 share in a mix is a `weight`; the density's sinking along gravity
  in the buoyancy is `temperature.sink`, which both fluids read (`Fluid2D._apply_buoyancy`).
- `modules/render/tests/test_flow_systems.py` loads each app preset into its system's stage
  settings and fails on an unknown key, and fails on a field of the flow's settings that no code
  in `modules/flow` or `modules/render/layers/flow` names: such a field is dead **(deduction)**.
- `modules/render/tests/test_stage_layer.py` checks, without GL, that a reset button asks and the
  stage's next update resets once before it runs, and asks nothing after `deallocate`.

## Open

What is open outside the 3D fluid, in order; what touches the fluid is in
[FLUID_3D.md's Plan](FLUID_3D.md#plan).

### 1. The motion's false z at a limb's edge

A limb moving sideways before the body reads false z at its edges: the leading edge carries 43% of a
sweep's z, the trailing edge 12%. The cause is the depth's ramp at the limb's edge. The camera's
temporal filter is off and its confidence does not sharpen the ramp. Accepted for now; the cue not
yet tried is a forward-splat count of the x/y motion (0 = uncovered, above 1 = covered).

### 2. The motion's range

`edge`, `tolerance` and `dead_zone` are in metres and tuned at 0.65–1.5 m. Stereo noise grows with
the square of the distance, so at 2–3.5 m they may need to be in disparity **(deduction)**. This
needs a recording at that distance.

### 3. The 2D fluid follows the 3D one

The 2D fluid adds the injected velocity times dt (`Fluid2D.add_velocity`), as Stam's Stable Fluids
adds a source; but the bridge's velocity is a velocity, in view widths per second like the
fluid's, so it is taken as an acceleration, `velocity.gain` acts as a rate, and the fluid stays
near 0.05 to 0.1 of the motion's speed **(deduction)**, a monitor needing a scale of 20 to 30 to
show it. The 3D fluid takes the boundary by a drag and the body force's acceleration times dt
([FLUID_3D.md](FLUID_3D.md#the-inputs)); the 2D fluid is to follow it: the drag, the two systems,
the one grid.

### 4. The depth velocity's `track`

The depth velocity's memory is named `track` in the code (`DepthVelocity._track`, the monitor's
`z.track`), a word that says nothing on its own; it holds the followed disparity, its rate and its
anchor. To be renamed by what it holds, with the monitor's port.

### 5. hd_trio's density fade from motion

Its wiring pass writes `density.fade_time` every tick from the pose's motion, a settings field
written per tick. A per-tick value on the fluid layer would take it off the settings.

### 6. A soft limit on the motion's z speed

A guard to try on the motion beside its finite guard ([MOTION.md](MOTION.md#the-guard)): z as
measured up to a knee (about 1.5 m/s), compressed above it toward a maximum (about 3 m/s), the speed
gate (6 m/s) behind it; a curve per reading, no time filter, checked first against the spread of z
speeds on standmoves and stillslowfast.

### 7. The depth's mask with another mask source

With `mask_source` RVM, deep_flow makes no `DepthMaskLayer`, so the motion's `depth_mask`, the depth's own
cut that gates z ([RESOLUTION.md](RESOLUTION.md#the-mask-and-the-depth-mask)), has nothing to come from: the
depth's cut has to be made whatever the mask source, cropped by the center mask layer in the image's space,
as RVM's mask is in the crop's. Later; the depth source comes first.
