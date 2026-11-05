# The 2D and 3D flow systems

Two app-independent flow systems, each a pipeline of three stages, motion, bridge and fluid: the
2D flow, a velocity on the image plane, and the 3D flow, a velocity with a component toward the
camera. Either system takes either kind of camera. An app picks a system and wires its cameras
to it.

"Code" is what the files show; **(deduction)** marks what is inferred from them.

## The two systems

| System  | Velocity | Motion                                          | Bridge      | Fluid            | Without depth                                            | With depth                                        |
|---------|----------|-------------------------------------------------|-------------|------------------|----------------------------------------------------------|---------------------------------------------------|
| 2D flow | x, y     | motion on the image plane                       | `bridge_2d` | 2D Navier–Stokes | its normal case                                          | depth is not wired; the system never reads it     |
| 3D flow | x, y, z  | motion on the image plane and toward the camera | `bridge_3d` | 3D Navier–Stokes | z is 0, and so is the fluid's w                          | z from the depth velocity; the fluid takes it as w |

The camera is the other axis. A 2D camera gives an image and a player mask; a 3D camera gives
those and a depth image on the same view. Every combination is a wiring, not a code path:

| Camera | 2D flow                 | 3D flow                              |
|--------|-------------------------|--------------------------------------|
| 2D     | image, mask             | image, mask; z and w are 0           |
| 3D     | image, mask; depth idle | image, mask, depth; z and w measured |

## Terms

| Term                 | Means                                                                                                 | In code                                                            |
|----------------------|-------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------|
| flow                 | the domain: everything from the camera's motion to the fluid                                          | `modules/flow`                                                     |
| motion               | the stage that measures a velocity field from two images: the optical flow                            | `motion/`, `GradientFlow`, `LucasKanadeFlow`, `LucasKanadeFlow3D`  |
| bridge               | the stage that turns the measured velocity into what the fluid injects: a velocity to add, density with its color, temperature, and in 3D onset and spin | `bridge_2d/`, `bridge_3d/`                 |
| injection            | one bridge output as one fluid takes it, at its weight                                                | `Injection`                                                        |
| fluid                | the Navier–Stokes simulation; its density is the show                                                  | `fluid_2d/`, `fluid_3d/`, `Fluid2D`, `Fluid3D`                     |
| volume               | the 3D fluid's z layers: their count, spacing and the composite's look                                | `VolumeSettings`, group `volume`                                   |
| depth                | the camera's depth in metres, on the flow's view                                                      | `DepthView`                                                        |
| depth velocity       | the z of the 3D flow: the depth change along the x/y displacement, in view widths per second          | `DepthVelocity`, `DepthVelocitySettings`, group `depth_velocity`   |
| 2D flow, 3D flow     | a system: a motion layer and a bridge layer per camera, a fluid layer per show output                 | `Motion2DLayer` + `Bridge2DLayer` + `Fluid2DLayer`; `…3D…`         |
| 2D camera, 3D camera | an image and a mask; those and a depth image on the same view                                         | `MotionInput` without / with `depth`                               |
| port                 | a stage's named texture with its kind: image, velocity, scalar, depth scalar; read from the kinds its input and output fields declare, plus its diagnostics | `Port`, `TextureKind`, `ports_of`                       |
| monitor              | a layer that shows one port of one stage, picked on the panel, drawn only while it is shown           | `MonitorLayer`, `monitor_a`, `monitor_b`                           |

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

Each stage takes one input dataclass and gives one output dataclass. Both hold values only:
textures and numbers, never an output; the fluid's input holds its injections as a list of
`Injection`, each values. The app's wiring pass builds each input new every tick, copying into
it what the stage reads from the output of the stage before and from the camera layers, and sets
it on the layer before the render loop. The output holds the stage's textures, allocated once,
and the layer hands it out as `output`; an output is a stage's only way out, and the next stage
reads its fields by name. A texture field declares its kind on the field
(`velocity: Annotated[Texture, TextureKind.VELOCITY]`), in inputs and outputs alike, and a stage's
`ports`, what the monitors show, are read from those declarations (`ports_of`): the output's
fields, then the input's as `input_<field>`, each texture written once. An output's
`diagnostics` add the stage's insides a monitor can show and nothing downstream reads (the 3D
bridge's kinetic trails, the 2D fluid's pressure). A layer's `texture` stays what it draws, read by
the composite. The layers hold no reference to each other, and no input knows an output's type. A field is named with the same noun for the
same kind of thing, in every input.

A texture field holds the `Fbo`, `SwapFbo` or `Tensor` object itself; a swap is its current
buffer, and its `back` is the other one, read inside the pass that writes the current and never
kept, since the two trade places at the next swap. `FlowBase` hands out its `SwapFbo`s for the
same reason. That is what lets an input be built before the render loop: the object named in it
is the one the stage before writes into during the loop, so the next stage reads this tick's
result. Values that are not textures (`timestamp`, `dt`, the weight) come from the app, never from
an output, or they would be the tick before's.

| Stage  | Input `@dataclass`, built by the app each tick                                                                                      | Output `@dataclass`, the stage's textures                                                                |
|--------|-------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------|
| motion | `MotionInput`: `image`, the picture the velocity is measured on by its brightness; `timestamp`, the capture time of the camera frame `image` holds, the frame's identity; `dt`, the render tick's; `depth`, a `DepthView` or `None`; `mask`, or `None` | `MotionOutput`: `velocity` (x, y, z), gated by `mask` with the 3D flow's `use_mask`; z is 0 without depth |
| bridge | `BridgeInput`: `velocity`, from the motion output; `dt`; `image`, or `None`, the picture whose RGB colors the density; `color`, an RGB tint on it, white by default; `mask`, or `None`, which gates the image | `BridgeOutput`: `velocity`; `density`, what the fluid injects: in 2D the amount alone, the trail's speed (R); in 3D RGBA, the amount (the kinetic emission or the speed) times `color`, times the image's RGB where there is one, with the amount as alpha; `temperature`; `onset` or `None`; `spin`, the curl at its `spin` setting, or `None` |
| fluid  | `FluidInput`: `dt`, and `injections`, a list of `Injection`: `velocity`, `density`, `temperature`, `onset`, `spin`, from one bridge output, with `weight`, 0 to 1, the injection's share of the fluid's velocity, density and temperature, as a layer's weight in the light renderer's mix, and `color`, the color the 2D fluid gives the injection's density at its output; the app builds one injection per bridge it wants in that fluid | `FluidOutput`: `density`, the show, which the composite draws as the layer's `texture` |
| monitor | `MonitorInput`: `ports`, every stage's ports keyed by stage; the monitor picks by its own `stage` and `texture` | none: it draws                                                                                           |

The motion layer measures once per camera frame: when `timestamp` differs from the one it last
measured; a source without stamps (`None`) measures every tick. A gap wider than a frame period
is a dropped frame. A camera frame has no other identity: `CameraFrame` and `Capture` carry a
timestamp and no sequence number, because the device's sequence numbers skip on the depth while
every frame arrives (`modules/camera/oak4/OAK4.md`). A source layer keeps the capture time of the
frame it holds (`CaptureImageLayer`, `DepthSourceLayer`, `MaskSourceLayer` and
`ForegroundSourceLayer` from `segmentation.Image.timestamp`, `DepthMaskLayer` from its depth), and
a view (`CenterCamLayer`, `CenterMaskLayer`, `CenterForegroundLayer`, `CenterDepthLayer`) takes its
source layer and forwards the stamp of the frame it drew, so the app reads it off the layer it
takes the texture from (`HasCaptureTime`, `modules/render/layers/layer_base.py`).

`DepthView(texture, view_span, depth_step)` (`modules/flow/motion/depth_view.py`) is the depth on
the flow's view: the texture, the metres one view width spans per metre of depth, and the uv
between two depth samples. The 3D camera's layers make it: `DepthSourceLayer` the texture and its
`sample_size`, `CenterGeometry` the `view_span` from its crop and the camera's horizontal field of
view it takes at construction, and the `depth_step` from the sample size and its crop;
`CenterDepthLayer.depth_view` puts them together.

Two clocks. The motion layer measures between two camera frames, so its interval comes from
`timestamp` (`frame_interval`, `motion_layer.py`, with `dt` as the fallback for a source without
stamps). The bridge and the fluid run per render frame, so they step on `dt`, the tick's measured
length, which the app holds to at most twice the mean tick (`hold_tick`, `time_step.py`): a stall
counts as one long frame and the rest is dropped, as a capture gap is held to twice the frame
period. The fluid's `TimeStep` smooths it over `fps_smoothing`; a raw per-frame dt looks jittery
in the sim **(site fact)**.

A tick without a new camera frame measures nothing: the motion output holds what the last pair
of frames gave, the tick's `dt` adds to the interval the next measurement covers, and the bridge
and the fluid run as on any tick, on the held velocity. A velocity held between samples and
integrated at the render rate is the right reading of it: the motion continues until the next
pair says otherwise, and the stamps keep the magnitude per second right whether a frame took one
tick or two. The bridges are filters in time (the kinetic bridge's trails, hold and reservoir
decay by `decay_weight(dt, smooth)`, its carry advects by velocity × dt), so a held velocity
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

| Class                                       | Role                                                                                                                                                                                  |
|---------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `StageLayer[I, O]` (ABC)                    | `set_input(I)` before the render loop; `output: O`, a `StageOutput`, whose `diagnostics` (keyword only) add to the ports its fields declare; `ports`, the output's then the input's; `reset()`, from rest; `update()` takes a reset the panel's `reset` button asked for, on the render thread, then runs `_update()`. Unbinds the button in `deallocate()` |
| `MotionLayer` (ABC)                         | `StageLayer[MotionInput, MotionOutput]`; `_update()` as the template method: `_measure` on a new frame. Holds the allocate and resize pattern; the flow runs at `settings.width` by `settings.height`. Draws nothing |
| `Motion2DLayer`, `Motion3DLayer`            | the subclasses, each on its optical flow (`GradientFlow`; `LucasKanadeFlow3D`). The 2D flow's velocity is per frame, not per second, and it reads no mask, until the 2D flow moves to the Lucas–Kanade solve |
| `BridgeLayer` (ABC)                         | `StageLayer[BridgeInput, BridgeOutput]`; `_update()`: the density's color (`DensityColor`: the image, or white, times the tint, gated by the mask), `_bridge` the input's velocity into velocity, density and temperature. The bridges' filters run at the velocity's size and their outputs at `settings.width` by `settings.height`; the layer follows both. Draws nothing |
| `Bridge2DLayer`, `Bridge3DLayer`            | the subclasses, each on its bridges (`bridge_2d`; `bridge_3d`, with `method` picking the trail or the kinetic bridge, whose curl is scaled by `onset.spin`)                            |
| `FluidLayer` (ABC)                          | `StageLayer[FluidInput, FluidOutput]`; `_update()`: `_inject` each injection's velocity, density and temperature at its weight, `_step(dt)`; `texture`, the show. Knows injections, not cameras; reads no poses. The stage's reset is the fluid's only one |
| `Fluid2DLayer`, `Fluid3DLayer`              | the subclasses, each on its own fluid. `Fluid2DLayer` keeps its injections apart: injection i's amount goes into channel i of the density, four at most, and the output colors each channel with its injection's color (`DensityColorize`), the brightness capped at white and the opacity the strongest channel's. `Fluid3DLayer` injects RGBA density, onset and spin, the velocity's z as the volume's w (its sign turned: w points away from the viewer), and takes the parallax eye the app steers by `set_eye(x, y)` |
| `MonitorLayer`                              | `set_input(MonitorInput)`; `update()` keeps the panel's list of the chosen stage's port names (`names`, the options of `texture`); `draw()` renders the port once per tick, however many panels show it, by its kind or by `mode` (image, arrows, scalar in two colors by `channel`, or strip: the mean amount of each channel over the last seconds, `Strip`), and nothing while not shown. Its `visualization` group holds the scale, spacing and arrows. Not a stage, since it has no output; its own `reset` button starts the strip over, the one mode with a history |
| `MotionLayerSettings`, `BridgeLayerSettings`, `FluidLayerSettings` | base settings the 2D and 3D settings subclass, each a `StageLayerSettings`: `reset`, then `width` and `height`, the stage's working size, first on every stage's panel. The motion's is the size its flow runs at, the bridge's the size it writes its outputs at (its filters follow the velocity's size), the fluid's its simulation grid (its density is at the size the app allocates). On the fluid layer `blend_mode`. A layer whose stage runs one component is that component's settings too (`Motion2DLayerSettings` is a `GradientFlowSettings`, `Motion3DLayerSettings` a `LucasKanadeFlow3DSettings`, the fluid layers' a `Fluid2DSettings` and a `Fluid3DSettings`), so the component's fields and groups sit on the stage's panel; the bridge layers, running several bridges, hold each as a group (`trail`, `kinetic`, `density`, `temperature`). No `draw_mode`, `visualization`, `stage`, `max_players` or `fps`: the stages draw nothing, read no poses and take `dt` from their input |

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
`inputs` group with `scale_by_motion` (`apps/deep_flow/settings.py`), reads the fields each tick,
and steers the eye toward the tracked head. Both apps give each camera two monitors, `monitor_a`
and `monitor_b`, each handed every stage's `ports`; they go into the panel's select lists like
any layer, and two show A beside B.

## Packages

| Package                         | Holds                                                                                                                                                      |
|---------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `modules/flow/motion/`          | `GradientFlow`; `LucasKanadeFlow`, the x/y solve with `LucasKanadeFlowSettings`; `LucasKanadeFlow3D`, composing it with `DepthVelocity`, with `LucasKanadeFlow3DSettings` (`use_mask`, the gains, `solve`, `depth_velocity`); `DepthVelocitySettings`; `DepthView` |
| `modules/flow/bridge_2d/`       | the 2D bridge                                                                                                                                              |
| `modules/flow/bridge_3d/`       | the 3D bridge                                                                                                                                              |
| `modules/flow/fluid_2d/`        | `Fluid2D`, `Fluid2DSettings` with its own `VelocitySettings`, `DensitySettings`, `TemperatureSettings`, `PressureSettings`; `shaders/`                     |
| `modules/flow/fluid_3d/`        | `Fluid3D`, `Fluid3DSettings` with its own groups and `VolumeSettings`; `shaders/`                                                                          |
| `modules/flow/shaders/`         | the shared fragment shaders, `AddBoolean` among them                                                                                                       |
| `modules/flow/time_step.py`, `modules/flow/boundary_mode.py` | shared                                                                                                                        |
| `modules/flow/visualization/`   | the monitors' drawing: `Visualizer` with its arrow and signed fields, `Strip`                                                                             |
| `modules/render/layers/flow/`   | `port.py` (`Port`, `TextureKind`); `motion_layer.py` (`MotionInput`, `MotionOutput`, `MotionLayer`, `frame_interval`), `motion_2d_layer.py`, `motion_3d_layer.py`; `bridge_layer.py` (`BridgeInput`, `BridgeOutput`, `BridgeLayer`), `bridge_2d_layer.py`, `bridge_3d_layer.py`; `fluid_layer.py` (`Injection`, `FluidInput`, `FluidOutput`, `FluidLayer`), `fluid_2d_layer.py`, `fluid_3d_layer.py`; `monitor_layer.py` (`MonitorInput`, `MonitorLayer`, `FlowStage`, `MonitorMode`, `Channel`) |

Neither `fluid_2d` nor `fluid_3d` imports the other. The settings group of each fluid is `fluid`.
`motion/` has no `_2d` and `_3d` packages because its 3D flow is its 2D flow plus z:
`LucasKanadeFlow3D` composes `LucasKanadeFlow` and `DepthVelocity`, where the bridges and the
fluids are two implementations. The visualization is shared because it is not a stage: a monitor
draws the port it is given, a 2-channel velocity as a 2D one and a 3- or 4-channel one as a 3D
one (`VelocityField.update`), a scalar in two colors, any texture as a strip.
The shader loader finds a shader's source by its `.py` module name beside it
(`modules/gl/shader.py`), so moved shaders need no edit.

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
  reads the pose frames there for the weights and the eye.
- A layer has no change flag: a source layer's `timestamp`, the capture time of the frame it
  holds, says whether it holds a new one. `MotionInput.timestamp` tells the motion layer, and
  `DepthMaskLayer` reads its depth source's the same way.
- The time step has one transport: the app hands the tick's `dt` in the inputs, and the fluid's
  `TimeStep` smooths it over `fps_smoothing`, or steps fixed at `fixed_fps`. No `fps` or
  `step_fps` field: runtime data stays off the settings.
- The stages draw nothing; the monitors draw, only while shown, once per tick. A port's kind says
  how it is drawn when nothing else is asked; the monitor's `mode` overrides it.
- The 2D bridge keeps three classes the 3D bridge also has (`DensityBridge`, `TemperatureBridge`,
  `VelocityMagnitude`) as copies, until the 2D bridge is rebuilt.
- The 3D fluid layer runs the 3D fluid only. Comparing the two fluids on one input means both
  systems wired in one app.
- An outside source gates an input through its weight: every injection carries `weight`, 1 by
  default, and the app sets it from whatever it reads (hd_trio: the pose's motion and the
  cross-camera similarity). The fluid never knows why.
- A multiplier on an input is a `gain` (the fluids' `velocity.gain`, `density.gain`,
  `temperature.gain`); a 0 to 1 share in a mix is a `weight`; the density's sinking along gravity
  in the buoyancy is `temperature.sink`, which both fluids read (`Fluid2D._apply_buoyancy`).
- `modules/render/tests/test_flow_systems.py` loads each app preset into its system's stage
  settings and fails on an unknown key, and fails on a field of the flow's settings that no code
  in `modules/flow` or `modules/render/layers/flow` names: such a field is dead **(deduction)**.
- `modules/render/tests/test_stage_layer.py` checks, without GL, that a reset button asks and the
  stage's next update resets once before it runs, and asks nothing after `deallocate`.

## Open

- hd_trio's density fade from motion. Its wiring pass writes `density.fade_time` every tick from
  the pose's motion, a settings field written per tick. A per-tick value on the fluid layer, as the
  eye is, would take it off the settings.
