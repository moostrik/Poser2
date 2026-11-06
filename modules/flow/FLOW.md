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
| bridge               | the stage that turns the measured velocity into what the fluid injects: a velocity to add, density with its color, temperature, and in 3D a second velocity | `bridge_2d/`, `bridge_3d/`                 |
| impulse              | a velocity change, Δv in widths per second, where a move starts or stops, read by the 3D bridge beside its trail: start, bow, stop, wake | `VelocityImpulseBridge`, `Impulse`, group `impulse` |
| region               | the pixels around one, which tell a part of the body changing speed from a limb arriving or leaving | `start_region_size`, `stop_region_size`, `*_region_weight`         |
| injection            | one bridge output as one fluid takes it, at its weight                                                | `Injection`                                                        |
| fluid                | the Navier–Stokes simulation; its density is the show                                                  | `fluid_2d/`, `fluid_3d/`, `Fluid2D`, `Fluid3D`                     |
| volume               | the 3D fluid's z layers: their count, spacing and the composite's look                                | `VolumeSettings`, group `volume`                                   |
| depth                | the camera's depth in metres, on the flow's view                                                      | `DepthView`                                                        |
| depth velocity       | the z of the 3D flow: the depth change along the x/y displacement, in view widths per second          | `DepthVelocity`, `DepthVelocitySettings`, group `depth_velocity`   |
| 2D flow, 3D flow     | a system: a motion layer and a bridge layer per camera, a fluid layer per show output                 | `Motion2DLayer` + `Bridge2DLayer` + `Fluid2DLayer`; `…3D…`         |
| 2D camera, 3D camera | an image and a mask; those and a depth image on the same view                                         | `MotionInput` without / with `depth`                               |
| port                 | a stage's named texture with its kind: image, velocity, scalar; its input and output fields declare theirs, every component its own | `Port`, `TextureKind`, `HasPorts`, `ports_of`, in `modules/flow` |
| monitor              | a layer that shows one port of one stage, picked on the panel, drawn only while it is shown           | `MonitorLayer`, `monitor_a`, `monitor_b`                           |
| glyphs               | a monitor's view of a field as marks on a grid: arrows for a velocity, dots for a scalar             | `MonitorView.GLYPHS`, `glyph_spacing`, `GlyphField`, `VolumeGlyphs` |
| cut                  | how a monitor lays out a volume: in 3D as stacked layers or a fog, or flat, one layer or a projection | `Cut`, `advanced.cut`, `VolumeCut`                                 |

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
(`velocity: Annotated[Texture, TextureKind.VELOCITY]`), in inputs and outputs alike (`ports_of`),
and every component that owns textures names them once with their kinds in its `ports`
(`HasPorts`): `FlowBase` its `input` and `output`, a subclass its others. A stage names its
components once (`_components`), by their settings group, and its `ports`, what the monitors
show, is one list with the data running bottom to top, the prefix its group: `output.*` on top,
each component's from the last the stage runs to the first (`z.track`, `xy.solve`,
`impulse.fast`, `trail.output`), `input.*` at the bottom. A texture not allocated is left out, so a part the stage does not run gives none.
Not by reflection over attributes: it would find
the FBOs, not their kinds, and a swap's back buffer would be a port too. A layer's `texture` stays
what it draws, read by the composite. The layers hold no reference to each other, and no input knows an output's type. A field is named with the same noun for the
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
| bridge | `BridgeInput`: `velocity`, from the motion output; `dt`; `image`, or `None`, the picture whose RGB colors the density; `color`, an RGB tint on it, white by default; `mask`, or `None`, which gates the image | `BridgeOutput`: `velocity`; `density`, what the fluid injects: in 2D the amount alone, the trail's speed (R); in 3D RGBA, the amount (the trail's speed) times `color`, times the image's RGB where there is one, with the amount as alpha; `temperature`; `onset`, a second velocity, or `None` |
| fluid  | `FluidInput`: `dt`, and `injections`, a list of `Injection`: `velocity`, `density`, `temperature`, `onset` from one bridge output, and `spin`, which no bridge gives, with `weight`, 0 to 1, the injection's share of the fluid's velocity, density and temperature, as a layer's weight in the light renderer's mix, and `color`, the color the 2D fluid gives the injection's density at its output; the app builds one injection per bridge it wants in that fluid | `FluidOutput`: `density`, the show, which the composite draws as the layer's `texture` |
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
tick or two. The bridges are filters in time (their trails decay by `decay_weight(dt, smooth)`,
the velocity trail's advect carries it by velocity × dt), so a held velocity
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
| `StageLayer[I, O]` (ABC)                    | `set_input(I)` before the render loop; `output: O`, a `StageOutput`, whose fields with a kind are its ports; `_components()`, the parts by group name; `ports`, input, parts, output; `reset()`, from rest; `update()` takes a reset the panel's `reset` button asked for, on the render thread, then runs `_update()`. Unbinds the button in `deallocate()` |
| `MotionLayer` (ABC)                         | `StageLayer[MotionInput, MotionOutput]`; `_update()` as the template method: `_measure` on a new frame. Holds the allocate and resize pattern; the flow runs at `settings.width` by `settings.height`. Draws nothing |
| `Motion2DLayer`, `Motion3DLayer`            | the subclasses, each on its optical flow (`GradientFlow`; `LucasKanadeFlow3D`). The 2D flow's velocity is per frame, not per second, and it reads no mask, until the 2D flow moves to the Lucas–Kanade solve |
| `BridgeLayer` (ABC)                         | `StageLayer[BridgeInput, BridgeOutput]`; `_update()`: the density's color (`DensityColor`: the image, or white, times the tint, gated by the mask), `_bridge` the input's velocity into velocity, density and temperature. The bridges' filters run at the velocity's size and their outputs at `settings.width` by `settings.height`; the layer follows both. Draws nothing |
| `Bridge2DLayer`, `Bridge3DLayer`            | the subclasses, each on its bridges (`bridge_2d`; `bridge_3d`, the velocity trail and beside it the velocity impulse bridge, [The impulses](#the-impulses), run while a monitor shows one of its ports, through their `prepare`, and not yet in the output) |
| `FluidLayer` (ABC)                          | `StageLayer[FluidInput, FluidOutput]`; `_update()`: `_inject` each injection's velocity, density and temperature at its weight, `_step(dt)`; `texture`, the show. Knows injections, not cameras; reads no poses. The stage's reset is the fluid's only one |
| `Fluid2DLayer`, `Fluid3DLayer`              | the subclasses, each on its own fluid. `Fluid2DLayer` keeps its injections apart: injection i's amount goes into channel i of the density, four at most, and the output colors each channel with its injection's color (`DensityColorize`), the brightness capped at white and the opacity the strongest channel's. `Fluid3DLayer` injects RGBA density, the onset and a spin where one is given, the velocity's z as the volume's w (its sign turned: w points away from the viewer), and takes the parallax eye the app steers by `set_eye(x, y)` |
| `MonitorLayer`                              | `set_input(MonitorInput)`; `update()` keeps the panel's list of the chosen stage's port names (`names`, the options of `texture`); `draw()` renders the port once per tick, however many panels show it, by its kind or by `view`, and nothing while not shown; its views and settings are in [The monitor](#the-monitor). Not a stage, since it has no output and no `reset` |
| `MotionLayerSettings`, `BridgeLayerSettings`, `FluidLayerSettings` | base settings the 2D and 3D settings subclass, each a `StageLayerSettings`: `reset`, then `width` and `height`, the stage's working size, first on every stage's panel. The motion's is the size its flow runs at, the bridge's the size it writes its outputs at (its filters follow the velocity's size), the fluid's its simulation grid (its density is at the size the app allocates). On the fluid layer `blend_mode`. A layer whose stage runs one component is that component's settings too (`Motion2DLayerSettings` is a `GradientFlowSettings`, `Motion3DLayerSettings` a `LucasKanadeFlow3DSettings`, the fluid layers' a `Fluid2DSettings` and a `Fluid3DSettings`), so the component's fields and groups sit on the stage's panel; the bridge layers, running several bridges, hold each as a group (`trail`, `impulse`, `density`, `temperature`). No `draw_mode`, `visualization`, `stage`, `max_players` or `fps`: the stages draw nothing, read no poses and take `dt` from their input |

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

## The 3D bridge

The bridge reads two things from a movement: the motion itself, a push in its direction that is
stronger for fast than for slow and coasts after a stop; and its impulses, the velocity changes
where a move starts and where it stops. The motion wants a long, soft reading, the impulses a
short, sharp one, so they are two bridges side by side on the same velocity (`bridge_3d/`):

| Bridge                  | Reads                                                         | Gives                                                          | Panel group |
|-------------------------|---------------------------------------------------------------|----------------------------------------------------------------|-------------|
| `VelocityTrailBridge`   | the flow's velocity through one trail: advect, attack/smooth, blur | `smooth_velocity`, the push; its speed is the density amount  | `trail`     |
| `VelocityImpulseBridge` | the flow's velocity through a fast and a slow trail, and their regions | `start`, `bow`, `stop`, `wake`, each also as `<impulse>_velocity` | `impulse`   |

The fluid makes its own swirl from what it is given, so the bridge gives no curl of its own.

### The velocity trail

One unit-gain Ema per pixel: a rising x/y speed follows over `attack`, a falling one and all of z
over `smooth`, so a flick reaches the fluid at full strength and a stop coasts. `advect` moves the
trail with its own motion before each step, so the average follows a moving limb instead of a
fixed pixel. `blur_radius` and `blur_steps` spread it in space; the ring it leaves around the
body is part of the look **(site fact)**.

### The impulses

The difference of the fast and the slow trail along the motion's direction rises where a move
starts and falls where it stops; it lasts about `slow_smooth`, as the slow trail catches up. It is
divided by the trails' step peak (`step_peak`, 0.58 at 25 and 150 ms), so a limb at speed v that
starts or stops at once reads v: the impulses are in the motion's own strength, image widths per
second, a velocity change Δv, an impulse in the physics sense (a change of momentum per unit of
mass), and a gradual start or stop reads less, by how much of its speed changes within the trails'
times. A pixel cannot tell a move starting from a limb arriving at it, nor a move stopping from a limb
leaving it: its speed rises or falls either way. The region around it can, so the region splits
each pixel's rise and fall in two, by the region weight (`start_region_weight`,
`stop_region_weight`; 0 gives all of it to the start and the stop):

| Impulse | Is                                              | Along                 |
|--------|--------------------------------------------------|-----------------------|
| start  | the share of a rise its region shares: a part of the body speeding up | the fast trail's direction |
| bow    | the rest of the rise: a limb arriving, the wave in front of it        | the fast trail's direction |
| stop   | the share of a fall its region shares: a part of the body slowing down | the slow trail's direction, the motion it ended |
| wake   | the rest of the fall: a limb leaving, behind it                        | the slow trail's direction |

Start + bow is a pixel's whole rise, stop + wake its whole fall. What a region shares:

- A rise reads the region's mean velocity over its moving pixels (those the motion gives a
  velocity), taken from the flow and then smoothed per region by the fast and the slow trail.
  A limb arriving adds moving pixels at its own speed and leaves that mean as it was; a limb
  speeding up raises it; jitter on a still body cancels in it. Two other readings fail: over all
  of a region's pixels an arriving limb raises the mean like a speedup, and taken of the two
  trails the slow trail's lag at the pixels a limb arrives at does too.
- A fall reads the region's mean speed of the two trails, which jitter never lowers and a turning
  limb keeps, where a mean velocity would shrink and read as a stop. The trails lag at the pixels
  a limb leaves, so a leaving limb lowers it too: part of a wake reads as a stop.

The impulses read x, y and z alike. The bridge filters in time only: noise within a frame is the
motion's to filter, and the trail's blur is a look. An impulse takes no hold: it lasts about
`slow_smooth` by itself, and what a stop injects is the speed it ended; a hold lifted steady motion
as much as the stops on the harness (below). No blur and no advect: the impulses stay sharp. The
impulse bridge runs while a monitor shows one of its ports (`Bridge3DLayer`); nothing it makes
reaches the fluid yet.

### Measured

`scripts/bridge_compare.py` on the standmoves recording (four moves, starts and stops), frames
captured on 2026-11-05 with deep_flow's preset flow (levels 5, threshold 0.5), each bridge stepped
once per flow frame. Per frame, the mean over the body's pixels that move in the loop; noise is the
still part's p99 per pixel; stop ÷ fast is the stop window's peak over steady fast motion.

| Reading                                             | Noise | Stop ÷ fast | Start ÷ fast | Coast or burst, s | Ring |
|-----------------------------------------------------|-------|-------------|--------------|-------------------|------|
| trail velocity (smooth 1 s, attack 0.001 s, advect 1, blur 4.43) | 0.018 | 0.69 | 1.16     | 0.87              | 0.19 |
| trail velocity, advect 0                            | 0.270 | 1.07        | 0.84         | 1.84              | 0.40 |
| trail velocity, blur 0                              | 0.017 | 0.68        | 1.09         | 0.84              | 0.16 |
| stop, region weight 0                               | 0.025 | 1.18        | 0.40         | 0.47              | 0.33 |
| stop, region weight 1                               | 0.009 | 1.64        | 0.48         | 0.38              | 0.35 |
| stop, region weight 1, x/y only                     | 0.000 | 2.26        | 0.93         | 0.27              | 0.28 |
| start, region weight 0                              | 0.104 | 0.12        | 1.46         |                   |      |
| start, region weight 0, x/y only                    | 0.001 | 0.05        | 1.89         |                   |      |

The impulse rows are measured before the step peak scaling; the ratios hold, the impulse's level is
1 / 0.58 = 1.72 times higher now. The stop rows hold 0.1 s. Ring is the share 4–24 px outside the mask, in steady fast motion for the
trail and in the stop windows for the impulses; x/y only is z left out of the impulses. Without
its region, a start rises along the front of every moving limb, 0.125 in steady fast motion against
0.18 at real starts. The start's region reading over moving pixels and the bow and wake are not
measured on the harness yet.

## Packages

| Package                         | Holds                                                                                                                                                      |
|---------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `modules/flow/motion/`          | `GradientFlow`; `LucasKanadeFlow`, the x/y solve with `LucasKanadeFlowSettings`; `LucasKanadeFlow3D`, composing it with `DepthVelocity`, with `LucasKanadeFlow3DSettings` (`use_mask`, the gains, `xy`, `z`); `DepthVelocitySettings`; `DepthView` |
| `modules/flow/bridge_2d/`       | the 2D bridge                                                                                                                                              |
| `modules/flow/bridge_3d/`       | the 3D bridge                                                                                                                                              |
| `modules/flow/fluid_2d/`        | `Fluid2D`, `Fluid2DSettings` with its own `VelocitySettings`, `DensitySettings`, `TemperatureSettings`, `PressureSettings`; `shaders/`                     |
| `modules/flow/fluid_3d/`        | `Fluid3D`, `Fluid3DSettings` with its own groups and `VolumeSettings`; `shaders/`                                                                          |
| `modules/flow/shaders/`         | the shared fragment shaders, `AddBoolean` among them                                                                                                       |
| `modules/flow/time_step.py`, `modules/flow/boundary_mode.py` | shared                                                                                                                        |
| `modules/flow/port.py`          | `Port`, `Ports`, `TextureKind`, `HasPorts`, `OnDemand`, `prefixed`, `ports_of`                                                                             |
| `modules/flow/visualization/`   | the monitors' drawing: `Visualizer` (image, colormap, glyphs and histogram into one FBO), `Strip`, `Legend` with `log_axis.py`, `DrawnSize`, `ALL_CHANNELS`; a volume's `Cut`, `VolumeCut`, `VolumeCube`, `VolumeGlyphs`, `VolumeFog` and their `Orbit` |
| `modules/render/layers/flow/`   | `motion_layer.py` (`MotionInput`, `MotionOutput`, `MotionLayer`, `frame_interval`), `motion_2d_layer.py`, `motion_3d_layer.py`; `bridge_layer.py` (`BridgeInput`, `BridgeOutput`, `BridgeLayer`), `bridge_2d_layer.py`, `bridge_3d_layer.py`; `fluid_layer.py` (`Injection`, `FluidInput`, `FluidOutput`, `FluidLayer`), `fluid_2d_layer.py`, `fluid_3d_layer.py`; `monitor_layer.py` (`MonitorInput`, `MonitorLayer`, `MonitorLayerSettings`, `MonitorAdvancedSettings`, `FlowStage`, `MonitorView`, `Channel`, `GlyphSpacing`) |

Neither `fluid_2d` nor `fluid_3d` imports the other. The settings group of each fluid is `fluid`.
`motion/` has no `_2d` and `_3d` packages because its 3D flow is its 2D flow plus z:
`LucasKanadeFlow3D` composes `LucasKanadeFlow` and `DepthVelocity`, where the bridges and the
fluids are two implementations. The visualization is shared because it is not a stage: a monitor
draws the port it is given by its channels, whatever the stage: arrows of a 2-channel velocity
are 2D and of a 3- or 4-channel one 3D (`Visualizer.process_glyphs`).
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
  how it is drawn when nothing else is asked; the monitor's `view` overrides it.
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

## The monitor

The monitor's panel group (`MonitorLayerSettings`) says what is drawn; its `advanced` group
(`MonitorAdvancedSettings`) says how. Neither has row breaks of its own; the panel wraps them.

| Setting                  | Values                                                      | Means                                       |
|--------------------------|-------------------------------------------------------------|---------------------------------------------|
| `stage`                  | `MOTION`, `BRIDGE`, `FLUID`                                 | the stage whose port is shown               |
| `texture`                | the stage's port names                                      | the port shown; a name the stage no longer has shows its first output |
| `view`                   | `AUTO`, `IMAGE`, `COLORMAP`, `GLYPHS`, `HISTOGRAM`, `STRIP` | how the channels are drawn                  |
| `camera`                 | read only                                                   | a volume's view: yaw, pitch, zoom, the layer |
| `advanced.channel`       | `X`, `Y`, `Z`, `W`, `ALL` (default)                         | which of the texture's channels are drawn   |
| `advanced.cut`           | `CUBE` (default), `FOG`, `SLICE`, `FRONT`, `TOP`, `SIDE`    | how a volume is laid out; not read for a 2D port |
| `advanced.scale`         | 0.01 to 100, a log slider                                   | multiplier on the values, in every view     |
| `advanced.glyph_spacing` | 4, 8, 16, 32, 64, 128 (`GlyphSpacing`)                      | distance between glyphs (pixels)            |
| `advanced.legend`        | on (default), off                                           | the log axis, or the colormap's key         |
| `advanced.blend_mode`    | a `Style.BlendMode`                                         | how the monitor is drawn over the panel     |

An arrow reaches one spacing at speed 1 and at most two (the shader rolls its length off toward
2), and its lines are spacing / 20 wide (`ARROW_WIDTH`), so the arrows keep their proportions at
every spacing. A scalar's dot is the colormap's color of its value, its radius half a spacing
times `tanh` of its amount, so small values draw small dots.

The strip starts over when `stage`, `texture` or `channel` changes, or when `view` turns to
`STRIP`, so it never shows another texture's history and needs no reset button.

| View     | `ALL`                                                              | one channel                                   |
|----------|--------------------------------------------------------------------|-----------------------------------------------|
| `IMAGE`  | the texture as stored, times `scale`                               | that channel as grey, times `scale`           |
| `COLORMAP` | x, y, z as red, green, blue around mid grey: grey is 0             | `ALL` with the other channels 0: x from grey toward red above 0 and toward cyan below, y green, z blue; w, which has no color in `ALL`, as grey |
| `GLYPHS` | a velocity's arrow of x and y, with the ring for z on a 3- or 4-channel texture (channel not read); a scalar's dot of x | a scalar's dot of that channel |
| `STRIP`  | four lanes, x, y, z, w: the frame-wide mean of each channel's amount (its absolute value; a signed mean cancels left against right), on a log axis | one lane, full height |
| `HISTOGRAM` | four lanes, x, y, z, w: how many values fall where from −1 to 1 after `scale`, 0 in the middle, outside in the end bins | one lane, full height |

`AUTO` is `IMAGE` for an image port, `GLYPHS` for a velocity, `COLORMAP` for a scalar, the three
kinds of `TextureKind`.

- A channel the texture does not have reads as sampled in every view: a one-channel texture
  repeats x in y and z (its grey swizzle, `modules/gl/texture.py`), a two-channel one has z 0, and
  a missing w is 1.
- The histogram's height is the log of a bin's count against the log of the samples, since most of
  a flow is still and its zero bin would flatten every other. It counts 256 bins on the GPU
  (`HistogramCount`, atomic adds) over the texture's own grid, capped at 512 × 512 samples
  (`histogram_grid`), so the 512 × 288 flow is counted whole, a 4k texture costs no more than the
  cap, and nothing is read back. The strip's mean reads the texture the same way (`summary_grid`).
- The monitor renders at the size it is drawn at (`DrawnSize`): the viewport the app fits it into,
  the largest of the tick before, grown at once when a larger draw turns up, so a smaller layout
  takes over a tick later. No setting: the size the app allocates a layer at (4k) is not the
  monitor's. Its legend draws at that size too, so its text keeps its pixel size and its lines sit
  on the bars; the strip's 540 columns are its history, 9 s at 60 ticks, and its height follows.
  The GPU times are in [Volumes](#volumes).
- One `scale` serves every view, since each monitor has its own view and its own scale. z takes
  `scale` like x and y, since the motion's `z_gain` already balances it.
- The strip's height is logarithmic over three decades, `(log10(mean × scale) + 3) / 3`: a frame's
  mean amount spans decades between a still player and a fast move, so a linear lane clips the one
  or loses the other. `scale` slides the window; a bar a third taller is ten times the amount.
- With `legend` on, every view has a title at the top center, the stage and the port shown
  (`motion / output.velocity`, a slash since the font atlas holds ASCII), and the colormap's key
  sits under it. The strip and the histogram show their log axis over the plot
  (`Legend`): the lane names, a 2 px grey line at each round decade labeled with what it reads
  (the strip's mean amount, 0.001, 0.01, 0.1, 1; the histogram's share of the samples, 0.01%,
  0.1%, 1%, 10%), and the histogram's 0 and its ±1/`scale` limits. `COLORMAP` shows a color key
  instead, in the top left: a small bar per channel shown (x, y, z on `ALL`) from −1/`scale`
  through 0 to +1/`scale` in its colors, since the map is read channel by channel. The legend is drawn into its own FBO only when the view, `channel` or `scale`
  changes and blitted every draw, since `Text` builds and uploads each string's glyphs per draw;
  the lines and labels come from `log_axis.py`, tested without GL in
  `modules/flow/tests/test_log_axis.py`.
- `Visualizer` draws into one FBO: `ChannelView` (image and colormap, the channel and the view as
  uniforms), `GlyphField`, and `HistogramCount` with `HistogramDraw`; `Strip` keeps its
  own, and `Legend` its `LogGrid` and `ColormapKey`.
  `scripts/depth_velocity_compare.py` draws through the same `Visualizer`.
- There is no hue-by-direction map: it is 2D only, on black, with z unread, where `COLORMAP` on
  `ALL` shows direction as a mix of red and green around grey, and z in blue.

### Volumes

A port's texture may be a volume, always a `Texture3D` (`Port.volume`), so each view is one
shader. The 3D fluid lists its volumes as ports: `velocity`, `density`, `temperature`,
`pressure`, `divergence`, `vorticity`, `vorticity_force`, `buoyancy`, `obstacle`, beside the 2D
`obstacle_source`. Each carries its box's depth (`Port.extent`, `Fluid3D._z_extent`); the
monitor draws the box as deep as it is wide (`BOX_DEPTH`), since the simulated slab, 16 layers at
spacing 4 on a 448 wide grid, is 0.14 of its width and reads as flat. A flat cut renders at the
part of the panel it fills in its proportions, so glyphs keep their spacing. Every view samples a volume at (u, v, w) from 0 to 1 through the
box, so the density's `layers` and the simulation's `sim_layers` need not match.

- The density, a 2D array at the output size (2160 × 3840 × `layers` of RGBA16F), is averaged
  down into a `Texture3D` of the simulation's width and height and its `layers`
  (`reduce_density.comp`, the mean of each block), so no view shimmers or reads the whole
  gigabyte. The fluid marks it stale every step (`OnDemand`) and the first view that asks this
  tick (`Port.prepare`) reduces it: two monitors showing it cost one reduction, none costs
  nothing.
- `cut` lays a volume out. `CUBE` and `FOG` are 3D, on the orbit camera; the others are flat, a
  2D texture (`VolumeCut`) every 2D view draws as it draws any texture:

  | `cut`    | Means                                                                       |
  |----------|-----------------------------------------------------------------------------|
  | `CUBE`   | the layers stacked in the box, one instanced draw from the far side to the near one, partly transparent; the selected layer opaque and pulled out, a gap on both sides fading over the next layers; the box's edges and axes |
  | `FOG`    | the volume drawn continuously, a ray per pixel marched front to back, each sample colored as `IMAGE` or `COLORMAP` colors it and its amount its opacity; the box's edges |
  | `SLICE`  | the selected layer                                                          |
  | `FRONT`, `TOP`, `SIDE` | the projection along z, y or x: the max for an image port, the mean for a velocity or a scalar, whose signs would cancel in a max; in the box's proportions, the top width × depth with the front at the bottom, the side depth × height with the front on the right |

- `IMAGE` and `COLORMAP` draw the cube or the fog; `GLYPHS` draws in 3D on either: lit arrows of
  a small mesh per grid point, so one pointing at the camera still shows, or a scalar's dots as
  point sprites, both with the depth test (`Fbo(depth=True)`), at most 96 across (`glyph_grid`).
  The stacked layers take no depth test, since they blend far to near. Every 3D view draws the box's
  edges and, from its first corner (x 0, y 0, the front layer), its axes 3 px wide in the
  colormap's colors at 1 (`CubeAxes`): x red along the width, y green up, z blue into the depth.
  With `legend` on, each axis's letter sits past its end (`axis_labels`, the legend redrawn when
  the camera moves). The strip reads a volume's
  front mean, the volume's mean; the histogram counts a grid through every layer, at most
  128 × 128 across (`histogram_volume_grid`). A view with no 3D draw of its own on a 3D cut takes
  the slice.
- The camera and the selected layer are the monitor's state, not settings, and `camera` shows
  them. In deep_flow a drag that starts anywhere in the main window goes to every monitor, so
  A and B stay on one view: the left button turns (a drag across the window 180°), the right
  zooms, scroll steps one of the show's density `layers` (away from the user is further back,
  a volume with fewer layers showing its nearest), a middle click without a move resets to the
  front view at the middle layer. `Orbit` frames the box by the narrower field of view, so it
  fits a 9:16 panel.

`scripts/monitor_harness.py` checks the monitor without an app or a camera: a hidden GL window,
known textures and volumes with values numpy computes (a cut against numpy's layer, max and mean,
the reduction against the block mean, the cube's and the fog's near side from both sides of the
box), pictures in `temp/monitor_harness`, and the GPU times, measured on the GPU's timer queries:

| Work                                   | At                                   | GPU (ms) |
|----------------------------------------|--------------------------------------|----------|
| image, colormap                        | 540 × 960 / 2160 × 3840              | 0.01 / 0.03 |
| glyphs, flat                           | 540 × 960 / 2160 × 3840, spacing 32  | 0.18 / 2.4 |
| histogram                              | 540 × 960 / 2160 × 3840              | 0.10 / 0.12 |
| strip                                  | 540 × 960                            | 0.05     |
| the density's reduction                | 2160 × 3840 × 16 to 448 × 800 × 16   | 1.1      |
| slice, front, top, side                | a 448 × 800 × 16 volume              | 0.01, 0.08, 0.61, 0.36 |
| cube, 3D arrows, fog                   | a 448 × 800 × 16 volume at 540 × 960 | 0.04, 0.05, 0.05 |

## Open

- The 3D bridge's finish, in this order; each step its tests, then the app:
  4. Measured on the harness, the frames captured again with the preset's flow: the GPU time of the
     trail and the impulse bridge per configuration, each alone; the best value of every setting by
     the one results table; the stop's region on a limb's back edge, built over moving pixels as the
     start's only where more than about a tenth of it reads as stop; the numbers and the advice here.
  5. The mix in `Bridge3DLayer`: `velocity` the trail, `impulse` the start, bow, stop and wake
     velocities, `density` and `temperature` the trail's speed and the four, each source at a weight
     0 to 1, the trail 1 and the rest 0 by default. No gain in the bridge: its sources share one
     unit, the motion's gains balance the axes and the fluid's gains set the strength. The trail and
     the impulse stay two velocity outputs, as the fluid takes them two ways (below).
  6. The 3D fluid's `input_x` and `input_y` go: the motion's `x_gain` and `y_gain` balance the axes.
     For the fluid refactor, as the fluid's own settings.
- For the fluid refactor: no bridge gives a spin any more, so `Injection.spin`, `Fluid3D`'s spin
  input (`set_spin`, `vorticity_force_3d`'s `spin`) and `Fluid3DLayer`'s use of it have no source.
- Whether the density's reduced view should also feed the fluid's own composite at a lower
  quality setting.
- How a fluid takes what the bridge gives, the base of the fluid refactor. Both fluids add the
  injected velocity times dt (`Fluid2D.add_velocity`, `Fluid3D.add_velocity`), as Stam's Stable
  Fluids adds a source; but the bridge's velocity is a velocity, in view widths per second like
  the fluid's, so it is taken as an acceleration, `velocity.gain` acts as a rate, and the fluid
  stays near 0.05 to 0.1 of the motion's speed **(deduction)**, a monitor needing a scale of 20 to
  30 to show it. In Navier–Stokes the external force term is an acceleration per unit mass, as
  gravity and buoyancy enter. The rule, by what each input is:

  | Input is           | The fluid                                                            | From the bridge                  |
  |--------------------|----------------------------------------------------------------------|----------------------------------|
  | a body's velocity  | takes it where the body is, a moving boundary: set, or pulled toward it at a share from a time constant, never past it | the trail's push   |
  | an impulse, Δv     | adds it once in total, without dt                                    | the start, bow, stop and wake    |
  | an acceleration    | adds it times dt                                                     | buoyancy, the spin's vorticity force |
  | a source           | adds it times dt, an amount per second                               | density, temperature             |

  The moving boundary is the physical model of a body in a fluid: the fluid touching it moves with
  it, and coasts on its own inertia once it stops. An impulse is added once: it lasts about
  `slow_smooth` over many ticks, so added directly each tick it counts once per tick, and a
  velocity added without dt every tick runs away while a limb keeps moving, 60 times its speed in
  a second; spread over its ticks, its rate (the change over its duration) times dt adds about its
  Δv in all **(deduction)**. A toggle between the force (as now) and these keeps today's look
  comparable.
- hd_trio's density fade from motion. Its wiring pass writes `density.fade_time` every tick from
  the pose's motion, a settings field written per tick. A per-tick value on the fluid layer, as the
  eye is, would take it off the settings.
- Unused channels written 0 in the whole flow module. The 3D motion's and the 3D bridge's
  velocities write 0 in their unused w (`compose_velocity.frag`, `velocity_trail.frag`,
  `velocity_impulse_direction.frag`), so a monitor shows w as
  nothing rather than a full strip lane and a histogram spike at 1 × `scale`. The rest of
  `modules/flow` still writes 1.0 into unused channels, an alpha left from when the monitors
  blitted textures as they are: among others `gradient_flow_shader.frag`,
  `lucas_kanade_shader.frag`, `displacement_threshold.frag`, the impulse scalars and
  `magnitude_shader.frag`. It matters only where the target texture has that channel (an R16F
  target drops it). Each wants its readers checked first, as the velocities' were: there the
  trails, the impulse shaders, `VelocityMagnitude` and `inject_flow_3d.comp` read xyz only,
  `trail_advect.frag` carries the vec4 along unread, and `Fluid3D.set_velocity` has no caller.
  The visualization's own shaders keep alpha 1, since they draw for the screen.
