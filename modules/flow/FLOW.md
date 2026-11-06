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
| change               | a velocity change, Δv in widths per second, where a move starts or stops, read by the 3D bridge beside its trail: start, bow, stop, wake | `VelocityChangeBridge`, `Change`, group `change` |
| region               | the pixels around one, which tell a part of the body changing speed from a limb arriving or leaving | `start_region_size`, `stop_region_size`, `*_region_weight`         |
| injection            | one bridge output as one fluid takes it, at its weight                                                | `Injection`                                                        |
| fluid                | the Navier–Stokes simulation; its density is the show                                                  | `fluid_2d/`, `fluid_3d/`, `Fluid2D`, `Fluid3D`                     |
| volume               | the 3D fluid's z layers: their count, spacing and the composite's look                                | `VolumeSettings`, group `volume`                                   |
| depth                | the camera's depth in metres, on the flow's view                                                      | `DepthView`                                                        |
| depth velocity       | the z of the 3D flow: the depth change along the x/y displacement, in view widths per second          | `DepthVelocity`, `DepthVelocitySettings`, group `depth_velocity`   |
| 2D flow, 3D flow     | a system: a motion layer and a bridge layer per camera, a fluid layer per show output                 | `Motion2DLayer` + `Bridge2DLayer` + `Fluid2DLayer`; `…3D…`         |
| 2D camera, 3D camera | an image and a mask; those and a depth image on the same view                                         | `MotionInput` without / with `depth`                               |
| port                 | a stage's named texture with its kind: image, velocity, scalar; its input and output fields declare theirs, every component its own | `Port`, `TextureKind`, `HasPorts`, `ports_of`, in `modules/flow` |
| monitor              | a layer that shows one port of one stage, picked on the panel, drawn only while it is shown; its views in [MONITOR.md](MONITOR.md) | `MonitorLayer`, `monitor_a`, `monitor_b`              |

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
`change.fast`, `trail.output`), `input.*` at the bottom. A texture not allocated is left out, so a part the stage does not run gives none.
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
| bridge | `BridgeInput`: `velocity`, from the motion output; `dt`; `image`, or `None`, the picture whose RGB colors the density; `color`, an RGB tint on it, white by default; `mask`, or `None`, which gates the image | `BridgeOutput`: `velocity`; `density`, what the fluid injects: in 2D the amount alone, the trail's speed (R); in 3D RGBA, the amount (the mix's, [The mix](#the-mix)) times `color`, times the image's RGB where there is one, with the amount as alpha; `temperature`; `acceleration`, a second velocity the fluid takes as an acceleration, or `None` |
| fluid  | `FluidInput`: `dt`, and `injections`, a list of `Injection`: `velocity`, `density`, `temperature`, `acceleration` from one bridge output, and `spin`, which no bridge gives, with `weight`, 0 to 1, the injection's share of the fluid's velocity, density and temperature, as a layer's weight in the light renderer's mix, and `color`, the color the 2D fluid gives the injection's density at its output; the app builds one injection per bridge it wants in that fluid | `FluidOutput`: `density`, the show, which the composite draws as the layer's `texture` |
| monitor | `MonitorInput`: `ports`, every stage's ports keyed by stage; the monitor picks by its own `stage` and `texture` | none: it draws                                                                                           |

The motion layer measures once per camera frame: when `timestamp` differs from the one it last
measured; a source without stamps (`None`) measures every tick. A gap wider than the expected
frame interval is a dropped frame. For its first 3 s after start the motion layer stays at rest
(`WARM_UP_TIME`): the camera's exposure settles then, and the Lucas–Kanade solve reads the
brightening as motion, x/y at full reach over the whole image **(site fact)**. A camera frame has no other identity: `CameraFrame` and `Capture` carry a
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
`timestamp` (`frame_interval`, `motion_layer.py`), held to half to twice the expected frame
interval, 1 / the camera's set frame rate (the motion settings' `fps`, shared from the app's
`input_fps`). The range is fixed, not learned from the gaps, so a burst of frames a few ms apart
while the camera comes up to speed is held once and does not carry into the frames after it; `dt`
is the fallback for a source without stamps or frame rate. The bridge and the fluid run per render
frame, so they step on `dt`, the tick's measured length, which the app holds to at most twice the
mean tick (`hold_tick`, `time_step.py`): a stall counts as one long frame and the rest is dropped,
as a capture gap is held to twice the expected frame interval. The fluid's `TimeStep` smooths it over `fps_smoothing`; a raw per-frame dt looks jittery
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
| `Bridge2DLayer`, `Bridge3DLayer`            | the subclasses, each on its bridges (`bridge_2d`; `bridge_3d`, the velocity trail and beside it the velocity change bridge, [The changes](#the-changes), and their mix, [The mix](#the-mix)) |
| `FluidLayer` (ABC)                          | `StageLayer[FluidInput, FluidOutput]`; `_update()`: `_inject` each injection's velocity, density and temperature at its weight, `_step(dt)`; `texture`, the show. Knows injections, not cameras; reads no poses. The stage's reset is the fluid's only one |
| `Fluid2DLayer`, `Fluid3DLayer`              | the subclasses, each on its own fluid. `Fluid2DLayer` keeps its injections apart: injection i's amount goes into channel i of the density, four at most, and the output colors each channel with its injection's color (`DensityColorize`), the brightness capped at white and the opacity the strongest channel's. `Fluid3DLayer` injects RGBA density, the acceleration (added as the velocity is) and a spin where one is given, the velocity's z as the volume's w (its sign turned: w points away from the viewer), and takes the parallax eye the app steers by `set_eye(x, y)` |
| `MonitorLayer`                              | `set_input(MonitorInput)`; `update()` keeps the panel's list of the chosen stage's port names (`names`, the options of `texture`); `draw()` renders the port once per tick, however many panels show it, by its kind or by `view`, and nothing while not shown; its views and settings are in [MONITOR.md](MONITOR.md). Not a stage, since it has no output and no `reset` |
| `MotionLayerSettings`, `BridgeLayerSettings`, `FluidLayerSettings` | base settings the 2D and 3D settings subclass, each a `StageLayerSettings`: `reset`, then `width` and `height`, the stage's working size, first on every stage's panel. The motion's is the size its flow runs at, the bridge's the size it writes its outputs at (its filters follow the velocity's size), the fluid's its simulation grid (its density is at the size the app allocates). On the fluid layer `blend_mode`. A layer whose stage runs one component is that component's settings too (`Motion2DLayerSettings` is a `GradientFlowSettings`, `Motion3DLayerSettings` a `LucasKanadeFlow3DSettings`, the fluid layers' a `Fluid2DSettings` and a `Fluid3DSettings`), so the component's fields and groups sit on the stage's panel; the bridge layers, running several bridges, hold them in two subpanels (the 3D bridge's `readings`, `trail` and `change`, and `outputs`, `velocity`, `acceleration`, `density`, `temperature`, each with its mix). No `draw_mode`, `visualization`, `stage`, `max_players` or `fps`: the stages draw nothing, read no poses and take `dt` from their input |

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

The bridge reads two things from a movement: the motion itself, a velocity in its direction that
is stronger for fast than for slow and coasts after a stop; and its changes, the velocity changes
where a move starts and where it stops. The motion wants a long, soft reading, the changes a
short, sharp one, so they are two bridges side by side on the same velocity (`bridge_3d/`), the
readings; the outputs mix them into what the fluid takes ([The mix](#the-mix)). The panel shows
them as two subpanels, `readings` and `outputs`:

| Bridge                 | Reads                                                                  | Gives                                                              | Panel group       |
|------------------------|------------------------------------------------------------------------|--------------------------------------------------------------------|-------------------|
| `VelocityTrailBridge`  | the flow's velocity through one trail: advect, attack/smooth, blur     | `smooth_velocity`, the trail; its speed is the density amount      | `readings.trail`  |
| `VelocityChangeBridge` | the flow's velocity through a fast and a slow trail, and their regions | `start`, `bow`, `stop`, `wake`, each also as `<change>_velocity`   | `readings.change` |

The fluid makes its own swirl from what it is given, so the bridge gives no curl of its own.

### The velocity trail

One unit-gain Ema per pixel: a rising 3D speed follows over `attack`, a falling one over `smooth`,
so a flick reaches the fluid at full strength and a stop coasts. Both bridges treat x, y and z
alike: z is a little noisier than x/y, and its one time constant is the motion's (the depth
velocity's `smooth`), which every reader of z then shares, as a z time per bridge would add the
same lag in more places. `advect` moves the
trail with its own motion before each step, so the average follows a moving limb instead of a
fixed pixel. `blur_radius` and `blur_steps` spread it in space; the ring it leaves around the
body is part of the look **(site fact)**.

### The changes

The difference of the fast and the slow trail along the motion's direction rises where a move
starts and falls where it stops; it lasts about `slow_smooth`, as the slow trail catches up. It is
divided by the trails' step peak (`step_peak`, 0.58 at 25 and 150 ms), so a limb at speed v that
starts or stops at once reads v: the changes are in the motion's own unit, image widths per
second, a velocity change Δv, and a gradual start or stop reads less, by how much of its speed
changes within the trails' times. A pixel cannot tell a move starting from a limb arriving at it, nor a move stopping from a limb
leaving it: its speed rises or falls either way. The region around it can, so the region splits
each pixel's rise and fall in two, by the region weight (`start_region_weight`,
`stop_region_weight`; 0 gives all of it to the start and the stop):

| Change | Is                                               | Along                 |
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

The changes read x, y and z alike. The bridge filters in time only: noise within a frame is the
motion's to filter, and the trail's blur is a look. A change takes no hold: it lasts about
`slow_smooth` by itself, and what a stop injects is the speed it ended, as a hold would lift steady
motion as much as the stops. No blur and no advect: the changes stay sharp. The change bridge
runs while an output takes a change or a monitor shows one of its ports.

### The mix

What the fluid takes is a mix of the two readings, each output named after the quantity it is and
each source at a share, 0 to 1, as the light renderer mixes its layers (`Bridge3DLayer`,
`WeightedSum`). Each output's group in `outputs` holds the five shares as knobs on the first row
(`MixSettings`: `trail`, `start`, `bow`, `stop`, `wake`), and the density's and the temperature's
own settings on the second (`DensityMixSettings`, `TemperatureMixSettings`):

| Output         | Sources                                                        | Second row             | Default       |
|----------------|----------------------------------------------------------------|------------------------|---------------|
| `velocity`     | the trail's velocity, the start, bow, stop and wake velocities |                        | the trail 1   |
| `acceleration` | the same                                                       |                        | all 0: `None` |
| `density`      | the trail's speed, the start, bow, stop and wake               | saturation, brightness | the trail 1   |
| `temperature`  | the same, as the temperature's mask                            | scale                  | the trail 1   |

The sources share one unit, image widths per second, so the mix has no gain: the motion's gains
balance the axes and the fluid's gains set the strength; a weight is a share. The velocity and the
acceleration are two outputs, since a fluid takes a velocity and an acceleration two ways
([Open](#open)); the 3D fluid adds both as the velocity is added today. The mix's ports are
`mix.velocity`, `mix.acceleration`, `mix.density` and `mix.temperature`.

### Measured

`scripts/bridge_compare.py` on the standmoves recording (`20261103-215324`: four whole-arm moves
from rest, with starts and stops, a player at 0.65–1.5 m), its frames as captured on 2026-11-05
(`temp/depth_track_1105/`, 1449 frames, 25 fps, the stamps the replay's decode times), the flow run
again on them with deep_flow's preset flow of 2026-11-06 (levels 4, same_surface and xy_smooth on,
the depth velocity's `smooth` 50 ms unless named), each bridge at its preset values (trail: smooth
1000 ms, attack 1 ms, advect 1, blur 4 px × 4; change: 25 / 150 ms, both region weights 1,
regions 128 px), stepped once per flow frame. Per frame, the mean over the body's pixels that move
in the loop; noise is the still part's p99 per pixel; slow and fast are the steady halves of the
moves; stop ÷ fast is the stop window's peak over steady fast motion, start ÷ fast the start
window's.

The depth velocity's `smooth`, with both bridges treating z like x/y:

| `smooth` | Flow z noise | Flow z, fast | Trail z noise | Trail z, fast | Start noise | Stop ÷ fast | Start share at starts / steady |
|----------|--------------|--------------|---------------|---------------|-------------|-------------|--------------------------------|
| 50 ms    | 0.160        | 0.093        | 0.243         | 0.387         | 0.079       | 4.07        | 0.34 / 0.37                    |
| 100 ms   | 0.012        | 0.079        | 0.066         | 0.329         | 0.000       | 4.24        | 0.38 / 0.36                    |
| 200 ms   | 0.000        | 0.047        | 0.005         | 0.171         | 0.000       | 10.41       | 0.42 / 0.30                    |
| 400 ms   | 0.000        | 0.020        | 0.000         | 0.057         | 0.000       | 10.88       | 0.48 / 0.24                    |

The trail follows a rising speed at once, so it keeps z's upward noise for about `smooth`: it needs
a calm z. At 100 ms the flow's z keeps 85% of its motion and loses 93% of its noise; at 200 ms its
noise is gone and half its motion is left; at 400 ms little of either. The start's region only
tells a start from a limb arriving once z is calm: its share at real starts against steady motion
grows from equal at 50 ms to 0.42 against 0.30 at 200 ms.

The stop's region, `smooth` 50 ms:

| Stop reading                                   | Noise | Stop ÷ fast | Falls in steady motion read as stop | At real stops |
|------------------------------------------------|-------|-------------|-------------------------------------|---------------|
| region weight 0                                | 0.040 | 2.46        | 1.00                                | 1.00          |
| region 64 px                                   | 0.019 | 3.57        | 0.44                                | 0.63          |
| region 128 px (the preset)                     | 0.010 | 4.07        | 0.38                                | 0.62          |
| region 256 px                                  | 0.008 | 4.31        | 0.36                                | 0.61          |
| region 128 px over the flow's moving pixels    | 0.011 | 3.48        | 0.45                                | 0.63          |

A region over the flow's moving pixels, as the start's, reads more of a leaving limb as a stop and
the real stops the same, so the stop's region reads the trails' mean speed. Without the region the
stop's noise is four times higher.

| GPU, each bridge alone | Mean, ms | p95, ms |
|------------------------|----------|---------|
| the trail              | 1.55     | 2.13    |
| the change bridge      | 1.88     | 2.38    |

Advice, from these numbers on this clip:

| Setting                       | Value  | Why |
|-------------------------------|--------|-----|
| the depth velocity's `smooth` | 200 ms | z calm for the trail's attack and the start's region, at +150 ms of z's lag; 100 ms for a livelier z at some noise |
| `stop_region_size`            | 128–256 px | 256 a little cleaner (noise 0.008, 4.31×), 64 worse |
| `stop_region_weight`          | 1      | without it the stop's noise is four times higher and its stops half as distinct |

The clip moves whole arms from rest, so a limb arriving and a part speeding up come together: the
start and bow split, `start_region_*`, `fast_smooth` and z's changes toward the camera are not
tested on it, and the trail's and change bridge's own settings are left as the preset.

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
- A slow frame never gives the fluid more work. Its step is at most 1/20 s (`MAX_STEP`), so a
  slower frame runs the fluid slower than real time and nothing moves further in one step; the
  pressure solve takes its `iterations` at any step, since the grid sets what it needs; the
  viscosity takes its `viscosity_iter` at a 60 fps step and fewer at a shorter one, never more
  (`viscosity_iterations`). Iterations that grew with the step kept a dropped frame rate down: at
  4 fps, 15 times the work per frame **(deduction)**.
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

## Open

- For the fluid refactor: the 3D fluid's `input_x` and `input_y` go, as the motion's `x_gain` and
  `y_gain` balance the axes.
- For the fluid refactor: a burst of false velocity slows the 3D fluid to 2 to 4 fps. With
  `WARM_UP_TIME` 0, the exposure ramp at startup reads as x/y motion; with the fluid updated the
  frame rate drops to 2 to 4 fps, without it it stays up **(site fact)**. Nothing in the fluid
  loops by the velocity, so the cost is in its reads: the advection traces each texel back by
  velocity × dt, noise points neighboring texels apart, and the density's advection over 2160 ×
  3840 × 16 RGBA16F, about 1 GB, then misses the texture cache on every read **(deduction, not
  measured)**. Ordinary motion moves neighbors alike and reads together. `dampen_threshold`
  (5 widths per second over `dampen_time`, 0.5 s) is too slow to stop it. The motion's
  warm-up keeps the startup burst out (`WARM_UP_TIME`); a capture run at 4 fps rather than 25
  (`scripts/depth_velocity_capture.py`) was this burst **(site fact)**. What remains is the fluid's side: measure
  the step's GPU time on a still velocity and on a burst, then cap the trace at a few cells per
  step in the advection, which bounds the scatter whatever comes in and which real motion stays
  under at 60 fps.
- For the fluid refactor: no bridge gives a spin any more, so `Injection.spin`, `Fluid3D`'s spin
  input (`set_spin`, `vorticity_force_3d`'s `spin`) and `Fluid3DLayer`'s use of it have no source.
- Whether the density's reduced view should also feed the fluid's own composite at a lower
  quality setting.
- How a fluid takes what the bridge gives, the base of the fluid refactor. Both fluids add the
  injected velocity times dt (`Fluid2D.add_velocity`, `Fluid3D.add_velocity`), as Stam's Stable
  Fluids adds a source; but the bridge's velocity is a velocity, in view widths per second like
  the fluid's, so it is taken as an acceleration, `velocity.gain` acts as a rate, and the fluid
  stays near 0.05 to 0.1 of the motion's speed **(deduction)**, a monitor needing a scale of 20 to
  30 to show it. In Navier–Stokes an outside term is an acceleration per unit mass, as gravity and
  buoyancy enter. The bridge gives the two quantities apart, `velocity` and `acceleration`; how
  the fluid takes each, the velocity as a velocity and the acceleration times dt, is the fluid
  refactor's. A change lasts about `slow_smooth` over many ticks, so a velocity change added
  without dt every tick would count once per tick, 60 times its Δv in a second **(deduction)**.
- hd_trio's density fade from motion. Its wiring pass writes `density.fade_time` every tick from
  the pose's motion, a settings field written per tick. A per-tick value on the fluid layer, as the
  eye is, would take it off the settings.
- Unused channels written 0 in the whole flow module. The 3D motion's and the 3D bridge's
  velocities write 0 in their unused w (`compose_velocity.frag`, `velocity_trail.frag`,
  `velocity_change_direction.frag`), so a monitor shows w as
  nothing rather than a full strip lane and a histogram spike at 1 × `scale`. The rest of
  `modules/flow` still writes 1.0 into unused channels, an alpha left from when the monitors
  blitted textures as they are: among others `gradient_flow_shader.frag`,
  `lucas_kanade_shader.frag`, `displacement_threshold.frag`, the change scalars and
  `magnitude_shader.frag`. It matters only where the target texture has that channel (an R16F
  target drops it). Each wants its readers checked first, as the velocities' were: there the
  trails, the change shaders, `VelocityMagnitude` and `inject_flow_3d.comp` read xyz only,
  `trail_advect.frag` carries the vec4 along unread, and `Fluid3D.set_velocity` has no caller.
  The visualization's own shaders keep alpha 1, since they draw for the screen.
