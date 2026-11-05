# The 2D and 3D flow systems

Two app-independent flow systems, each a pipeline of three stages, motion, bridge and fluid: the
2D flow, a velocity on the image plane, and the 3D flow, a velocity with a component toward the
camera. Either system takes either kind of camera. An app picks a system and wires its cameras
to it. This document states the design and the steps that bring the code to it; steps 0 to 5
have run.

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
layer per stage, so the layers are the packages.

### Inputs and outputs

Each stage takes one input dataclass and gives one output dataclass. Both hold values only:
textures and numbers, never an output; the fluid's input holds its injections as a list of
`Injection`, each values. The app's wiring pass builds each input new every
tick, copying into it what the stage reads from the output of the stage before and from the
camera layers, and sets it on the layer before the render loop. The output holds the stage's
textures, allocated once, and the layer hands it out as `output`. The layers hold no reference
to each other, and no input knows an output's type. A field is named with the same noun for the
same kind of thing, in every input.

A texture field holds the `Fbo`, `SwapFbo` or `Tensor` object itself, which stands for its
current buffer, never a raw `.texture` or `.back_texture`: a swap's buffer expires at the next
swap, and holding one raises `StaleTextureError` (`modules/gl/swap_handle.py`). That is what
lets an input be built before the render loop: the object named in it is the one the stage
before writes into during the loop, so the next stage reads this tick's result. Values that are
not textures (`timestamp`, `dt`, the weights) come from the app, never from an output, or they
would be the tick before's.

| Stage  | Input `@dataclass`, built by the app each tick                                                                                      | Output `@dataclass`, the stage's textures                                                                |
|--------|-------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------|
| motion | `MotionInput`: `image`, the picture the velocity is measured on by its brightness; `timestamp`, the capture time of the camera frame `image` holds, the frame's identity; `depth`, a `DepthView` or `None`; `mask`, or `None`; `dt`, the render tick's | `MotionOutput`: `velocity` (x, y, z), gated by `mask` with `use_mask`; z is 0 without depth    |
| bridge | `BridgeInput`: `velocity`, from the motion output; `image`, or `None`, the picture whose RGB colors the density; `color`, an RGB tint on it, white by default; `mask`, or `None`; `dt` | `BridgeOutput`: `velocity`; `density`, RGBA, what the fluid injects: the amount (the trail's speed in 2D; the kinetic emission or the speed in 3D) times `color`, times the image's RGB where there is one, with the amount as alpha; `temperature`; `onset` or `None`; `spin` at its `spin` setting or `None` |
| fluid  | `FluidInput`: `dt`, and `injections`, a list of `Injection`: `velocity`, `density`, `temperature`, `onset`, `spin`, from one bridge output, with `weight`, 0 to 1, the injection's share of the fluid's velocity, density and temperature, as a layer's weight in the light renderer's mix; the app builds one injection per bridge it wants in that fluid | the layer's `texture`: the show                                                       |

The motion layer measures once per camera frame: when `timestamp` differs from the one it last
measured. A gap wider than a frame period is a dropped frame. A camera frame has no other
identity: `CameraFrame` and `Capture` carry a timestamp and no sequence number, because the
device's sequence numbers skip on the depth while every frame arrives (`modules/camera/oak4/OAK4.md`).

`DepthView(texture, view_span, depth_step)` is the depth on the flow's view: the texture, the
metres one view width spans per metre of depth, and the uv between two depth samples. The 3D
camera's layers make it: `DepthSourceLayer` the texture and its sample size, `CenterGeometry`
the span, from the crop and the camera's horizontal field of view it takes at construction.

Two clocks. The motion layer measures between two camera frames, so its interval comes from
`timestamp` (`Motion3DLayer._frame_interval`, with `dt` as the fallback for a source without
stamps). The bridge and the fluid run per render frame, so they step on `dt`, which the fluid's
`TimeStep` smooths over `fps_smoothing`, an Ema of the `dt` it is handed; a raw per-frame dt
looks jittery in the sim **(site fact)**.

A tick without a new camera frame measures nothing: the motion output holds what the last pair
of frames gave, the tick's `dt` adds to the interval the next measurement covers, and the bridge
and the fluid run as on any tick, on the held velocity. A velocity held between samples and
integrated at the render rate is the right reading of it: the motion continues until the next
pair says otherwise, and the stamps keep the magnitude per second right whether a frame took one
tick or two. The bridges are filters in time (the kinetic bridge's trails, hold and reservoir
decay by `decay_weight(dt, smooth)`, its carry advects by velocity × dt), so a held velocity
stepped every tick on `dt` is the same filter sampled finer; today the kinetic bridge steps once
per camera frame on the frame interval, borrowing the motion stage's signal (`Motion3DLayer.update`).
Between two 50 fps frames the render takes one or two 60 Hz ticks, so one frame's integral of
the held velocity is 17 or 33 ms worth instead of 20; the mean is right, and the trails and the
fluid smooth it **(deduction)**.

The render runs on the display's vsync (`WindowManager._render_loop`, `swap_interval(1)`); the
camera runs at 50 fps, the mains rate **(site fact)**, against a 60 Hz display. The board holds
one latest capture per camera (`CaptureStoreMixin.set_capture`), which `CaptureImageLayer.update`
takes once per tick. At these rates every frame gets a tick of its own and none is lost; a frame
waits for the next vsync, up to one 60 Hz tick, before it is measured.

### The layers

| Class                                       | Role                                                                                                                                                                                  |
|---------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `MotionLayer` (ABC)                         | `set_input(MotionInput)` before the render loop; `update()` as the template method: `_measure` on a new frame, visualize; `output: MotionOutput`. Holds the visualizer, the allocate and resize pattern, the draw-mode selection |
| `Motion2DLayer`, `Motion3DLayer`            | the subclasses, each on its optical flow (`GradientFlow`; `LucasKanadeFlow3D`)                                                                                                        |
| `BridgeLayer` (ABC)                         | `set_input(BridgeInput)`; `update()`: gate the image, where there is one, by the mask, `_bridge` the input's velocity into velocity, density and temperature, visualize; `output: BridgeOutput`                    |
| `Bridge2DLayer`, `Bridge3DLayer`            | the subclasses, each on its bridges (`bridge_2d`; `bridge_3d`)                                                                                                                        |
| `FluidLayer` (ABC)                          | `set_input(FluidInput)`; `update()`: `_inject` each injection's velocity, density and temperature at its weight, `_step(dt)`; `texture`. Knows injections, not cameras; reads no poses |
| `Fluid2DLayer`, `Fluid3DLayer`              | the subclasses, each on its own fluid, both injecting RGBA density; `Fluid3DLayer` also injects onset and spin and takes the parallax eye the app steers by `set_eye(x, y)`              |
| `MotionLayerSettings`, `BridgeLayerSettings`, `FluidLayerSettings` | base settings the 2D and 3D settings subclass: `draw_mode`, `blend_mode`, `visualization`, and on the motion layer `width`, `height`; the stage's own settings as a group named after its method (`gradient`, `lucas_kanade`; `trail`, `kinetic`, `density`, `temperature`). No `stage`, `max_players` or `fps`: the layers read no poses and take `dt` from their input |

The apps' render is a layer graph of `LayerBase` layers updated by one loop calling `update()`
on each (`apps/deep_flow/render.py`, `apps/hd_trio/render.py`), with a wiring pass before it.
The wiring pass builds a motion and a bridge input per camera and a fluid input per fluid layer,
with one injection per bridge that fluid takes, and sets them; the loop, which `UPDATE_LAYERS`
orders motion, bridge, fluid, stays uniform. Which layers feed the inputs is app policy: hd_trio
wires fixed layers; deep_flow picks them on its panel, live, one `Layers` field per picture
(`motion_image`, `bridge_image`, `mask`) in a group of its own (`apps/deep_flow/settings.py`), and
its wiring pass reads the fields each tick. A source layer keeps the capture time of the frame it
holds (`CaptureImageLayer.timestamp`); a derived view (`CenterCamLayer`, `CenterMaskLayer`,
`CenterDepthLayer`) carries it from its source, so the app reads it off the layer it takes the
texture from.

## Packages

| Package                         | Holds                                                                                                                                                      |
|---------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `modules/flow/motion/`          | `GradientFlow`; `LucasKanadeFlow`, the x/y solve with `LucasKanadeSettings`; `LucasKanadeFlow3D`, composing it with `DepthVelocity` and `DepthVelocitySettings` |
| `modules/flow/bridge_2d/`       | the 2D bridge                                                                                                                                              |
| `modules/flow/bridge_3d/`       | the 3D bridge                                                                                                                                              |
| `modules/flow/fluid_2d/`        | `Fluid2D`, `Fluid2DSettings` with its own `VelocitySettings`, `DensitySettings`, `TemperatureSettings`, `PressureSettings`, `BoundaryMode`; `shaders/` (11) |
| `modules/flow/fluid_3d/`        | `Fluid3D`, `Fluid3DSettings` with its own groups and `VolumeSettings`; `shaders/` (19)                                                                     |
| `modules/flow/shaders/`         | the shared fragment shaders, `AddBoolean` among them                                                                                                       |
| `modules/flow/time_step.py`, `modules/flow/visualization/` | shared                                                                                                                          |
| `modules/render/layers/flow/`   | `motion_layer.py` (`MotionInput`, `MotionOutput`, `MotionLayer`), `motion_2d_layer.py`, `motion_3d_layer.py`; `bridge_layer.py` (`BridgeInput`, `BridgeOutput`, `BridgeLayer`), `bridge_2d_layer.py`, `bridge_3d_layer.py`; `fluid_layer.py` (`Injection`, `FluidInput`, `FluidLayer`), `fluid_2d_layer.py`, `fluid_3d_layer.py`; `depth_view.py` |

Neither `fluid_2d` nor `fluid_3d` imports the other. The settings group of each fluid is `fluid`.
`motion/` has no `_2d` and `_3d` packages because its 3D flow is its 2D flow plus z:
`LucasKanadeFlow3D` composes `LucasKanadeFlow` and `DepthVelocity`, where the bridges and the
fluids are two implementations. The visualization is shared because it is not a stage: it draws
the texture it is given, a 2-channel one as a 2D velocity and a 3- or 4-channel one as a 3D
velocity (`VelocityField.update`), and the signed views serve the 3D flow's scalars.
The shader loader finds a shader's source by its `.py` module name beside it
(`modules/gl/shader.py`), so moved shaders need no edit.

## Constraints

- Each pipeline stage is one package and one layer; a 2D and a 3D version of a stage sit side by
  side inside the package. Nothing is grouped by app or by camera.
- The 2D system's measurement and look stay as on `main` (`GradientFlow`, `bridge_2d`, `Fluid2D`)
  until the 2D optical flow moves to the Lucas–Kanade solve. Its names, its wiring and where its
  density is colored may change, and it takes the mask gate with the 3D system. hd_trio measures
  on the mask itself (`apps/hd_trio/render.py`), so the gate changes nothing there **(deduction)**.
- The fluid knows injections, not cameras. The 2D density is RGBA colored by the bridge, from the
  app's track colors in hd_trio; `Fluid2D.add_density_channel` and the colorizing of one channel
  per camera at the output go, and with them the limit of four cameras. Advection and fade are
  linear per channel, so the picture is the same; where cameras overlap the alpha is the sum, not
  the max, the clamp of V at 1 goes, and `density.dampen_*` reads the color's length **(deduction)**.
- The inputs and outputs are the contract and the base classes hold the two kinds to one shape;
  there are no `Protocol`s for the boundaries. Shared code lives in the base, never in a shared
  concrete class.
- The optical flow classes are GL but no `LayerBase`, so that the motion stage can later run on a
  thread of its own, driven by the camera, with the render thread reading its textures; a
  `LayerBase` ties the measurement to the render tick. Until then it runs in the motion layer, on
  the render thread, as all GL does.
- The image, the mask and the depth come pushed in the inputs: a GL texture lives on the render
  thread and is not board data. The layers read nothing from the board; the app's wiring pass
  reads the pose frames there for the weights and the eye.
- The motion layer reads no layer's `dirty`: `MotionInput.timestamp` says whether the image is a
  new camera frame. `LayerBase.dirty` stays as the source layers' change flag for their other
  readers (`DepthMaskLayer`).
- The time step has one transport: the app hands the tick's `dt` in the inputs, and the fluid's
  `TimeStep` smooths it over `fps_smoothing`, or steps fixed at `fixed_fps`. No `fps` or
  `step_fps` field: runtime data stays off the settings.
- One `Visualizer` serves both systems: it draws a 2-channel texture as a 2D velocity and a 3- or
  4-channel one as a 3D velocity (`VelocityField.update`). The signed views belong to the 3D flow.
- The 2D bridge keeps three classes the 3D bridge also has (`DensityBridge`, `TemperatureBridge`,
  `VelocityMagnitude`) as copies, until the 2D bridge is rebuilt.
- The 3D fluid layer runs the 3D fluid only. Comparing the two fluids on one input means both
  systems wired in one app.
- An outside source gates an input through its weight: every injection carries `weight`, 1 by
  default, and the app sets it from whatever it reads (hd_trio: the pose's motion and the
  cross-camera similarity). The fluid never knows why.
- A multiplier on an input is a `gain` (the fluids' `velocity.gain`, `density.gain`,
  `temperature.gain`); a 0 to 1 share in a mix is a `weight`; the density's sinking along gravity
  in the buoyancy is `temperature.sink` (today `weight`, which both fluids read:
  `Fluid2D._apply_buoyancy`).

## Current state

The distance between the code on `deep-flow-prep` and the design; the steps close it.

| Area          | Code                                                                                                                                                                                                                                                              |
|---------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| fluid layer   | `Fluid2DLayer` is one fluid per camera fed by every camera's flow layer, storing camera i's density in channel i (`Fluid2D.add_density_channel`) and coloring the channels with the track colors at the output (`DensityColorize`), so the 2D system takes at most four cameras |
| motion and bridge | one layer holds both stages: `Motion2DLayer` takes its mask as a source layer in the constructor and has no time step; `Motion3DLayer` has seven setters and 26 draw modes mixing the two stages' views; the outputs are `velocity` on one and `smooth_velocity` on the other |
| optical flow  | `LucasKanadeFlow` holds the x/y solve and the depth track in one class, with `lucas_kanade` and `depth_track` as its settings groups                                                                                                                                 |
| time step     | the apps write the measured tick rate into the fluid's `fps` field (`_measure_fps` in both `render.py`) and deep_flow reads `step_fps` back to step its motion layer; hd_trio's presets carry both READ fields                                                            |
| policy        | `Fluid2DLayer` reads the board for each camera's weight (the pose's motion, the cross-camera similarity), the same value on velocity and density, injects temperature at 1, and sets `density.fade_time` from the motion; `Fluid3DLayer` reads the board for `scale_by_motion` and the parallax eye, and the motion layer's settings for `onset.spin` |
| new-frame signal | `LayerBase.dirty` defaults to `True`, so a center view (`CenterCamLayer`, redrawn every tick as the view follows the pose) claims a new frame on every tick; the flow asked it would measure between two ticks holding one camera frame, find no motion and count a one-tick interval. deep_flow therefore maps each view to the source behind it (`FLOW_SOURCES`) and hands the flow the texture from the view and the source for the signal and the stamp |

## Steps

Each step is one commit that leaves both apps importing and both presets loading. Names first,
so nothing moves twice.

| #  | Step                                                       | Files                                                                                                                                                                                                                                 | Presets                                                                                                   |
|----|------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------|
| 0  | 2D arrow views at `main`'s size                            | none                                                                                                                                                                                                                                  | hd_trio: `element_length` 40.0 → 2.0 (40 px at `spacing` 20) in both `visualization` groups               |
| 1  | The names: classes, `volume`, the stages                   | the renames in [Terms](#terms); `modules/flow/__init__.py`, `modules/render/layers/__init__.py`, both apps' `render.py` and `settings.py` (`Layers.motion`); the glossary; `tests/test_glossary.py` if a term is guarded | `flow` → `motion`, `fluid_flow` → `fluid`, `depth` → `volume` with its keys, in both apps. `.ui_state.json` is the panel layout and needs no edit **(site fact)** |
| 2  | `AddBoolean` to the shared shaders                         | `fluid/shaders/add_boolean.*` → `shaders/`; both `shaders/__init__.py`; the two fluids' imports                                                                                                                                        | none                                                                                                      |
| 3  | The packages flat: `optical/` → `motion/`, `fluid/` → `fluid_2d/`, `fluid3d/` → `fluid_3d/`, `bridge/bridge_2d/` → `bridge_2d/`, `bridge/bridge_3d/` → `bridge_3d/` | `git mv`; `bridge/__init__.py` out; the bridges' relative imports one level up (`from .. import FlowBase`); `modules/flow/__init__.py`                                                                               | none                                                                                                      |
| 4  | Split the fluid settings                                   | `fluid_2d/fluid_settings.py` without the 3D-only fields and the volume; `fluid_3d/fluid_settings.py` without `density.dampen_*`; `fluid_3d.py`, `fluid_3d_layer.py` on `Fluid3DSettings`; `input_strength` → `gain` on velocity, density and temperature, `temperature.weight` → `sink`, in both | hd_trio: drop `fluid.fluid.volume`; deep_flow: drop `fluid.fluid.density.dampen_*`; `input_strength` → `gain`, `temperature.weight` → `sink` in both |
| 5  | `Fluid3DLayer` on the 3D fluid alone                       | `fluid_3d_layer.py`: `FluidSimEngine`, the engine swap and the `isinstance` branches out                                                                                                                                              | deep_flow: drop `fluid.sim_engine`                                                                        |
| 6a | The inputs and outputs                                     | `depth_view.py` (`DepthView`); `CenterGeometry` takes the fov and gives the span, `DepthSourceLayer` the sample size; `MotionInput`, `MotionOutput`, `BridgeInput` with `color`, `BridgeOutput`, `Injection`, `FluidInput` in their layers' files | none                                                                                                      |
| 6b | The bases                                                  | `motion_layer.py`, `bridge_layer.py`, `fluid_layer.py`: the ABCs with `set_input`, `update()` as the template method, `output`, the visualizer, the allocate and resize pattern, the draw-mode selection, and their `…LayerSettings`, out of today's two motion layers and two fluid layers; `TimeStep.update` takes the tick's `dt` in place of `measured_fps`; `fps` and `step_fps` out of the fluid settings | hd_trio: drop `fluid.fluid.fps` and `fluid.fluid.step_fps`                                               |
| 6c | The 3D side on them                                        | `Motion3DLayer` keeps the optical flow and its draw modes and gives the bridges and theirs to `Bridge3DLayer`; `fluid_3d_layer.py` injects each injection at its weight, `set_eye`; deep_flow's wiring pass builds the three inputs, with the image and a white `color` for the bridge, the tick's `dt` in each, the weight, and steers the eye; `_measure_fps` and `set_time_step` out; deep_flow's live source choice (`FluidInputSettings`, today a subclass of the motion layer's settings) becomes its own group on the app's root | deep_flow: `motion.optical_flow` → `motion.lucas_kanade`, `motion.flow_bridge` → `bridge`, `motion.draw_mode` split in `motion.draw_mode` and `bridge.draw_mode`, `motion.inputs.{flow,density,mask}` → `inputs.{motion_image,bridge_image,mask}`; `fluid.scale_by_motion`, `stage`, `max_players` to the app's groups |
| 6d | The 2D side on them                                        | `Motion2DLayer` keeps `GradientFlow` and gives the bridges to `Bridge2DLayer` (the smooth trail; `density` RGBA from the speed and `color`, times the image where there is one; temperature from the density; with the gate; main's RGBA density bridge stays a draw mode); `fluid_2d_layer.py` injects each injection at its weight, `cam_id`, the board, the flow layers, `ColorSettings`, `DensityColorize` and its FBO out, `Fluid2D.add_density_channel` and its region variant out; the weight from motion and similarity and the fade from motion go to hd_trio's wiring pass, which builds a fluid input per screen with an injection per camera, the tick's `dt` in each input, and gives each bridge its track color as `color`, no image; `_measure_fps` out | hd_trio: `motion.optical_flow` → `motion.gradient`, `motion.{velocity_trail,density_bridge,temperature_bridge}` → `bridge.…`, `motion.draw_mode` split; `fluid.similarity_scale`, `stage`, `max_players` to the app's groups; `motion.stage`, `motion.fps` out |
| 6e | The views carry the capture time                           | `CenterCamLayer`, `CenterMaskLayer`, `CenterDepthLayer` forward `timestamp` from their source; deep_flow's `flow_sources` out. Three places read `.dirty` today (`DepthMaskLayer`, the two motion layers); the motion layers read the stamp instead | none                                                                                                      |
| 7  | The 3D optical flow as its own class                       | `motion/lucas_kanade_flow.py` keeps the x/y solve and `LucasKanadeSettings`; `lucas_kanade_flow_3d.py` new, `LucasKanadeFlow3D` with `DepthVelocitySettings` (today `DepthTrackSettings`; the glossary's `depth_track` line with it); `motion_3d_layer.py` on it; `scripts/depth_track_*.py` → `depth_velocity_*.py`; `LUCAS_KANADE_FLOW.md`; the x/y group `lucas_kanade` inside it becomes `solve`, so the panel reads `motion.lucas_kanade.solve`, not `lucas_kanade.lucas_kanade` | deep_flow: `motion.lucas_kanade.lucas_kanade` → `motion.lucas_kanade.solve`, `motion.lucas_kanade.depth_track` → `motion.lucas_kanade.depth_velocity` |
| 8  | A settings test per system                                 | `modules/render/tests/test_flow_systems.py`: build each system's settings, load each app preset into it, assert no unknown key; and for each settings class, that every field's name appears in the source of the module that holds the class **(deduction)**: a field no code names is dead | none                                                                                                      |
| 9  | This file becomes `modules/flow/FLOW.md`                   | this file                                                                                                                                                                                                                             | none                                                                                                      |

Tests after each step:

```
python -m unittest discover -s tests -t .
.venv\Scripts\python -m unittest discover -s modules/flow/tests -t .
.venv\Scripts\python -m unittest discover -s modules/render/tests -t .
.venv\Scripts\python -m unittest discover -s apps/hd_trio/tests -t .
.venv-oak4\Scripts\python -m unittest discover -s apps/deep_flow/tests -t .
```

Neither fluid has a unit test until step 8; the apps are the integration test, and steps 6c and
6d each get a check in their app. `.gitignore` ignores `temp*`, so a moved or new `temperature_*`
file needs `git add -f`.

## Open

- hd_trio's density fade from motion. `Fluid2DLayer.update` writes `density.fade_time` every tick
  from the pose's motion; step 6d moves the write to the app, where it is still a settings field
  written per tick. A per-tick value on the fluid layer, as the eye is, would take it off the
  settings.
- The wait for vsync. A frame is measured at the first render tick after it arrives, up to one
  60 Hz tick later. Measuring on arrival means the motion stage on its own thread, driven by the
  camera, with its own GL context: the one GL outside the render thread. To measure the
  capture-to-measure latency and any dropped frames from the stamps first, at the real rates.
- `LayerBase.dirty` defaults to `True`. A layer that does not answer the question claims a new
  frame every tick. No default, so a layer either says it or cannot be asked: a small change of
  its own, outside these steps.
- The 3D fluid's synthesized w. `Fluid3D.add_velocity` makes w from the x/y flow (`push`,
  `squeeze`) and blends it with the velocity's z by `depth_mix`, and the fluid layer passes z
  only when the motion layer measured it (`Motion3DLayer.depth_enabled`). The design takes z as w,
  with no flag and no blend; `push`, `squeeze` and `depth_mix` go with it. deep_flow's preset runs
  `push` 0.5 and `depth_mix` 0 today, so its w is the synthesized one; removing them changes the
  show and is a change of its own, after the separation.
- A 2D bridge after a 3D motion, or the reverse, is expressible: a `BridgeInput` holds a velocity
  texture, and a 2D bridge reads its x and y. Nothing in the plan needs a mixed pair.
