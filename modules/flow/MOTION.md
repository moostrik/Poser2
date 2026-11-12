# The motion

Every quantity and setting here is in a meaningful unit ([FLOW.md](FLOW.md#the-2d-and-3d-flow-systems)).

The motion stage measures a velocity field from two camera frames, the optical flow: x and y on the
image plane by a Lucas–Kanade solve, and in the 3D flow z toward the camera by the depth velocity.
How the stages fit together is in [FLOW.md](FLOW.md); what is next, the motion's included, in
[PLAN.md](PLAN.md#the-motions-details). "Code" is what the files show; **(deduction)** marks what is inferred
from them, **(site fact)** what the operator saw.

How the flow works is in the module docstrings of `motion/lucas_kanade_flow_3d.py`,
`motion/lucas_kanade_flow.py` (the x/y solve) and `motion/depth_velocity.py` (z); the layer is
`MotionLayer` (`modules/render/layers/flow/motion_layer.py`) with `Motion3DLayer` and `Motion2DLayer`.

## When it measures

The motion layer measures once per camera frame: when `timestamp` differs from the one it last
measured. A frame is a picture with a capture time: a view exists, black, from the app's start, before
the camera delivers, and its `timestamp` is `None` until then. So `None` is no frame: the motion stays
at rest and counts nothing, and a `None` after frames (the camera lost) starts it over. The solve needs
two counted frames, so its first measurement is always a real frame against a real frame. A gap wider
than the expected frame interval is a dropped frame. A camera frame has
no other identity: `CameraFrame` and `Capture` carry a timestamp and no sequence number, because the
device's sequence numbers skip on the depth while every frame arrives (`modules/camera/oak4/OAK4.md`).
A source layer keeps the capture time of the frame it holds (`CaptureImageLayer`, `CaptureDepthLayer`,
`MaskSourceLayer` and `ForegroundSourceLayer` from `segmentation.Image.timestamp`), and a view (the
view stage, `ViewLayer`; hd_trio's `CenterCamLayer`, `CenterMaskLayer`, `CenterForegroundLayer`) forwards
the stamp of the frame it drew, so the app reads it off the layer it takes the texture from
(`HasCaptureTime`, `modules/render/layers/layer_base.py`); deep_flow hands the motion its capture's own. The interval a measurement divides by is in
[FLOW.md's two clocks](FLOW.md#inputs-and-outputs).

`DepthView(texture, view_span, depth_step)` (`modules/flow/motion/depth_view.py`) is the depth on
the flow's view: the texture, the metres one view width spans per metre of depth, and the uv
between two depth samples. The view stage makes it (`ViewOutput.depth`): the depth read into the
view's window across a surface, the `view_span` from the window and the camera's horizontal field of
view it takes at construction, and the `depth_step` from the aligned depth's grid and the window
(`modules/flow/view/view_window.py`).

## The settings

The settings of `LucasKanadeFlow3D`, in panel order, with their defaults and the reason for each.
The defaults are the best measured for deep_flow, in this order of importance:

- A: least z noise
- B: a clear difference between slow and fast motion, in x, y and z
- C: least smear

Every number is measured with `scripts/depth_velocity_compare.py` on the GPU, unless marked
**(deduction)**. The configuration: the flow at 448×800, deep_flow's preset, Neural Depth on an
OAK4, and two recordings at 0.65–1.5 m: `stillslowfast` (still, then an arm slow and fast) and
`standmoves` (arm sweeps, pushes toward the camera). The whole flow takes 2.0 ms per frame.

### lucas_kanade

| Setting                | Default | Range     | For | Why                                         |
|------------------------|---------|-----------|-----|---------------------------------------------|
| use_mask               | on      | on/off    | C   | no x/y spill onto the background            |
| x_gain, y_gain, z_gain | 1       | −10 to 10 | –   | 1 = true speed, 0 = axis off, < 0 inverted  |

The flow's velocity is in image widths per second times its gains. `z_gain` is the motion's one
knob on z's size; the depth velocity's settings set z's noise and lag, not its scale.

### solve (x/y)

| Setting      | Default | Range (step)        | For | Why                                    |
|--------------|---------|---------------------|-----|----------------------------------------|
| levels       | 3       | 1–5 (1)             | B   | follows fast motion                    |
| window       | 7×7     | 3×3–9×9             | B   | fast ÷ noise 4.4, against 3.7 at 5×5   |
| damping      | 0.03    | 0–0.1 (0.001)       | B   | takes the x/y noise, keeps slow motion |
| same_surface | on      | on/off              | C   | less smear, a hand far in front        |
| threshold    | 0.5     | 0–5 px/frame (0.05) | B   | the solve's jitter off                 |

- levels: 1 and 2 barely see motion at 448×800. Each level is solved at its own size (Bouguet's
  pyramid), so more levels stay clean. 5 levels reach about 32 px per frame: on `stillslowfast` with
  thresholds 0, fast x/y goes 0.51 → 1.30 and fast ÷ noise 3.6 → 5.3, for +0.33 ms; on `standmoves`
  the sweep's z drops 2.04 → 1.55 and the push keeps its z.
- window: 7×7 smears more than 5×5, but B ranks above C.
- same_surface works only where a no-depth band separates the hand from what is behind it, as with a
  hand held far out. Where a limb is about 20 cm before the body, Neural Depth ramps over about 30 px
  instead of stepping, and the patch's samples stay within the tolerance; see
  [PLAN.md](PLAN.md#the-motions-details). It costs +0.05 ms.

### depth_velocity (z)

| Setting   | Default      | Range (step)    | For | Why                                    |
|-----------|--------------|-----------------|-----|----------------------------------------|
| edge      | 0.1 m        | 0.05–0.3 (0.01) | A   | halves the edge spikes                 |
| window    | 7×7          | 1×1–9×9         | A   | the most samples averaged              |
| tolerance | 0.2 m        | 0.05–0.5 (0.01) | B   | keeps a push's z                       |
| smooth    | 50 ms        | 0–500 (1)       | A   | at 0 the still z noise is 15× higher   |
| xy_smooth | off          | on/off          | B   | x/y without extra lag                  |
| catch_up  | 8 ms         | 0–50 (1)        | B   | fast z 75% → 96%, same still noise     |
| dead_zone | 0.02 m       | 0–0.05 (0.001)  | A   | takes the depth's wander               |
| threshold | 1.0 px/frame | 0–5 (0.05)      | A   | with the dead zone, still z noise is 0 |

- tolerance at 0.05: a push loses 77% of its z, and the track restarts 6× as often while moving.
- smooth: lower lags less; keep it as low as looks right. The bridges read z as they read x/y, so
  this is z's one time constant; what it does to them, and the 200 ms advised for them, is in
  [BRIDGE_3D.md's Measured](BRIDGE_3D.md#measured).
- xy_smooth puts x/y through the depth velocity's alpha–beta filter, carried along the motion, with
  z's `smooth`. It is the only x/y state kept from frame to frame: the solve starts each frame from
  no motion. The motion's reset clears it.

Beside the velocity the 3D motion hands out where the body is now in the box's depth, `z_position`
(`ZPosition`): from the depth track's disparity, (depth − near) / (far − near) over the player's depth
range, the depth source's `near` and `far` (`DepthView`), 0 front to 1 back, times its weight in red and
the weight in green (RG16F). The weight is the depth mask's, where the depth is the player's (the depth's
own cut on the view, `MotionInput.depth_mask`), times the mask's with `use_mask` (each 1 without), and 0
without depth: the cut decides where the player's depth is, so the z position is held to 0..1 where the
track lags past the range. The depth mask gates z the same way, always: with `use_mask` the mask gates
x, y and z, and z also takes the depth mask, so z exists only where the depth itself is the player's, never
on an edge the mask's guided filter moved onto the wall's depth.
z stays in view widths per second: the bridge takes the length of x, y and z, so z in a share of the
range would mix units there. What the bridge and the fluid do with it is in
[BRIDGE_3D.md](BRIDGE_3D.md#the-z-positions).

## The guard

`guard`, beside `reset` on the 3D motion's panel, on by default: the motion's output and its two
memories set a non-finite value to 0 and count it into the motion's `FiniteGuard`, logged as
`motion: … non-finite <output> values` ([FLUID_3D.md's guard](FLUID_3D.md#the-guard) has how):

| Output           | Where                                 | Holds                                          |
|------------------|---------------------------------------|------------------------------------------------|
| `velocity`       | `compose_velocity.frag`               | the motion's output, x/y/z in widths/s         |
| `displacement`   | `displacement_filter.frag`            | the x/y filter's memory, with `xy_smooth`      |
| `depth_velocity` | `depth_velocity_shader.frag`          | z's memory: the followed disparity, its rate and anchor (`track`) |

Measured in a hidden window at 448 × 800, 40 frames at 25 fps: a NaN mask over 200 × 200 px gave
1 000 000 non-finite velocity values in the first second, zeroed, the velocity finite after; the x/y
filter's memory filled with NaN, 358 400 displacement values once, every pixel, and the filter went on
from 0. z's memory filled with NaN was counted 0 times and the velocity stayed finite: the carry takes
a track's texel only within `tolerance` of the measurement, a comparison a NaN fails, so the track
starts over and the NaN is gone without a non-finite value written.

## Debugging a solve

With the environment variable `POSER_SOLVE_CAPTURE` set to a folder, a solve that comes out wrong (a
non-finite value, or a step past half the image a frame) is saved there with the two luminance frames
at each mip level and the depth as they were just before it ran (`motion/solve_capture.py`), up to three
a run. `scripts/solve_replay.py` reruns a capture through the GPU solve in a hidden window and the
numpy reference: a capture the solve gets right there and wrong in the app points at the app's state,
not at the solve. `gl_state()` (`modules/gl`) returns the GL state a pass runs with.

## Constraints

- The coarsest level starts from no motion inside the shader (`has_estimate` off), never from a
  cleared buffer: a clear handed a temporary array for its color sometimes left garbage at startup,
  the bit pattern of 100 or 1.75 in one channel, 7 of 8 starts, and the whole pyramid took it as the
  start, every x/y arrow the same ("all arrows up"); with the `xy_smooth` filter it stuck until a reset.
  Every clear to 0 now passes no memory (`clear_value`, `modules/gl/texture.py`).

- The false z at an edge cannot be told from a push by time. Restarting the track where the depth
  changed past the tolerance over 2 frames cuts the edge z, and a push's z by 26%.
- It cannot be told by the depth's slope either. A weight on the slope cuts edges and pushes alike,
  by about 10%.
- Sharpening the depth at the image's edges does not separate them. A luminance-guided joint bilateral
  filter on the depth lowers push and sweep z alike, by 7%, for +0.14 ms.
- A forward–backward check on x/y drops half the fast motion.
- Iterations per level do less than one more level, at more GPU time.
- Each level is solved at its own size. Solving every full-size pixel at a coarse level puts each
  window a fraction of a texel off its neighbor's, and where a window is badly conditioned the solve
  flips from pixel to pixel in rows and columns: streaks from 4 levels up. A median between levels
  also removes them, at +0.35 ms or more, so there is none.
- The exposure is the camera's to settle, not the solve's: the solve matches no brightness between
  frames.

