# The motion

Every quantity and setting here is in a meaningful unit ([FLOW.md](FLOW.md#the-2d-and-3d-flow-systems)).

The motion stage measures a velocity field from two camera frames, the optical flow: x and y on the
image plane by a Lucas–Kanade solve, and in the 3D flow z toward the camera by the depth velocity.
How the stages fit together is in [FLOW.md](FLOW.md); what is open, the motion's included, in
[FLOW.md's Open](FLOW.md#open). "Code" is what the files show; **(deduction)** marks what is inferred
from them, **(site fact)** what the operator saw.

How the flow works is in the module docstrings of `motion/lucas_kanade_flow_3d.py`,
`motion/lucas_kanade_flow.py` (the x/y solve) and `motion/depth_velocity.py` (z); the layer is
`MotionLayer` (`modules/render/layers/flow/motion_layer.py`) with `Motion3DLayer` and `Motion2DLayer`.

## When it measures

The motion layer measures once per camera frame: when `timestamp` differs from the one it last
measured; a source without stamps (`None`) measures every tick. A gap wider than the expected
frame interval is a dropped frame. For its first 3 s after start the motion layer stays at rest
(`WARM_UP_TIME`): the camera's exposure settles then, and the Lucas–Kanade solve reads the
brightening as motion, x/y at full reach over the whole image **(site fact)**. A camera frame has
no other identity: `CameraFrame` and `Capture` carry a timestamp and no sequence number, because the
device's sequence numbers skip on the depth while every frame arrives (`modules/camera/oak4/OAK4.md`).
A source layer keeps the capture time of the frame it holds (`CaptureImageLayer`, `DepthSourceLayer`,
`MaskSourceLayer` and `ForegroundSourceLayer` from `segmentation.Image.timestamp`, `DepthMaskLayer`
from its depth), and a view (`CenterCamLayer`, `CenterMaskLayer`, `CenterForegroundLayer`,
`CenterDepthLayer`) takes its source layer and forwards the stamp of the frame it drew, so the app
reads it off the layer it takes the texture from (`HasCaptureTime`,
`modules/render/layers/layer_base.py`). The interval a measurement divides by is in
[FLOW.md's two clocks](FLOW.md#inputs-and-outputs).

`DepthView(texture, view_span, depth_step)` (`modules/flow/motion/depth_view.py`) is the depth on
the flow's view: the texture, the metres one view width spans per metre of depth, and the uv
between two depth samples. The 3D camera's layers make it: `DepthSourceLayer` the texture and its
`sample_size`, `CenterGeometry` the `view_span` from its crop and the camera's horizontal field of
view it takes at construction, and the `depth_step` from the sample size and its crop;
`CenterDepthLayer.depth_view` puts them together.

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

The flow's velocity is in image widths per second times its gains.

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
  [FLOW.md's Open](FLOW.md#open). It costs +0.05 ms.

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
  [BRIDGE.md's Measured](BRIDGE.md#measured).
- xy_smooth puts x/y through the depth velocity's alpha–beta filter, carried along the motion, with
  z's `smooth`. It is the only x/y state kept from frame to frame: the solve starts each frame from
  no motion. The motion's reset clears it.

## Constraints

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
  frames, and the warm-up keeps the start's ramp out.

