# The motion

Every quantity and setting here is in a meaningful unit ([FLOW.md](FLOW.md#the-2d-and-3d-flow-systems)).

The motion stage measures a velocity field from two camera frames, the optical flow: x and y on the
image plane by a Lucas–Kanade solve, and in the 3D flow z toward the camera by the depth velocity.
How the stages fit together is in [FLOW.md](FLOW.md); what is next, the motion's included, in
[FLOW_PLAN.md](FLOW_PLAN.md#c-the-motion). "Code" is what the files show; **(deduction)** marks what is inferred
from them, **(site fact)** what the operator saw.

The x/y solve is `LucasKanadeFlow` (`motion/lucas_kanade_flow.py`), z the depth velocity
(`DepthVelocity`, `motion/depth_velocity.py`), and `LucasKanadeFlow3D` (`motion/lucas_kanade_flow_3d.py`)
composes them; the layer is `MotionLayer` (`modules/render/modules/flow/motion_layer.py`) with `Motion3DLayer`
and `Motion2DLayer`.

## The goal

The player's velocity per pixel, from two frames: x, y and z in widths/s, with as little noise as possible
while it keeps as much of the real motion as possible. What one frame shows is the view's; what
to trust over time is the motion's ([FLOW.md](FLOW.md#the-goals)).

## How it measures

### x/y: the Lucas–Kanade solve

Each pixel's shift between two luminance frames, from a window of gradients, coarse to fine (Bouguet's
pyramidal Lucas–Kanade): each level is solved at its own size from the coarser level's estimate, read
linearly, with the previous frame warped by it ([Constraints](#constraints) has why its own size). Out
comes the displacement in uv per frame, `threshold` (px per frame) taken off its length, as below it is
the solve's jitter. With a depth and `same_surface`, a window counts only its samples within `tolerance`
of its center's depth. The settings in pixels go with the flow's resolution: a level doubles the motion
it follows, about 2 × 2^(levels − 1) px per frame at full size.

### z: the depth velocity

One depth sample is mostly noise from frame to frame, and the change between two frames doubles it, so z
is tracked, not differenced: each pixel holds a disparity and its rate, carried along the x/y
displacement, and each measurement only corrects them (an alpha–beta filter) over a time that is `smooth` at
rest and shortens as the image moves. The parts:

| Part        | How                                                                           | Why                                                           |
|-------------|-------------------------------------------------------------------------------|---------------------------------------------------------------|
| disparity   | the track holds 1 / depth                                                     | stereo noise is about constant in disparity                   |
| `tolerance` | metres: what tells two surfaces apart                                         | a hand 10 cm before the chest is 10 cm at any distance        |
| measurement | the mean disparity of the window's samples within `tolerance`, by their share | a thin window at a silhouette corrects little                 |
| carry       | from where the pixel was a displacement back, same surface only               | a sliding surface keeps its depth; nothing crosses an edge    |
| restart     | a miss past `tolerance`, or faster than `max_speed`                           | a limb passing; a sloped edge sliding off adds up to 9–11 m/s |
| the time    | `smooth` over 1 + image speed / `halving_speed`, the 1€ filter's rule          | quiet at rest, little lag in a motion; x/y carries little noise |
| `dead_zone` | an anchor: z is 0 within it, past it z is whole; 0 by default                 | a surface's depth wanders by a few mm, unsmoothed             |
| step filter | no depth where 3 × 3 samples span a step past `edge`                          | silhouette jitter; a band a depth sample wide reads no z      |

The time is the 1€ filter's (Casiez, Roussel, Vogel 2012): the image speed is the largest displacement of the
3 × 3 texels with `xy.threshold` taken off, so x/y's noise at rest opens nothing, and with `smooth` above 0 the
carry takes each texel's depth a frame on by its rate. The anchor follows z past the dead zone, so a motion loses
only its first `dead_zone`; with z smoothed it costs z and lag and adds no quiet, so it is 0 by default. The
step filter is `DepthStepFilter`, the window's test of "another surface".
`tolerance` is no limit on speed: 0.2 m a frame is 5 m/s at 25 fps. z comes in view widths per frame, over
the metres one view width spans at the depth (`view_span` per metre), and `threshold` comes off it. With
`smooth` 0 and a full window it is the plain change between two frames.

### The 3D velocity

`LucasKanadeFlow3D` turns the displacement and z, both view widths per frame, into one velocity in widths
per second, times `x_gain`, `y_gain`, `z_gain`, gated as below. The tracks are carried by the displacement as
solved, before `xy.threshold`: a slide too slow for the output still moves them, or on a slope it would read
as z. With `xy_smooth`, x/y goes through z's filter over z's `smooth`, so both lag alike: a displacement is a step
of position, so with a steady frame interval and a full z window x/y answers a change of speed exactly as z does. The settings in flow
pixels go with the flow's resolution; those in metres and milliseconds do not.

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
`SegmentationMaskLayer` and `SegmentationForegroundLayer` from `segmentation.Image.timestamp`), and a view (the
view stage, `ViewLayer`; hd_trio's `CenterCamLayer`, `CenterMaskLayer`, `CenterForegroundLayer`) forwards
the stamp of the frame it drew, so the app reads it off the layer it takes the texture from
(`HasCaptureTime`, `modules/render/layer_base.py`); deep_flow hands the motion its capture's own. The interval a measurement divides by is in
[FLOW.md's two clocks](FLOW.md#inputs-and-outputs).

`DepthView(texture, view_span, depth_step)` (`modules/flow/motion/depth_view.py`) is the depth on
the flow's view: the texture, the metres one view width spans per metre of depth, and the uv
between two depth samples. The view stage makes it (`ViewOutput.depth`): the depth read into the
view's crop across a surface, the `view_span` from the crop and the camera's horizontal field of
view it takes at construction, and the `depth_step` from the aligned depth's grid and the crop
(`modules/flow/view/view_window.py`).

## The settings

The settings of `LucasKanadeFlow3D`, in panel order, with their defaults and the reason for each.
The defaults are the best measured for deep_flow, in this order of importance:

- A: least z noise
- B: a clear difference between slow and fast motion, in x, y and z
- C: least smear

Every number is measured with `scripts/depth_velocity_capture.py` and `depth_velocity_compare.py` on the GPU,
unless marked **(deduction)**, 2026-11-16 and 17, commits fdf8f8de to 73861281: the flow at 448×800, the preset
then, Neural Depth on an OAK4, every camera frame (50 a second), each part against its neutral value with the
parts after it neutral, on the motion's raw output (the thresholds, `smooth` and `dead_zone` at 0) unless a row
says. Three captures a recording agree within 0.1; the rows here are their mean. Noise is the still stretch's
p99 per pixel inside the player (12 px and more from the mask's edge) and at its edge (within 6 px), in px per
frame; kept is a side's share of the raw output's motion over the moves. The whole motion takes about 2 ms a
frame. The recordings, at 0.65–1.5 m, their stretches in `segments.json` in each recording's folder:

| Recording                 | What the player does                                   | Distance  | Still       |
|---------------------------|--------------------------------------------------------|-----------|-------------|
| `20261112-125832_measure` | still, then an arm sideways, slow then fast            | 1.2 m     | 1–5 s       |
| `20261112-124918_hands`   | hands pushed toward the camera and back, close         | 0.5–0.8 m | 0.8 s, not quite |
| `20261112-130012_edges`   | arm waves and hands at the silhouette, a turn           | 1.2 m     | 2 s         |
| `20261112-130105_show`    | free movement, walking in and out                       | 1.0–1.6 m | 0.9 s       |

The measure recording's arm moves sideways, so it says little about z kept or its lag; those come from hands.

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
| window       | 7×7     | 3×3–9×9             | B   | follows more of the motion than 3×3 or 5×5 |
| damping      | 0.03    | 0–0.1 (0.001)       | B   | x/y's noise a quarter of undamped      |
| same_surface | on      | on/off              | C   | no measured effect                     |
| threshold    | 0.5     | 0–5 px/frame (0.05) | B   | the still x/y noise 0                  |

The x/y solve, on the raw output, with `levels` 4 (the preset) as the reference: x/y kept over the moves, and
x/y's still noise inside (measure / hands / edges / show):

| Change                 | x/y kept                    | x/y still noise              |
|------------------------|-----------------------------|------------------------------|
| none                   | 1                           | 0.22 / 0.96 / 0.27 / 0.42    |
| `levels` 1             | 0.17 / 0.17 / 0.30 / 0.12   | 0.09 / 0.22 / 0.14 / 0.22    |
| `levels` 3             | 0.67 / 0.68 (measure / hands) | 0.19 / 0.65                |
| `levels` 5             | 1.22 / 1.27                 | 0.25 / 1.09                  |
| `window` 3×3           | 0.57 / 0.56 / 0.73 / 0.53   | 0.17 / 0.46 / 0.23 / 0.36    |
| `window` 5×5           | 0.85 / 0.82                 | 0.20 / 0.84                  |
| `window` 9×9           | 1.08 / 1.13                 | 0.22 / 1.01                  |
| `damping` 0            | 1.39 / 1.44 / 1.09 / 1.36   | 1.23 / 1.76 / 1.11 / 1.10    |
| `damping` 0.015        | 1.11 / 1.09                 | 0.29 / 1.00                  |
| `damping` 0.06         | 0.86 / 0.89                 | 0.16 / 0.95                  |
| `same_surface` off     | 1.00 / 1.02 / 1.00 / 1.00   | the same                     |
| `threshold` 0.5        | 0.81 / 0.83 / 0.73 / 0.92   | 0                            |

The x/y solve leaves z as it is, but a solve that follows less reads part of a slide as z (`levels` 1: z 22%
larger on show). A level doubles the motion the solve follows, about 2 × 2^(levels − 1) px per frame; `levels`
5 follows about a fifth more than 4. `same_surface` works only where a no-depth band separates the hand from
what is behind it; the step filter and the mask leave little for it to separate **(deduction)**.

### depth_velocity (z)

| Setting   | Default      | Range (step)    | For | Why                                    |
|-----------|--------------|-----------------|-----|----------------------------------------|
| edge      | 0.1 m        | 0.05–0.3 (0.01) | A   | the edge noise 40% below 0.3 m          |
| window    | 7×7          | 1×1–9×9         | A   | the edge noise a quarter of 1×1         |
| tolerance | 0.2 m        | 0.05–0.5 (0.01) | B   | no measured effect against 0.5 m        |
| smooth        | 100 ms       | 0–500 (1)       | A   | the still z noise 31× below the raw    |
| halving_speed | 0.5 px/frame | 0.05–5 (0.05)   | B   | a push lags 11 ms, keeps 87% of its z  |
| xy_smooth     | off          | on/off          | B   | x/y without extra lag                  |
| dead_zone     | 0 m          | 0–0.05 (0.001)  | B   | with z smoothed it costs z and lag     |
| threshold     | 0.5 px/frame | 0–5 (0.05)      | A   | the still z noise 0                    |

- edge, window and tolerance on the raw output, z's still noise inside / at the edge (measure / hands / edges /
  show):

  | Change              | Inside                       | At the edge                  |
  |---------------------|------------------------------|------------------------------|
  | none                | 12.9 / 30.4 / 11.8 / 14.1    | 6.0 / 10.1 / 5.9 / 6.1       |
  | `edge` 0.05 m       | 12.0 / 26.3 (measure / hands) | 4.4 / 5.5                   |
  | `edge` 0.2 m        | 14.0 / 33.6                  | 8.0 / 17.9                   |
  | `edge` 0.3 m        | 14.4 / 34.1 / 13.1 / 14.6    | 9.4 / 19.9 / 9.3 / 10.0      |
  | `window` 1×1        | 15.4 / 35.1 / 14.5 / 16.1    | 24.3 / 28.8 / 24.3 / 24.1    |
  | `window` 5×5        | 13.5 / 31.6                  | 9.5 / 13.4                   |
  | `window` 9×9        | 12.2 / 29.4                  | 4.7 / 9.3                    |
  | `tolerance` 0.5 m   | 13.0 / 30.7 / 11.9 / 14.1    | 6.1 / 12.0 / 6.0 / 6.1       |

  `edge` 0.05 m keeps 97% of hands' z; z's `window` changes how much a small part blurs, not z's size. A
  `tolerance` of 0.5 m against 0.2 kept a little more z with smoothing on show (0.47 against 0.42 with the
  catch-up then).
- smooth and halving_speed, measured 2026-11-17 on one capture of the hands recording (lag, z kept against
  the raw) and of the measure recording (the still noise inside / at the edge, px per frame), `dead_zone` 0,
  `tolerance` 0.5 m:

  | The time                                | Lag    | Kept | Still noise   |
  |-----------------------------------------|--------|------|---------------|
  | none (`smooth` 0)                       | 0 ms   | 1.00 | 13.0 / 6.1    |
  | `smooth` 25, the catch-up before it     | 14 ms  | 0.86 | 2.4 / 2.2     |
  | `smooth` 50, the catch-up before it     | 43 ms  | 0.83 | 0.97 / 1.2    |
  | `smooth` 50, halving 0.5                | 5 ms   | 0.88 | 0.96 / 1.2    |
  | `smooth` 100, halving 0.5               | 11 ms  | 0.87 | 0.42 / 0.56   |
  | `smooth` 100, halving 1                 | 42 ms  | 0.85 | 0.42 / 0.51   |
  | `smooth` 200, halving 0.5               | 56 ms  | 0.84 | 0.20 / 0.26   |
  | `smooth` 100, halving 0.5, `threshold` 0.5 | 11 ms | 0.83 | 0 / 0.07   |

  A low halving speed is what keeps the lag down; z's own speed beside the image's, tried, let the still noise
  up to 2 to 4 for no less lag. The bridges read z as they read x/y; what z's smoothing does to them is in
  [BRIDGE_3D.md's Measured](BRIDGE_3D.md#measured).
- xy_smooth puts x/y through the depth velocity's alpha–beta filter, carried along the motion, with
  z's `smooth` at rest. It is the only x/y state kept from frame to frame: the solve starts each frame from
  no motion. The motion's reset clears it.

Beside the velocity the 3D motion hands out where the body is now in the box's depth, `z_position`
(`ZPosition`): from the depth track's disparity, (depth − near) / (far − near) over the player's depth
range, the depth source's `near` and `far` (`DepthView`), 0 front to 1 back, times its weight in red and
the weight in green (RG16F). The weight is the mask's with `use_mask` (1 without), and 0 without depth;
the z position is held to 0..1 where the track lags past the range. With `use_mask` the mask weights x, y
and z alike. The view hands the player's depth wherever the mask is above 0 and none elsewhere
([VIEW.md](VIEW.md#the-depth)), so z never takes a depth from behind the player.
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

