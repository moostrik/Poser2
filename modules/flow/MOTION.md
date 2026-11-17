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
the solve's jitter. The settings in pixels go with the flow's resolution: a level doubles the motion
it follows, about 2 × 2^(levels − 1) px per frame at full size.

### z: the depth velocity

One depth sample is mostly noise from frame to frame, and the change between two frames doubles it, so z
is tracked, not differenced: each pixel holds a disparity and its rate, carried along the x/y
displacement, and each measurement only corrects them (an alpha–beta filter) over a time that is `smooth` at
rest and shortens as x/y moves. The steps, in the order they run, each with its settings; the panel holds
them in that order, a line each for the surface, the smoothing and the cut:

| Step         | Settings                                  | What it does                                                         |
|--------------|-------------------------------------------|----------------------------------------------------------------------|
| step filter  | `edge`                                    | no depth where 3 × 3 samples span a step past `edge`, the silhouette's flicker |
| window       | `window`, `edge`                          | the mean disparity of the samples within `edge` of the pixel's depth |
| another surface | `max_speed`                            | the track starts over where the depth jumps or slides faster than a body moves |
| smoothing    | `smooth`, `xy_rest`, `xy_halving`         | the time the track follows over, long at rest, short as x/y moves   |
| dead zone    | `dead_zone`                               | z 0 while the depth stays within it of its anchor; 0 by default      |
| cut          | `threshold`                               | z below it is noise, taken off                                       |

The parts:

| Part        | How                                                                           | Why                                                           |
|-------------|-------------------------------------------------------------------------------|---------------------------------------------------------------|
| disparity   | the track holds 1 / depth                                                     | stereo noise is about constant in disparity                   |
| measurement | the mean disparity of the window's samples within `edge`, by their share      | a thin window at a silhouette corrects little                 |
| carry       | from where the pixel was a displacement back, the texels within `max_speed` × the frame's time of the measurement | a sliding surface keeps its depth; nothing crosses another surface |
| restart     | a miss past `max_speed` × the frame's time, or a filtered rate past `max_speed` | a limb passing; a sloped edge sliding off adds up to 9–11 m/s |
| the time    | `smooth` over 1 + x/y speed / `xy_halving`, the 1€ filter's rule              | quiet at rest, little lag in a motion; x/y carries little noise |
| `dead_zone` | an anchor: z is 0 within it, past it z is whole; 0 by default                 | a surface's depth wanders by a few mm, unsmoothed             |

Within a frame `edge` tells two surfaces apart, between frames `max_speed`: a body moves at most `max_speed` ×
the frame's time (12 cm at 50 fps), so a depth further from the prediction is another surface, and a filtered
rate past it is a surface sliding away (each step small against the prediction). 0 checks neither. The time is
the 1€ filter's (Casiez, Roussel, Vogel 2012), opened by x/y's speed, not z's own as in the original filter: the
x/y speed is the largest displacement of the 3 × 3 texels with `xy_rest` taken off, so x/y's noise at rest opens
nothing; 0 `xy_halving` is `smooth` throughout. With `smooth` above 0 the carry takes each texel's depth a frame
on by its rate. The anchor follows z past the dead zone, so a motion loses only its first `dead_zone`; with z
smoothed it costs z and lag and adds no quiet, so it is 0 by default. The step filter is `DepthStepFilter`. z
comes in view widths per frame, over the metres one view width spans at the depth (`view_span` per metre), and
`threshold` comes off it. With `smooth` 0 and a full window it is the plain change between two frames.

### The 3D velocity

`LucasKanadeFlow3D` turns the displacement and z, both view widths per frame, into one velocity in widths
per second, times `x_gain`, `y_gain`, `z_gain`, gated as below. The tracks are carried by the displacement as
solved, before `xy.threshold`: a slide too slow for the output still moves them, or on a slope it would read
as z. With `z_smooth`, x/y goes through z's filter over z's `smooth`, so both lag alike: a displacement is a step
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
| z_smooth     | off     | on/off              | B   | x/y without extra lag                  |
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
| `threshold` 0.5        | 0.81 / 0.83 / 0.73 / 0.92   | 0                            |

The x/y solve leaves z as it is, but a solve that follows less reads part of a slide as z (`levels` 1: z 22%
larger on show). A level doubles the motion the solve follows, about 2 × 2^(levels − 1) px per frame; `levels`
5 follows about a fifth more than 4.

### depth_velocity (z)

| Setting    | Default      | Range (step)    | For | Why                                    |
|------------|--------------|-----------------|-----|----------------------------------------|
| edge       | 0.1 m        | 0.05–0.3 (0.01) | A   | the edge noise 40% below 0.3 m         |
| window     | 7×7          | 1×1–9×9         | A   | the edge noise a quarter of 1×1        |
| max_speed  | 6 m/s        | 0–20 (0.5)      | A   | without it a sliding arm's z 2.5×      |
| smooth     | 100 ms       | 0–500 (1)       | A   | the still z noise 31× below the raw    |
| xy_rest    | 0.5 px/frame | 0–5 (0.05)      | A   | above x/y's noise at rest              |
| xy_halving | 0.5 px/frame | 0–5 (0.05)      | B   | a push lags 11 ms, keeps 87% of its z  |
| dead_zone  | 0 m          | 0–0.05 (0.001)  | B   | with z smoothed it costs z and lag     |
| threshold  | 0.5 px/frame | 0–5 (0.05)      | A   | the still z noise 0                    |

- edge and window on the raw output, with `tolerance` (gone since) at 0.2 m, z's still noise inside / at the edge (measure / hands / edges /
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

  `edge` 0.05 m keeps 97% of hands' z; z's `window` changes how much a small part blurs, not z's size.
  `tolerance` (0.2 m, a second "another surface" step for the window, the carry and the restart) against 0.5 m
  changed nothing on four recordings, as the step filter has already taken the jump out **(deduction)**; it
  went for `edge` within a frame and `max_speed` between frames, and x/y's `same_surface`, which read it, with
  it (off and on the same on four recordings).
- smooth and the halving speed, measured 2026-11-17 on one capture of the hands recording (lag, z kept against
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
- z_smooth (in the x/y panel, between `damping` and `threshold`) puts x/y through the depth velocity's alpha–beta filter, carried along the motion, over z's time,
  the 1€ rule's, worked out per pixel from the same x/y speed. It is the only x/y state kept from frame to frame:
  the solve starts each frame from no motion. The motion's reset clears it.
- C1's change (`tolerance` and `same_surface` gone, `max_speed` before and after the filter, `xy_rest` in place
  of `xy.threshold`, `xy_smooth`, now `xy.z_smooth`, over z's time), 2026-11-17, one capture of hands and of measure, the defaults
  with `dead_zone` 0, before and after:

  | Row                  | z still noise, measure | z still noise, hands | z kept / lag, hands | x/y noise, hands / kept, measure | restarts at rest, measure |
  |----------------------|------------------------|----------------------|---------------------|----------------------------------|---------------------------|
  | defaults, before     | 0 / 0.06               | 4.1 / 2.3            | 0.83 / 11 ms        | 0.50 / 0.81                      | 0.06%                     |
  | defaults, after      | 0 / 0.05               | 3.7 / 2.0            | 0.83 / 11 ms        | 0.50 / 0.81                      | 0.14%                     |
  | `threshold` 0, before | 0.41 / 0.55           | 4.6 / 2.8            | 0.86 / 11 ms        |                                  |                           |
  | `threshold` 0, after | 0.40 / 0.54            | 4.2 / 2.5            | 0.87 / 11 ms        |                                  |                           |
  | `z_smooth` on, before | 0 / 0.06             | 4.1 / 2.3            | 0.83 / 11 ms        | 8.3 (lag 218 ms) / 0.42          |                           |
  | `z_smooth` on, after  | 0 / 0.05              | 3.7 / 2.0            | 0.83 / 11 ms        | 5.4 (lag 26 ms) / 0.89           |                           |
  | `xy_halving` 0, after | 0 / 0                 | 3.2 / 2.2            | 0.40 / 202 ms       |                                  |                           |
  | `max_speed` 0, after | 0 / 0.07               | 4.9 / 3.0            | 0.84 / 10 ms        |                                  | 0                         |

  Hands' still noise fell by a tenth with the window's samples within `edge` (0.1 m) where `tolerance` (0.2 m)
  let more of the surface behind in; its still stretch is a hand not quite still. Without `max_speed`'s checks the
  sideways arm on measure reads 2.5 times the z (0.97 against 0.39 fast), a sliding surface read as z.

- The values, a result, 2026-11-17, one run on one capture of each recording (three captures agreed within 0.1 before): each row one change from the defaults (`levels` 4, the windows 7×7, `damping` 0.03,
  x/y's `threshold` 0.5, `edge` 0.1 m, `max_speed` 6 m/s, `smooth` 100 ms, `xy_rest` 0.5, `xy_halving` 0.5,
  `dead_zone` 0, z's `threshold` 0.5), on one capture of each recording, the surfaces in the mask that are not
  the player left out (`exclude` in each recording's `segments.json`: the desk and the curtain in hands, the
  sloping ceiling and the window at 1.6 m in the others). Still noise inside / at the edge and x/y's in px per
  frame, kept and lag against the raw row; measure's and edges' arm moves sideways, so their z kept and lag say
  little, hands' and show's are the ones for z. The read: `levels` 5 takes hands' z lag from 17 to 7 ms with
  the same noise and x/y following 9 to 22% more; `xy_halving` 0.25 takes it to 6 ms for a little more noise at
  the edge; `max_speed` 8 restarts less with nothing worse; `smooth` 50 is not quiet, 200 lags; `edge` 0.05
  cleans the edge but costs z and lag; `z_smooth` on makes x/y noisy at rest on hands (6 against 0.7), not
  explained yet. The two best were each measured alone, not together.

  measure (`measure_1`, 2026-11-17 23:34, commit 4f72f124+):

  | Row                        | z still noise | z kept        | z lag         | x/y kept      | x/y still noise | restarts at rest | ms            |
  |----------------------------|---------------|---------------|---------------|---------------|-----------------|------------------|---------------|
  | raw                        | 12.34 / 5.51  | 1.00          | 0 ms          | 1.00          | 0.22 / 1.52     | 0.00%            | 1.2           |
  | defaults                   | 0.00 / 0.09   | 0.24          | 3 ms          | 0.80          | 0.00 / 1.03     | 0.14%            | 1.4           |
  | smooth 50 halving 0.25     | 0.35 / 0.84   | 0.42          | 1 ms          | 0.80          | 0.00 / 1.03     | 0.14%            | 1.5           |
  | smooth 50 halving 0.5      | 0.35 / 0.67   | 0.34          | 2 ms          | 0.80          | 0.00 / 1.03     | 0.14%            | 1.4           |
  | smooth 50 halving 1        | 0.35 / 0.55   | 0.27          | 3 ms          | 0.80          | 0.00 / 1.03     | 0.14%            | 1.2           |
  | smooth 100 halving 0.25    | 0.00 / 0.24   | 0.32          | 2 ms          | 0.80          | 0.00 / 1.03     | 0.14%            | 1.7           |
  | smooth 100 halving 1       | 0.00 / 0.02   | 0.17          | 5 ms          | 0.80          | 0.00 / 1.03     | 0.14%            | 1.2           |
  | smooth 200 halving 0.25    | 0.00 / 0.00   | 0.23          | 3 ms          | 0.80          | 0.00 / 1.03     | 0.14%            | 1.5           |
  | smooth 200 halving 0.5     | 0.00 / 0.00   | 0.16          | 5 ms          | 0.80          | 0.00 / 1.03     | 0.14%            | 1.4           |
  | smooth 200 halving 1       | 0.00 / 0.00   | 0.11          | 9 ms          | 0.80          | 0.00 / 1.03     | 0.14%            | 1.2           |
  | xy_rest 0.25               | 0.00 / 0.17   | 0.25          | 3 ms          | 0.80          | 0.00 / 1.03     | 0.14%            | 1.5           |
  | xy_rest 1                  | 0.00 / 0.01   | 0.23          | 3 ms          | 0.80          | 0.00 / 1.03     | 0.14%            | 1.2           |
  | z threshold 0.25           | 0.12 / 0.34   | 0.25          | 3 ms          | 0.80          | 0.00 / 1.03     | 0.14%            | 1.5           |
  | z threshold 1              | 0.00 / 0.00   | 0.23          | 2 ms          | 0.80          | 0.00 / 1.03     | 0.14%            | 1.4           |
  | z threshold 2              | 0.00 / 0.00   | 0.19          | 2 ms          | 0.80          | 0.00 / 1.03     | 0.14%            | 1.2           |
  | edge 0.05                  | 0.00 / 0.00   | 0.23          | 3 ms          | 0.80          | 0.00 / 1.03     | 0.09%            | 1.5           |
  | z window 9x9               | 0.00 / 0.00   | 0.24          | 3 ms          | 0.80          | 0.00 / 1.03     | 0.15%            | 1.2           |
  | max_speed 4                | 0.00 / 0.06   | 0.24          | 3 ms          | 0.80          | 0.00 / 1.03     | 0.22%            | 1.5           |
  | max_speed 8                | 0.00 / 0.10   | 0.24          | 3 ms          | 0.80          | 0.00 / 1.03     | 0.09%            | 1.4           |
  | levels 5                   | 0.00 / 0.09   | 0.26          | 2 ms          | 0.98          | 0.00 / 1.03     | 0.14%            | 1.2           |
  | xy window 9x9              | 0.00 / 0.09   | 0.25          | 2 ms          | 0.87          | 0.00 / 1.04     | 0.14%            | 1.7           |
  | damping 0.02               | 0.00 / 0.10   | 0.25          | 3 ms          | 0.87          | 0.00 / 1.04     | 0.14%            | 1.2           |
  | damping 0.045              | 0.00 / 0.09   | 0.23          | 3 ms          | 0.73          | 0.00 / 1.00     | 0.14%            | 1.5           |
  | xy threshold 0.25          | 0.00 / 0.09   | 0.24          | 3 ms          | 0.89          | 0.00 / 1.27     | 0.14%            | 1.4           |
  | xy threshold 0.4           | 0.00 / 0.09   | 0.24          | 3 ms          | 0.83          | 0.00 / 1.12     | 0.14%            | 1.2           |
  | z_smooth on                | 0.00 / 0.09   | 0.24          | 3 ms          | 0.82          | 0.00 / 1.03     | 0.14%            | 1.7           |

  hands (`hands_1`, 2026-11-17 23:28, commit 4f72f124+):

  | Row                        | z still noise | z kept        | z lag         | x/y kept      | x/y still noise | restarts at rest | ms            |
  |----------------------------|---------------|---------------|---------------|---------------|-----------------|------------------|---------------|
  | raw                        | 16.49 / 7.76  | 1.00          | 0 ms          | 1.00          | 1.17 / 1.78     | 0.00%            | 1.3           |
  | defaults                   | 4.19 / 2.22   | 0.83          | 17 ms         | 0.84          | 0.67 / 1.29     | 0.00%            | 1.5           |
  | smooth 50 halving 0.25     | 2.57 / 3.60   | 0.85          | 4 ms          | 0.84          | 0.67 / 1.29     | 0.00%            | 1.6           |
  | smooth 50 halving 0.5      | 2.03 / 2.87   | 0.84          | 5 ms          | 0.84          | 0.67 / 1.29     | 0.00%            | 1.5           |
  | smooth 50 halving 1        | 1.77 / 2.31   | 0.83          | 9 ms          | 0.84          | 0.67 / 1.29     | 0.00%            | 1.4           |
  | smooth 100 halving 0.25    | 4.16 / 2.95   | 0.85          | 6 ms          | 0.84          | 0.67 / 1.29     | 0.00%            | 1.7           |
  | smooth 100 halving 1       | 4.11 / 1.64   | 0.81          | 45 ms         | 0.84          | 0.67 / 1.29     | 0.00%            | 1.5           |
  | smooth 200 halving 0.25    | 7.12 / 2.46   | 0.83          | 25 ms         | 0.84          | 0.67 / 1.29     | 0.01%            | 1.6           |
  | smooth 200 halving 0.5     | 7.14 / 1.70   | 0.80          | 56 ms         | 0.84          | 0.67 / 1.29     | 0.01%            | 1.5           |
  | smooth 200 halving 1       | 7.20 / 2.52   | 0.74          | 88 ms         | 0.84          | 0.67 / 1.29     | 0.00%            | 1.5           |
  | xy_rest 0.25               | 3.90 / 2.25   | 0.83          | 11 ms         | 0.84          | 0.67 / 1.29     | 0.00%            | 1.6           |
  | xy_rest 1                  | 4.45 / 2.07   | 0.83          | 29 ms         | 0.84          | 0.67 / 1.29     | 0.00%            | 1.5           |
  | z threshold 0.25           | 4.44 / 2.47   | 0.85          | 17 ms         | 0.84          | 0.67 / 1.29     | 0.00%            | 1.7           |
  | z threshold 1              | 3.69 / 1.73   | 0.80          | 16 ms         | 0.84          | 0.67 / 1.29     | 0.00%            | 1.5           |
  | z threshold 2              | 2.70 / 0.76   | 0.74          | 15 ms         | 0.84          | 0.67 / 1.29     | 0.00%            | 1.4           |
  | edge 0.05                  | 4.17 / 1.02   | 0.81          | 22 ms         | 0.84          | 0.67 / 1.29     | 0.00%            | 1.6           |
  | z window 9x9               | 4.10 / 2.23   | 0.83          | 19 ms         | 0.84          | 0.67 / 1.29     | 0.00%            | 1.5           |
  | max_speed 4                | 4.13 / 2.22   | 0.84          | 18 ms         | 0.84          | 0.67 / 1.29     | 0.02%            | 1.7           |
  | max_speed 8                | 4.21 / 2.22   | 0.83          | 17 ms         | 0.84          | 0.67 / 1.29     | 0.00%            | 1.5           |
  | levels 5                   | 3.16 / 2.20   | 0.83          | 7 ms          | 1.09          | 0.76 / 1.30     | 0.00%            | 1.5           |
  | xy window 9x9              | 3.78 / 2.09   | 0.83          | 9 ms          | 0.95          | 0.67 / 1.42     | 0.00%            | 1.8           |
  | damping 0.02               | 4.20 / 2.19   | 0.83          | 11 ms         | 0.88          | 0.69 / 1.36     | 0.00%            | 1.5           |
  | damping 0.045              | 4.23 / 2.20   | 0.83          | 24 ms         | 0.78          | 0.64 / 1.19     | 0.00%            | 1.6           |
  | xy threshold 0.25          | 4.19 / 2.22   | 0.83          | 17 ms         | 0.90          | 0.92 / 1.54     | 0.00%            | 1.5           |
  | xy threshold 0.4           | 4.19 / 2.22   | 0.83          | 17 ms         | 0.86          | 0.77 / 1.39     | 0.00%            | 1.4           |
  | z_smooth on                | 4.19 / 2.22   | 0.83          | 17 ms         | 1.02          | 6.11 / 2.52     | 0.00%            | 1.8           |

  edges (`edges_1`, 2026-11-17 23:45, commit 4f72f124+):

  | Row                        | z still noise | z kept        | z lag         | x/y kept      | x/y still noise | restarts at rest | ms            |
  |----------------------------|---------------|---------------|---------------|---------------|-----------------|------------------|---------------|
  | raw                        | 11.15 / 5.22  | 1.00          | 0 ms          | 1.00          | 0.27 / 0.25     | 0.00%            | 0.6           |
  | defaults                   | 0.05 / 0.10   | 0.08          | 7 ms          | 0.73          | 0.00 / 0.00     | 0.10%            | 0.6           |
  | smooth 50 halving 0.25     | 0.38 / 0.35   | 0.23          | 1 ms          | 0.73          | 0.00 / 0.00     | 0.10%            | 0.7           |
  | smooth 50 halving 0.5      | 0.38 / 0.35   | 0.16          | 3 ms          | 0.73          | 0.00 / 0.00     | 0.10%            | 0.6           |
  | smooth 50 halving 1        | 0.38 / 0.35   | 0.10          | 6 ms          | 0.73          | 0.00 / 0.00     | 0.10%            | 0.6           |
  | smooth 100 halving 0.25    | 0.05 / 0.11   | 0.14          | 3 ms          | 0.73          | 0.00 / 0.00     | 0.10%            | 0.7           |
  | smooth 100 halving 1       | 0.05 / 0.10   | 0.05          | 15 ms         | 0.73          | 0.00 / 0.00     | 0.10%            | 0.5           |
  | smooth 200 halving 0.25    | 0.00 / 0.01   | 0.07          | 7 ms          | 0.73          | 0.00 / 0.00     | 0.10%            | 0.7           |
  | smooth 200 halving 0.5     | 0.00 / 0.00   | 0.04          | 16 ms         | 0.73          | 0.00 / 0.00     | 0.10%            | 0.6           |
  | smooth 200 halving 1       | 0.00 / 0.00   | 0.03          | 154 ms        | 0.73          | 0.00 / 0.00     | 0.10%            | 0.6           |
  | xy_rest 0.25               | 0.05 / 0.11   | 0.09          | 6 ms          | 0.73          | 0.00 / 0.00     | 0.10%            | 0.6           |
  | xy_rest 1                  | 0.05 / 0.11   | 0.06          | 7 ms          | 0.73          | 0.00 / 0.00     | 0.10%            | 0.5           |
  | z threshold 0.25           | 0.30 / 0.35   | 0.09          | 7 ms          | 0.73          | 0.00 / 0.00     | 0.10%            | 0.7           |
  | z threshold 1              | 0.00 / 0.00   | 0.06          | 6 ms          | 0.73          | 0.00 / 0.00     | 0.10%            | 0.6           |
  | z threshold 2              | 0.00 / 0.00   | 0.03          | 5 ms          | 0.73          | 0.00 / 0.00     | 0.10%            | 0.6           |
  | edge 0.05                  | 0.04 / 0.02   | 0.07          | 7 ms          | 0.73          | 0.00 / 0.00     | 0.05%            | 0.6           |
  | z window 9x9               | 0.02 / 0.05   | 0.08          | 7 ms          | 0.73          | 0.00 / 0.00     | 0.11%            | 0.5           |
  | max_speed 4                | 0.05 / 0.11   | 0.08          | 7 ms          | 0.73          | 0.00 / 0.00     | 0.16%            | 0.7           |
  | max_speed 8                | 0.05 / 0.11   | 0.08          | 7 ms          | 0.73          | 0.00 / 0.00     | 0.06%            | 0.6           |
  | levels 5                   | 0.05 / 0.10   | 0.08          | 6 ms          | 0.76          | 0.00 / 0.00     | 0.10%            | 0.6           |
  | xy window 9x9              | 0.05 / 0.10   | 0.08          | 6 ms          | 0.74          | 0.00 / 0.00     | 0.10%            | 0.7           |
  | damping 0.02               | 0.05 / 0.10   | 0.08          | 6 ms          | 0.74          | 0.00 / 0.00     | 0.10%            | 0.5           |
  | damping 0.045              | 0.05 / 0.11   | 0.08          | 7 ms          | 0.70          | 0.00 / 0.00     | 0.10%            | 0.7           |
  | xy threshold 0.25          | 0.05 / 0.10   | 0.08          | 7 ms          | 0.85          | 0.02 / 0.00     | 0.10%            | 0.6           |
  | xy threshold 0.4           | 0.05 / 0.10   | 0.08          | 7 ms          | 0.77          | 0.00 / 0.00     | 0.10%            | 0.6           |
  | z_smooth on                | 0.05 / 0.10   | 0.08          | 7 ms          | 0.74          | 0.00 / 0.00     | 0.10%            | 0.6           |

  show (`show_1`, 2026-11-17 23:46, commit 4f72f124+):

  | Row                        | z still noise | z kept        | z lag         | x/y kept      | x/y still noise | restarts at rest | ms            |
  |----------------------------|---------------|---------------|---------------|---------------|-----------------|------------------|---------------|
  | raw                        | 13.33 / 5.24  | 1.00          | 0 ms          | 1.00          | 0.43 / 0.68     | 0.00%            | 0.6           |
  | defaults                   | 0.71 / 0.74   | 0.59          | 2 ms          | 0.92          | 0.00 / 0.19     | 0.01%            | 0.6           |
  | smooth 50 halving 0.25     | 0.90 / 0.92   | 0.69          | 1 ms          | 0.92          | 0.00 / 0.19     | 0.01%            | 0.7           |
  | smooth 50 halving 0.5      | 0.88 / 0.79   | 0.65          | 1 ms          | 0.92          | 0.00 / 0.19     | 0.01%            | 0.6           |
  | smooth 50 halving 1        | 0.87 / 0.73   | 0.60          | 2 ms          | 0.92          | 0.00 / 0.19     | 0.01%            | 0.6           |
  | smooth 100 halving 0.25    | 0.73 / 0.81   | 0.64          | 1 ms          | 0.92          | 0.00 / 0.19     | 0.01%            | 0.7           |
  | smooth 100 halving 1       | 0.71 / 0.75   | 0.54          | 5 ms          | 0.92          | 0.00 / 0.19     | 0.01%            | 0.6           |
  | smooth 200 halving 0.25    | 1.46 / 1.63   | 0.59          | 2 ms          | 0.92          | 0.00 / 0.19     | 0.01%            | 0.7           |
  | smooth 200 halving 0.5     | 1.54 / 1.64   | 0.53          | 5 ms          | 0.92          | 0.00 / 0.19     | 0.01%            | 0.6           |
  | smooth 200 halving 1       | 1.52 / 1.58   | 0.46          | 17 ms         | 0.92          | 0.00 / 0.19     | 0.01%            | 0.6           |
  | xy_rest 0.25               | 0.70 / 0.73   | 0.60          | 2 ms          | 0.92          | 0.00 / 0.19     | 0.01%            | 0.7           |
  | xy_rest 1                  | 0.71 / 0.76   | 0.58          | 2 ms          | 0.92          | 0.00 / 0.19     | 0.01%            | 0.6           |
  | z threshold 0.25           | 0.96 / 0.99   | 0.61          | 2 ms          | 0.92          | 0.00 / 0.19     | 0.01%            | 0.7           |
  | z threshold 1              | 0.21 / 0.25   | 0.56          | 2 ms          | 0.92          | 0.00 / 0.19     | 0.01%            | 0.6           |
  | z threshold 2              | 0.00 / 0.00   | 0.51          | 2 ms          | 0.92          | 0.00 / 0.19     | 0.01%            | 0.6           |
  | edge 0.05                  | 0.69 / 0.52   | 0.52          | 3 ms          | 0.92          | 0.00 / 0.19     | 0.00%            | 0.7           |
  | z window 9x9               | 0.70 / 0.81   | 0.59          | 2 ms          | 0.92          | 0.00 / 0.19     | 0.01%            | 0.6           |
  | max_speed 4                | 0.72 / 0.74   | 0.54          | 2 ms          | 0.92          | 0.00 / 0.19     | 0.01%            | 0.7           |
  | max_speed 8                | 0.71 / 0.74   | 0.61          | 2 ms          | 0.92          | 0.00 / 0.19     | 0.00%            | 0.6           |
  | levels 5                   | 0.71 / 0.75   | 0.56          | 2 ms          | 1.09          | 0.00 / 0.21     | 0.01%            | 0.6           |
  | xy window 9x9              | 0.71 / 0.74   | 0.58          | 2 ms          | 1.01          | 0.00 / 0.13     | 0.01%            | 0.7           |
  | damping 0.02               | 0.71 / 0.74   | 0.58          | 2 ms          | 0.98          | 0.00 / 0.20     | 0.01%            | 0.5           |
  | damping 0.045              | 0.72 / 0.75   | 0.60          | 2 ms          | 0.84          | 0.00 / 0.16     | 0.01%            | 0.7           |
  | xy threshold 0.25          | 0.71 / 0.74   | 0.59          | 2 ms          | 0.96          | 0.18 / 0.43     | 0.01%            | 0.6           |
  | xy threshold 0.4           | 0.71 / 0.74   | 0.59          | 2 ms          | 0.93          | 0.03 / 0.29     | 0.01%            | 0.6           |
  | z_smooth on                | 0.71 / 0.74   | 0.59          | 2 ms          | 0.95          | 0.00 / 0.16     | 0.01%            | 0.7           |

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
| `displacement`   | `displacement_filter.frag`            | the x/y filter's memory, with `z_smooth`      |
| `depth_velocity` | `depth_velocity_shader.frag`          | z's memory: the followed disparity, its rate and anchor (`estimate`) |

Measured in a hidden window at 448 × 800, 40 frames at 25 fps: a NaN mask over 200 × 200 px gave
1 000 000 non-finite velocity values in the first second, zeroed, the velocity finite after; the x/y
filter's memory filled with NaN, 358 400 displacement values once, every pixel, and the filter went on
from 0. z's memory filled with NaN was counted 0 times and the velocity stayed finite: the carry takes
a track's texel only within `max_speed` × the frame's time of the measurement, a comparison a NaN fails, so the
track starts over and the NaN is gone without a non-finite value written (with `max_speed` 0 the carry takes
every texel, so the guard counts it).

## Debugging a solve

With the environment variable `POSER_SOLVE_CAPTURE` set to a folder, a solve that comes out wrong (a
non-finite value, or a step past half the image a frame) is saved there with the two luminance frames
at each mip level and the depth as they were just before it ran (`motion/solve_capture.py`), up to three
a run. `scripts/solve_replay.py` reruns a capture through the GPU solve in a hidden window: a capture the
solve gets right there and wrong in the app points at the app's state,
not at the solve. `gl_state()` (`modules/gl`) returns the GL state a pass runs with.

## Constraints

- The coarsest level starts from no motion inside the shader (`has_estimate` off), never from a
  cleared buffer: a clear handed a temporary array for its color sometimes left garbage at startup,
  the bit pattern of 100 or 1.75 in one channel, 7 of 8 starts, and the whole pyramid took it as the
  start, every x/y arrow the same ("all arrows up"); with the `z_smooth` filter it stuck until a reset.
  Every clear to 0 now passes no memory (`clear_value`, `modules/gl/texture.py`).

- The false z at an edge cannot be told from a push by time. Restarting the track where the depth
  changed past a fixed step over 2 frames cuts the edge z, and a push's z by 26%.
- It cannot be told by the depth's slope either. A weight on the slope cuts edges and pushes alike,
  by about 10%.
- Sharpening the depth at the image's edges does not separate them. A luminance-guided joint bilateral
  filter on the depth lowers push and sweep z alike, by 7%, for +0.14 ms.
- A forward–backward check on x/y drops half the fast motion.
- z from the x/y flow's spread (looming: the speed toward the camera is the depth times the divergence over 2,
  Lee 1976) does not tell a push from other motion. On the captures of 2026-11-17 (the motion raw, the flow
  blurred 3 to 6 px, the signed mean over the player's moving pixels against the depth's z) it follows a push
  (hands, correlation +0.59 at half the size), a sideways arm less (measure, +0.40 at twice the size) and free
  movement the wrong way (show, −0.42): a sideways limb, a turn or a limb passing in front spreads the flow too.
- Iterations per level do less than one more level, at more GPU time.
- Each level is solved at its own size. Solving every full-size pixel at a coarse level puts each
  window a fraction of a texel off its neighbor's, and where a window is badly conditioned the solve
  flips from pixel to pixel in rows and columns: streaks from 4 levels up. A median between levels
  also removes them, at +0.35 ms or more, so there is none.
- The exposure is the camera's to settle, not the solve's: the solve matches no brightness between
  frames.

