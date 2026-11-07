# The monitor

The flow's monitor (`MonitorLayer`, `modules/flow/visualization/`): a layer that shows one port of
one stage, picked on the panel, drawn only while it is shown. The stages and their ports are in
[FLOW.md](FLOW.md#the-stages-and-their-layers). "Code" is what the files show; **(deduction)**
marks what is inferred from them.

| Term   | Means                                                                                                 | In code                                                             |
|--------|-------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------|
| glyphs | a monitor's view of a field as marks on a grid: arrows for a velocity, dots for a scalar              | `MonitorView.GLYPHS`, `glyph_spacing`, `GlyphField`, `VolumeGlyphs` |
| cut    | how a monitor lays out a volume: in 3D as stacked layers or a fog, or flat, one layer or a projection | `Cut`, `advanced.cut`, `VolumeCut`                                  |

## Settings and views

The monitor's panel group (`MonitorLayerSettings`) says what is drawn; its `advanced` group
(`MonitorAdvancedSettings`) says how. Neither has row breaks of its own; the panel wraps them.

| Setting                  | Values                                                      | Means                                       |
|--------------------------|-------------------------------------------------------------|---------------------------------------------|
| `stage`                  | `MOTION`, `BRIDGE`, `FLUID`                                 | the stage whose port is shown               |
| `texture`                | the stage's port names                                      | the port shown; a name the stage no longer has shows its first output |
| `view`                   | `AUTO`, `IMAGE`, `COLORMAP`, `GLYPHS`, `HISTOGRAM`, `STRIP` | how the channels are drawn                  |
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

## Volumes

A port's texture may be a volume, always a `Texture3D` (`Port.volume`), so each view is one
shader. The 3D fluid lists its volumes as ports: `velocity`, `density`, `temperature`,
`pressure`, `divergence`, `vorticity`, `buoyancy`, `obstacle`, beside the 2D
`obstacle_source`. Each carries its box's depth (`Port.extent`, `Fluid3D._z_extent`); the
monitor draws the box as deep as it is wide (`BOX_DEPTH`), since the simulated slab, 64 cubic
cells on a 448 wide grid, is 0.14 of its width and reads as flat. A flat cut renders at the
part of the panel it fills in its proportions, so glyphs keep their spacing. Every view samples a
volume at (u, v, w) from 0 to 1 through the box, so the density may sit on a finer grid than the
simulation's (`density.size`) and is read as it is.

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
- The orbit and the selected layer are the monitor's state, not settings, and the panel shows
  neither. In deep_flow a drag that starts anywhere in the main window goes to every monitor, so
  A and B stay on one view: the left button turns (a drag across the window 180°), the right
  zooms, scroll steps one of the show's density `layers` (away from the user is further back,
  a volume with fewer layers showing its nearest), a middle click without a move resets to the
  front view at the middle layer. `Orbit` frames the box by the narrower field of view, so it
  fits a 9:16 panel.

`scripts/monitor_harness.py` checks the monitor without an app or a camera: a hidden GL window,
known textures and volumes with values numpy computes (a cut against numpy's layer, max and mean,
the cube's and the fog's near side from both sides of the box), pictures in `temp/monitor_harness`, and the GPU times, measured on the GPU's timer queries:

| Work                                   | At                                   | GPU (ms) |
|----------------------------------------|--------------------------------------|----------|
| image, colormap                        | 540 × 960 / 2160 × 3840              | 0.01 / 0.03 |
| glyphs, flat                           | 540 × 960 / 2160 × 3840, spacing 32  | 0.18 / 2.4 |
| histogram                              | 540 × 960 / 2160 × 3840              | 0.10 / 0.12 |
| strip                                  | 540 × 960                            | 0.05     |
| slice, front, top, side                | a 448 × 800 × 16 volume              | 0.01, 0.08, 0.61, 0.36 |
| cube, 3D arrows, fog                   | a 448 × 800 × 16 volume at 540 × 960 | 0.04, 0.05, 0.05 |
