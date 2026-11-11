# The monitor

The flow's monitor (`MonitorLayer`, `modules/flow/visualization/`): a layer that shows one port of
one stage, picked on the panel, drawn only while it is shown. The stages and their ports are in
[FLOW.md](FLOW.md#the-stages-and-their-layers). "Code" is what the files show; **(deduction)**
marks what is inferred from them.

| Term   | Means                                                                                                 | In code                                                             |
|--------|-------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------|
| glyphs | a monitor's view of a field as marks on a grid: arrows for a velocity, dots for a scalar              | `MonitorView.GLYPHS`, `glyph_spacing`, `GlyphField`, `VolumeGlyphs` |
| cube   | a monitor's view of a volume in 3D: the volume drawn continuously in its box, a ray per pixel         | `VolumeCube`, `VolumeCubeShader`, `CubeColor`                       |
| slice  | the scrolled layer of the cube in focus, framed, the rest dimmed                                      | `advanced.slice`                                                    |

## Settings and views

The monitor's panel group (`MonitorLayerSettings`) says what is drawn; its `advanced` group
(`MonitorAdvancedSettings`) says how. Neither has row breaks of its own; the panel wraps them.

| Setting                  | Values                                                      | Means                                       |
|--------------------------|-------------------------------------------------------------|---------------------------------------------|
| `stage`                  | `MOTION`, `BRIDGE`, `FLUID`                                 | the stage whose port is shown               |
| `texture`                | the stage's port names                                      | the port shown; a name the stage no longer has shows its first output |
| `view`                   | `COLORMAP` (default), `GLYPHS`, `IMAGE`, `HISTOGRAM`, `STRIP` | how the channels are drawn, the most used first |
| `advanced.channel`       | `X`, `Y`, `Z`, `W`, `ALL` (default)                         | which of the texture's channels are drawn   |
| `advanced.scale`         | 0.01 to 100, a log slider                                   | multiplier on the values, in every view; in the cube 1 after it is full |
| `advanced.opacity`       | 0.1 to 100, a log slider, 6 (default)                       | the cube: how much a full sample hides what is behind it, per box width |
| `advanced.depth_fade`    | 0 (default) to 10                                           | the cube: dimming per box width behind the near side |
| `advanced.slice`         | on, off (default)                                           | the cube: the scrolled layer framed and in focus, the rest dimmed |
| `advanced.slice_position`| 0 (front) to 1 (back), 0.5 (default)                        | the slice's place; the wheel steps it a layer |
| `advanced.slice_focus`   | 0 to 1, 0.5 (default)                                       | opacity moved from the rest to the slice: 0 none, 1 only the slice |
| `advanced.orbit`         | `yaw`, `pitch` (degrees), `zoom` (1 frames the box)        | the 3D camera; the mouse turns and zooms it |
| `advanced.glyph_spacing` | 4, 8, 16, 32, 64, 128 (`GlyphSpacing`)                      | distance between glyphs (pixels)            |
| `advanced.legend`        | on (default), off                                           | the log axis, or the colormap's key         |
| `advanced.blend_mode`    | a `Style.BlendMode`                                         | how the monitor is drawn over the panel     |

An arrow reaches one spacing at speed 1 and at most two (the shader rolls its length off toward
2), and its lines are spacing / 20 wide (`ARROW_WIDTH`), so the arrows keep their proportions at
every spacing. A scalar's dot is the colormap's color of its value, its radius half a spacing
times `tanh` of its amount, so small values draw small dots. A z position is drawn as itself, 0 front
to 1 back, in X and its weight in Y (`ZPositionView`), not as the weighted pair it travels as.

The strip starts over when `stage`, `texture` or `channel` changes, or when `view` turns to
`STRIP`, so it never shows another texture's history and needs no reset button.

| View     | `ALL`                                                              | one channel                                   |
|----------|--------------------------------------------------------------------|-----------------------------------------------|
| `COLORMAP` | by the port's kind: a velocity by its direction, 0.5 + 0.5 × the unit vector, as bright as its size capped at 1, black at 0; an image as stored; a scalar's x, y, z as red, green, blue around mid grey, grey 0 | a velocity's channel by its sign along its axis; a scalar's with the other channels 0: x from grey toward red above 0 and toward cyan below, y green, z blue; w as grey |
| `GLYPHS` | a velocity's arrow of x and y, with the ring for z on a 3- or 4-channel texture (channel not read); an acceleration's push, a filled arrowhead; a scalar's dot of x, and so any one-channel texture's, which has no direction | a scalar's dot of that channel |
| `IMAGE`  | the raw values as stored, times `scale`, for debugging             | that channel as grey, times `scale`           |
| `HISTOGRAM` | a lane for each channel the port holds, x on top: how many values fall where from −1 to 1 after `scale`, 0 in the middle, outside in the end bins | one lane, full height |
| `STRIP`  | a lane for each channel the port holds, x on top: the frame-wide mean of each channel's amount (its absolute value; a signed mean cancels left against right), on a log axis | one lane, full height |

- A port says the channels that hold its values, x first (`Port.channels`): a vector's components,
  at most x, y and z, since a 3D vector sits in RGBA with w unused; a scalar's and an image's every
  channel of its format; or `held` where the format holds more than the values (the light, RGB in
  RGBA; the 3D Lucas–Kanade filter, its displacement in RG beside its position error). The lanes,
  the colormap's key and its legend's names are those channels; a channel the port does not hold
  draws nothing. A shader writes 0 in a channel nothing reads.
- In the other views a channel the texture does not have reads as sampled: a one-channel texture
  repeats x in y and z (its grey swizzle, `modules/gl/texture.py`), a two-channel one has z 0. So a
  one-channel texture would draw arrows along the diagonal; the glyphs draw it as dots, whatever its
  port's kind.
- An acceleration (widths/s²) is drawn in every view as the velocity it adds in one reference step,
  times `REFERENCE_STEP` (1/60 s), as the fluid adds a body force × dt: a quick hand's 30 widths/s²
  reads 0.5, in the range of the velocities, so one `scale` reads both. Its title says "per step".
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
- A velocity's colors are those of the grey-centered map at its direction (+x reddish, +y greenish,
  +z bluish), without the grey: slow and fast water going one way share a color, and where nothing
  moves is black, so where a force is and which way it points read apart.

## Volumes

A port's texture may be a volume, always a `Texture3D` (`Port.volume`), so each view is one
shader. The 3D fluid lists its volumes as ports: `velocity`, `density`, `temperature`,
`pressure`, `divergence`, `vorticity`, `buoyancy`, `obstacle`, beside the 2D
`obstacle_source`. Each carries its box's depth (`Port.extent`, the fluid's `depth_ratio`); the
monitor draws the box that deep (`box_depth`), as deep as it is wide where a port gives none. Every view samples a volume at
(u, v, w) from 0 to 1 through the box, so the upscaled density, finer than the simulation, is read
as it is.

- `COLORMAP` and `IMAGE` draw a volume as the cube (`VolumeCube`), on the orbit camera: a ray per
  pixel marched front to back, 96 samples through the box, each sample colored by `CubeColor` (in
  `COLORMAP` by the port's kind as the table above, in `IMAGE` as stored), its amount times `scale`
  capped at 1, its opacity `opacity` per box width, the samples behind the near side dimmed by
  `depth_fade`; the box's edges over it.
- With `slice` on, a ray reads the layer at `slice_position` where it crosses it, and `slice_focus`
  moves opacity from the rest to it: at 0 the slice is a sample as any other, the plain cube; at 1
  its opacity is its amount, capped at 1, full where its value is, clear where it is empty, and the
  rest has none; in between both by the same share, so the layer stands out while the rest still
  shows where it is and which way it moves. The layer's frame is the box's edges flattened to its
  depth, in white, with the other outlines.
- `GLYPHS` draws in 3D: lit arrows of a small mesh per grid point, so one pointing at the camera
  still shows, or a scalar's dots as point sprites, both with the depth test (`Fbo(depth=True)`),
  at most 96 across (`glyph_grid`). An arrow's shape is its size and its color its direction, at
  full brightness: one cell long at 1 after `scale`, rolling off toward two, its thickness growing
  with its length up to full at one cell and its head never narrower than the dot; a force shorter
  than the dot, a tenth of a cell across and at least 2 px, draws as the dot, lit, in its direction's
  color, so weak forces still show which way. Nothing is cut off above 0. An acceleration draws the cone
  alone, a push, without the shaft; a one-channel volume draws dots. Every 3D view draws the box's edges. Its axes are the
  overlay's, the legend's, drawn only with `legend` on: from the box's first corner (x 0, y 0, the
  front layer), 3 px wide in the colormap's colors at 1 (`CubeAxes`), x red along the width, y green
  up, z blue into the depth, each axis's letter past its end (`axis_labels`, the legend redrawn when
  the camera moves). So the views themselves hold only what they show, and the harness, which draws
  without the legend, checks them alone. With the legend the cube also draws where the light is, when
  its port carries one (`Port.light`, the fluid's volumes while the light is on): a lamp as a small cube
  at its place, a sun as a line from the box's center out past where its light enters, the small cube
  at the line's end, in amber (`light_marks`, outlined by `CubeEdges`).
- The strip reads a volume's front mean (`VolumeCut`, the projection along z), the volume's mean;
  the histogram counts a grid through every layer, at most 128 × 128 across
  (`histogram_volume_grid`). `VolumeCut` also lays a volume flat for the scripts that read layers
  back: one layer, or the projection along z, y or x, the max for an image port and the mean for a
  velocity or a scalar, whose signs would cancel in a max.
- The orbit is `advanced.orbit` (`yaw`, `pitch`, `zoom`) and the slice's place `slice_position`,
  settings the mouse writes, so the panel shows and sets them and a preset keeps them. In deep_flow a drag that starts anywhere in the main window goes to every monitor, so
  A and B stay on one view: the left button turns (a drag across the window 180°), the right
  zooms, scroll steps one of the fluid's `depth` layers (away from the user is further back,
  a volume with fewer layers showing its nearest), a middle click without a move resets to the
  front view at the middle layer. `Orbit` frames the box by the narrower field of view, so it
  fits a 9:16 panel.

`scripts/monitor_harness.py` checks the monitor without an app or a camera: a hidden GL window,
known textures and volumes with values numpy computes (a cut against numpy's layer, max and mean,
the cube's near side from both sides of the box, a velocity's direction color, the slice brought
forward), pictures in `temp/monitor_harness`, and the GPU times, measured on the GPU's timer queries:

| Work                                   | At                                   | GPU (ms) |
|----------------------------------------|--------------------------------------|----------|
| image, colormap                        | 540 × 960 / 2160 × 3840              | 0.01 / 0.03 |
| glyphs, flat                           | 540 × 960 / 2160 × 3840, spacing 32  | 0.18 / 2.4 |
| histogram                              | 540 × 960 / 2160 × 3840              | 0.10 / 0.12 |
| strip                                  | 540 × 960                            | 0.05     |
| a flat cut: slice, front, top, side    | a 448 × 800 × 16 volume              | 0.01, 0.08, 0.61, 0.36 |
| 3D arrows, cube                        | a 448 × 800 × 16 volume at 540 × 960 | 0.05, 0.05 |

The cube's time is the one measured before its colors by kind and its slice; the harness times both
again, not yet run.
