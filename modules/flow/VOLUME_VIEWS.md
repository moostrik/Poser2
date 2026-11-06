# Volume views: the plan

A temporary plan, deleted when its last step is in; what it builds is then stated in
`FLOW.md`'s monitor section. "Code" is what the files show; **(deduction)** marks what is inferred.

## What exists (code)

- The 3D fluid's fields are volumes: `Fluid3D._velocity`, `_temperature`, `_pressure`
  (`SwapTexture3D`, the simulation's width × height × `sim_layers`), `_divergence_vol`,
  `_curl_vol`, `_vorticity_force_vol`, `_buoyancy_force_vol`, `_simulation_obstacle`
  (`Texture3D`), and `_density` (`SwapTexture2DArray` at the output size, 2160 × 3840 × `layers`
  of RGBA16F, about 1 GB). `layers` and `sim_layers` are separate settings (`VolumeSettings`).
  `Fluid3D.ports` lists only the 2D obstacle.
- `Port.texture` is a 2D `Texture`. The monitor's views (`MonitorView`: `IMAGE`, `COLORMAP`,
  `ARROWS`, `HISTOGRAM`, `STRIP`) all read a 2D texture through `Visualizer` and `Strip`, and draw
  with full-screen quads (`channel_view`, `velocity_arrow_field`, `histogram_*`, `strip_*`).
- The monitor renders at the size the app allocates its layer, 2160 × 3840 in both apps
  (`MonitorLayer.allocate` hands it to the visualizer), to be blitted into a panel about 1000
  pixels high.
- The strip and the histogram sample a fixed grid, whatever the texture's size: the strip a
  128 × 228 summary, averaged by its mipmaps to one mean per channel; the histogram 256 × 256
  samples into 256 bins per channel.
- Nothing in the render uses a depth buffer (`Fbo` has no depth attachment) or instanced drawing.
- The window manager reports mouse moves and buttons on the render thread
  (`add_mouse_move_callback`, `add_mouse_button_callback`, 0 to 1 across the window); nothing
  registers for them, and scroll is not wired.

## The design

- **The monitor renders at the size it is drawn at.** It reads the viewport in `draw()`, the
  rect the app fits it into, and renders there in the same call, so it renders at that size and
  reallocates when it changes (a window resize, a layout change). In a panel about 1000 pixels
  high it renders 1000 high; in the output on a 4k screen, 4k; shown at two sizes in one tick,
  at the larger, the smaller scaled down. No setting: a fixed number is wrong as soon as the
  window or the layout changes. The same holds for what it draws over a view: the strip's plot
  (its 540 columns stay, its history of 9 s at 60 ticks, its height follows) and the legend (lane
  names, log lines, limits, the colormap's key), whose text keeps a fixed pixel size so it stays
  sharp and the same in any panel, and whose lines sit on the bars because they share the size.
  The legend still redraws only when its content changes; a size change is one more reason.
- **The summaries read the texture's own grid, capped.** The strip's summary and the histogram's
  samples follow the texture's size up to today's 128 × 228 and 256 × 256, so the 512 × 288
  flow is read whole and only the 4k textures are sampled, at a fixed cost.
- **Volume ports.** A port's texture may be a volume, always a 3D texture (`sampler3D`), so each
  view is one shader. The density, stored as a 2D array at the output size, is averaged down into
  a 3D texture of the simulation's width and height and its own `layers`, so no view shimmers or
  reads the whole gigabyte. The fluid owns that reduction: it marks the reduced density stale at
  every step, and the first view that asks for it this tick (`Port.prepare`) reduces it, so two
  monitors showing it cost one reduction and none shows it costs nothing.
- **A volume is addressed by position, not by layer.** Every view samples a volume at (u, v, w)
  from 0 to 1 through the box, so each volume brings its own layer count: the density's `layers`
  and the simulation's `sim_layers` need not match, and a density dot and a velocity arrow at one
  grid point sample the same place.
- **Picture views draw a volume as a cube**, summaries read all of it:

  | View       | 2D texture (as now)        | Volume                                                       |
  |------------|----------------------------|--------------------------------------------------------------|
  | `IMAGE`    | the picture                | stacked slices, each drawn as the picture                    |
  | `COLORMAP` | colormap on mid grey       | stacked slices, each drawn as the colormap                   |
  | `GLYPHS`   | arrows, or dots for a scalar | 3D arrows, or 3D dots for a scalar, on a coarse grid       |
  | `STRIP`    | mean per channel over time | the same over the whole volume: its front mean projection, the volume's mean |
  | `HISTOGRAM`| counts of the samples      | counts of a 3D grid of samples, 64 × 64 × the layers         |

  `GLYPHS` replaces `ARROWS`, renamed in the step that builds it. `view` says how values get
  their colors; how a volume is laid out in space is one more choice, `cut`, which every picture
  view takes:

  | `cut`   | Means                                                                     |
  |---------|---------------------------------------------------------------------------|
  | `CUBE`  | the 3D view, the volume as stacked slices (default for a volume)          |
  | `FOG`   | the same cube drawn continuously, a ray through the box per pixel: with `IMAGE` the density's own colors as a cloud, close to the show's composite, with `COLORMAP` the signed values as a cloud; `GLYPHS` take it as `CUBE` |
  | `SLICE` | the volume at the slice position, its nearest layer                       |
  | `FRONT`, `TOP`, `SIDE` | the projection along z, y or x: the max for an image port, the mean for a velocity or a scalar, whose signs would cancel in a max; drawn in the box's proportions (the top is width × z extent, the side z extent × height), so the depth looks as deep as it is, as in the cube |

  A flat cut is a 2D texture, so every 2D view draws it as it is; from the top, the glyphs of x
  and z show the depth movement. `cut` is not read for a 2D port.
- **The camera and the slice are the monitor's state, not settings.** Yaw, pitch and distance of
  the orbit camera around the volume's center, and the slice position (0 to 1), live in the
  monitor; the mouse changes them and the monitor's `reset` sets them back to the front view at
  the middle layer. No setting is written per tick. A read-only readout on the panel shows the
  angle and the slice. The volume is a box of the fluid's width, height and z extent
  (`Fluid3D._z_extent`), so a layer gap looks as deep as it is.
- **The mouse.** A drag that starts anywhere in the main window goes to every monitor. The
  left button turns yaw and pitch, the right button zooms the distance (a vertical drag), and
  scroll selects the layer, in every view alike: on a flat slice it is the slice shown, in the
  cube it is the layer drawn brighter and pulled out of the stack: a gap opens on both sides of
  it, fading over the next layers, so it shows whole from any angle, an exploded stack. A scroll
  step is one layer of the density, the show's `layers`, for every monitor, and a volume with
  fewer layers shows its nearest; a drag across the window's width turns 180°. A and B get the
  same mouse, so they stay at one angle, one distance and one position. Nothing else in the
  main window takes the mouse.
- **Transparent slices in order, in one draw.** The stacked slices are one instanced draw of a
  quad, one instance per layer, blended from the far side to the near one: a uniform with the
  sign of the camera's direction along z makes instance i draw layer i or the last minus i, so
  the order flips exactly when the view turns past the side, with no loop over the layers.

## Steps

Each step ends with its tests and a commit; the user runs the app.

1a. **The monitor at its drawn size.** Fixes today's 4k rendering of every monitor, before any
   volume work.
   - The monitor: the visualizer, the strip's plot height and the legend allocate at the
     viewport's size in `draw()` and follow a change: the largest viewport of the previous tick,
     since the first draw of a tick cannot see the second's, and a re-render at once when a
     larger one turns up, so a layout change lags one tick and nothing else; `allocate` no longer
     sizes them from the layer's 4k. `STRIP_HEIGHT` goes; the
     legend's text is drawn at a fixed pixel size into its viewport-sized FBO.
   - The summaries' read grid: the texture's size, capped at 128 × 228 for the strip and
     256 × 256 for the histogram, its total the samples read.
   - Tests: GL-free, the size followed (the previous tick's largest, a larger one at once).
1b. **Volume ports, flat cuts and the summaries over a volume.**
   - `modules/flow/port.py`: `Port.texture` is a `Texture | Texture3D`; `Port.volume` says which;
     `ports_of` takes a 3D texture as a field's value too. `Port.prepare`, a callable or None, is
     what a view calls before it reads the port.
   - `Fluid3D.ports` adds `velocity`, `density`, `temperature`, `pressure`, `divergence`,
     `vorticity`, `vorticity_force`, `buoyancy`, `obstacle`. The density's port is a `Texture3D`
     of the simulation's width and height and the density's `layers`, its `prepare`
     `Fluid3D._reduce_density`: a compute shader (`fluid_3d/shaders/reduce_density.comp`, the mean
     of each block of texels), run at the first call after a step, marked stale by the step, and
     never when no view asks. Where the density is no larger than the simulation, the pass is a
     plain copy into the 3D texture, which every view needs for its one sampler type. The fluid
     allocates, reallocates and frees that texture with its simulation fields.
   - `modules/flow/visualization`: `VolumeCut`, one shader for a slice and the three projections
     (`volume_cut.frag`, `sampler3D`, the slice at w, the projections walking each texel of their
     axis), into an FBO of the cut's size; the monitor's views then read that FBO.
   - The summaries: the strip reads a volume through its front mean projection;
     `histogram_count.comp` gets the 3D grid (64 × 64 × the layers, its total the full height).
   - The monitor: `advanced.cut`, defaulting to `SLICE` until step 3 builds the cube; the slice
     at the middle, moved by the mouse from step 2.
   - Tests: GL-free, the port's `volume`, the reduction run once per step however many calls,
     the projection rule by kind, the histogram total for a volume.
2. **The mouse and the camera.**
   - `WindowManager.add_mouse_scroll_callback`, beside the move and button ones.
   - `modules/flow/visualization/orbit.py`: `Orbit`, yaw, pitch and distance with their limits,
     and the view and projection matrices in numpy, no GL.
   - `MonitorLayer.drag(dx, dy, button)` and `scroll(steps)`: the left button the orbit's yaw
     and pitch, the right its distance, scroll the layer; `reset` sets them back; a read-only
     readout shows them. Each app registers the window's mouse callbacks once and hands a drag
     that started with a button down, with its button, and each scroll, to its monitors.
   - Tests: GL-free, the matrices (the center maps to the middle, yaw turns about y), each
     button's drag and a scroll clamping pitch, distance and the layer, reset back to the front
     view.
3. **The 3D view: stacked slices.** No depth buffer: the slices blend from far to near, which a
   depth test would break; only the glyphs need one.
   - `modules/flow/visualization`: `VolumeCube` draws every layer as a quad at its depth, with
     the camera's matrices, from the far side to the near one by the sign of the camera's
     direction along z, each through the channel view's image or colormap, partly transparent,
     the selected layer brighter and pulled out of the stack (a gap on both sides of it, fading
     over the next layers), and the box's edges for orientation. New: a vertex shader with
     a `mat4` (`cube_slice.vert`, `.frag`).
   - The monitor: `cut` defaults to `CUBE` for a volume.
   - Tests: GL-free, the layer order flipping with the side the camera is on.
4. **Glyphs.**
   - `ARROWS` becomes `GLYPHS` (`MonitorView`, the `AUTO` map, both apps' presets, `FLOW.md`).
   - Flat: `glyph_field.frag` replaces `velocity_arrow_field.frag`: an arrow for a velocity as
     now, a dot for a scalar (the colormap's color, its size the value, small values small or
     transparent).
   - `modules/gl`: an `Fbo` with a depth attachment, on request (`Fbo(depth=True)`).
   - 3D: instanced glyphs on a grid of about 16 × 9 × the layers, the volume sampled at (u, v, w)
     in the vertex shader; an arrow as a small mesh, a shaft and a head of about 30 triangles
     with simple lighting, so its direction reads from any angle (a camera-facing flat arrow
     pointing at the camera would be a dot, and z, the depth movement, is what the cube is for);
     a dot as a point sprite (`GL_PROGRAM_POINT_SIZE`); both drawn with the depth test. New:
     instanced drawing and the arrow mesh in `modules/gl` or beside the shader.
5. **The fog cut.**
   - `volume_fog.frag`: a ray through the box per pixel from the camera's inverse matrix, each
     sample colored as the view colors it (the image's color, or the colormap's) and its amount as
     opacity; at the size the monitor is drawn at. `cut` gains `FOG`.
6. **Close.** `FLOW.md`'s monitor section states the views as built, its Open item goes, and
   this file is deleted.

## Open

- Whether the density's reduced view should also feed the fluid's own composite at a lower
  quality setting is outside this plan.
