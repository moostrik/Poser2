# Volume views: the plan

A temporary plan, deleted when its last step is in; what it builds is then stated in
`FLOW.md`'s monitor section. "Code" is what the files show; **(deduction)** marks what is inferred.

## What exists (code)

- The 3D fluid's fields are volumes: `Fluid3D._velocity`, `_temperature`, `_pressure`
  (`SwapTexture3D`, the simulation's width × height × `sim_layers`), `_divergence_vol`,
  `_curl_vol`, `_vorticity_force_vol`, `_buoyancy_force_vol`, `_simulation_obstacle`
  (`Texture3D`), and `_density` (`SwapTexture2DArray` at the output size, 2160 × 3840 × `layers`
  of RGBA16F, about 1 GB). `Fluid3D.ports` lists only the 2D obstacle.
- `Port.texture` is a 2D `Texture`. The monitor's views (`MonitorView`: `IMAGE`, `COLORMAP`,
  `ARROWS`, `HISTOGRAM`, `STRIP`) all read a 2D texture through `Visualizer` and `Strip`, and draw
  with full-screen quads (`channel_view`, `velocity_arrow_field`, `histogram_*`, `strip_*`).
- Nothing in the render uses a depth buffer (`Fbo` has no depth attachment) or instanced drawing.
- The window manager reports mouse moves and buttons on the render thread
  (`add_mouse_move_callback`, `add_mouse_button_callback`, 0 to 1 across the window); nothing
  registers for them, and scroll is not wired.

## The design

- **Volume ports.** A port's texture may be a volume. Every view reads a volume at the
  simulation size: the density is averaged down to the simulation's width, height and
  `sim_layers`, once per tick and only while a monitor shows it, so no view shimmers or reads
  the whole gigabyte. The other volumes are at that size already.
- **Picture views draw a volume as a cube**, summaries read all of it:

  | View       | 2D texture (as now)        | Volume                                                       |
  |------------|----------------------------|--------------------------------------------------------------|
  | `IMAGE`    | the picture                | stacked slices, each drawn as the picture                    |
  | `COLORMAP` | colormap on mid grey       | stacked slices, each drawn as the colormap                   |
  | `GLYPHS`   | arrows, or dots for a scalar | 3D arrows, or 3D dots for a scalar, on a coarse grid       |
  | `FOG`      | (none)                     | the scalar ray-marched as a colormap fog                     |
  | `STRIP`    | mean per channel over time | the same over all the voxels                                 |
  | `HISTOGRAM`| counts of the samples      | counts of all the voxels                                     |

  `GLYPHS` replaces `ARROWS`, renamed in the step that builds it. A volume in any picture view
  can instead be drawn flat, with one more choice, `cut`:

  | `cut`   | Means                                                                     |
  |---------|---------------------------------------------------------------------------|
  | `CUBE`  | the 3D view (default for a volume)                                        |
  | `SLICE` | the layer at `layer`, 0 to 1 through the volume                           |
  | `FRONT`, `TOP`, `SIDE` | the projection along z, y or x: the max for an image port, the mean for a velocity or a scalar, whose signs would cancel in a max |

  A flat cut is a 2D texture, so every 2D view draws it as it is; from the top, the glyphs of x
  and z show the depth movement. `cut` and `layer` are not read for a 2D port.
- **The orbit camera.** The 3D views share one camera on each monitor, around the volume's
  center: `orbit.yaw`, `orbit.pitch` (degrees), `orbit.distance` (box sizes), `orbit.spin` (a
  turn on its own, degrees per second, 0 off). The volume is a box of the fluid's width,
  height and z extent (`Fluid3D._z_extent`), so a layer gap looks as deep as it is.
- **The mouse.** A drag that starts anywhere in the main window goes to every monitor: on a
  cube it turns yaw and pitch, on a slice it moves `layer`; scroll zooms `distance` or steps one
  layer. The mouse writes the monitor's settings, so the panel shows the same values; A and B
  follow one drag. Nothing else in the main window takes the mouse.

## Steps

Each step ends with its tests and a commit; the user runs the app.

1. **Volume ports and flat cuts.**
   - `modules/flow/port.py`: `Port.texture` is a `Texture | Texture3D | Texture2DArray`;
     `Port.volume` says which. `Port.prepare`, a callable or None, is what a view calls before it
     reads the port, at most once per tick.
   - `Fluid3D.ports` adds `velocity`, `density`, `temperature`, `pressure`, `divergence`,
     `vorticity`, `vorticity_force`, `buoyancy`, `obstacle`. The density's port is a `Texture3D`
     at the simulation size with `prepare` set to `Fluid3D._reduce_density`, a new compute shader
     (`fluid_3d/shaders/reduce_density.comp`, the mean of each block of texels and layers).
   - `modules/flow/visualization`: `VolumeCut`, one shader for a slice and the three
     projections (`volume_cut.frag`, a `sampler3D` or a `sampler2DArray` by a uniform), into an
     FBO of the cut's size; the monitor's views then read that FBO. `Strip` and the histogram
     count read a volume through its `FRONT` mean, which is the volume's mean **(deduction:
     exact for the strip; the histogram then counts the projection, not the voxels, until step 5)**.
   - The monitor: `advanced.cut` and `advanced.layer`; `cut` defaults to `SLICE` until step 3
     builds the cube.
   - Tests: GL-free, the port's `volume` and `prepare` called once per tick, the projection
     rule by kind.
2. **The mouse and the orbit settings.**
   - `WindowManager.add_mouse_scroll_callback`, beside the move and button ones.
   - `modules/flow/visualization/orbit.py`: `OrbitSettings` (`yaw`, `pitch`, `distance`,
     `spin`) and the view and projection matrices in numpy, no GL.
   - `MonitorLayer.drag(dx, dy)` and `scroll(steps)`, writing its settings; each app registers
     the window's mouse callbacks once and hands a drag that started with a button down, and
     each scroll, to its monitors.
   - Tests: GL-free, the matrices (the center maps to the middle, yaw turns about y), a drag
     and a scroll writing the settings and clamping pitch and `layer`.
3. **The 3D view: stacked slices.**
   - `modules/gl`: an `Fbo` with a depth attachment, on request (`Fbo(depth=True)`), for this
     step and the next.
   - `modules/flow/visualization`: `VolumeCube` draws every layer as a quad at its depth, with
     the camera's matrices, back to front, each through the channel view's image or colormap,
     partly transparent, the layer at `layer` brighter, and the box's edges for orientation.
     New: a vertex shader with a `mat4` (`cube_slice.vert`, `.frag`).
   - The monitor: `cut` defaults to `CUBE` for a volume.
4. **Glyphs.**
   - `ARROWS` becomes `GLYPHS` (`MonitorView`, the `AUTO` map, both apps' presets, `FLOW.md`).
   - Flat: `glyph_field.frag` replaces `velocity_arrow_field.frag`: an arrow for a velocity as
     now, a dot for a scalar (the colormap's color, its size the value).
   - 3D: instanced glyphs on a grid of about 16 × 9 × the layers, the volume sampled in the
     vertex shader; an arrow as a camera-facing ribbon (core GL draws no wide lines), a dot as a
     point sprite (`GL_PROGRAM_POINT_SIZE`), drawn with the depth test. New: instanced drawing
     and a glyph mesh in `modules/gl` or beside the shader.
5. **Fog, and the summaries over all voxels.**
   - `volume_fog.frag`: a ray through the box per pixel, the colormap's color and the value's
     size as opacity, from the camera's inverse matrix.
   - `histogram_count.comp` counts a volume's voxels directly; the strip keeps the mean.
6. **Close.** `FLOW.md`'s monitor section states the views as built, its Open item goes, and
   this file is deleted.

## Open

- Whether the density's reduced view should also feed the fluid's own composite at a lower
  quality setting is outside this plan.
