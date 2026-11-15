# The GL's plan

The GL held up to the rules (2026-11-15, the apps' plan F), after the render's code move brought 15 shaders in. Not
started. Each step is built on Matthias's "go", which covers the harness runs the steps name. What the code shows is
marked **(seen)**.

The plan changes the GL's form, not what it draws: after it, every app looks as it does now. The harness holds every
gl shader, the texture's draws and the text to the reference within 1/255; where two shaders do not draw the same,
both stay (G). At the end, Matthias looks at all four apps.

| Commit           | Checked by                                  |
|------------------|---------------------------------------------|
| 0                | its own first run, the reference images     |
| A, B, D, E       | the tests and the harness                   |
| C                | the harness, then Matthias's look at the apps |
| F                | the harness                                 |
| G, three commits | the harness after each                      |

## 0. The harness

`scripts/shader_check.py`: in a hidden GL context, its own render thread, it compiles every shader of `modules/gl`,
draws each once on a known input, and draws a texture the ways C names (whole, a `dst`, a `src`, flipped) and a few
strings of text; it compares each image with the reference its first run made, within 1/255 per channel, since a
GPU's output is not bit-exact across two code paths that draw the same, so a shader no app draws is checked too. Its first run, before any change, makes the reference; its calls change with the steps, as the shaders' `use()`
do. It keeps the three shaders A moves to smile on its list, so a missed rename of a moved or renamed shader's source
(A, B, E) shows: a shader finds its source by its file's name, and no test compiles one.

## A. Smile's shaders out of the GL

`shaders/convenience/` holds smile's show effects: `FlashIn` and `FlashOut` mix a logo texture in, and `Omission` is
the bloom's dissolve through the mask, with smile's constants (`× 2.5 − 1.3`). Only smile's flash, reveal and bloom
layers use them **(seen)**. They move to `apps/smile/render/shaders/`: their `from .. import Shader` becomes
`from modules.gl import Shader`, smile's `shaders/__init__.py` exports them, and those layers import them from it.

## B. The order of `modules/gl/shaders`

After A, `convenience/` holds `BlurH`, `BlurV`, `Curtain`, `NoiseBlend` and `NoiseSimplexBlend`, all generic
operations. `convenience/` dissolves into `shaders/`, which stays one flat folder; grouping the shaders by kind of work
is left out.

## C. A texture draws itself

`modules.gl.Blit` (`utils.py`) is a class that draws through one shared `shaders.Blit` it creates at its first use and
never frees; `shaders.Blit` is the shader a layer owns (one user, `LayerCompositor`) **(seen)**. `BlitRect`,
`BlitRegion` and `BlitFlip` come in the same pairs; 33 calls use the shared ones, `BlitFlip` one of them (`Image`'s
upload), `BlitRect` and `BlitRegion` none **(seen)**.

As GL frameworks have it (openFrameworks' `ofTexture::draw` and `drawSubsection`, Cinder's `gl::draw(texture,
srcArea, dstRect)`), a texture draws itself: `Texture.draw(dst: Rect | None = None, src: Rect | None = None, opacity:
float = 1.0)`, which `Fbo`, `SwapFbo` and `Image` have by inheritance. `dst` is where it lands in the target (the whole
target without one), `src` the part of the texture drawn (all of it without one); a `src` with a negative width or
height draws it flipped. One shader does it, `Blit`, its vertices placed by `dst` and its texture coordinates by
`src`; `BlitRect`, `BlitRegion` and `BlitFlip` go into it.

The default shader belongs to the GL context, as Cinder's `gl::Context` holds its stock shaders and default vertex
array, one per GL context, and openFrameworks' renderer one per window: `gl.Context`, one per window's GL context,
made once that context is up and freed before it closes, holds its `Blit` and the quad's vertex array `draw_quad`
assumes bound. `gl.context()` is the current one; `Texture.draw` and `Text` (F) draw through it, so `Texture` imports
no shader when it loads, and nothing looks up a window. Today the window manager makes the quad per window
(`_bind_quad_vao`) and the two glfw scripts bind an empty vertex array of their own (`monitor_harness.py`,
`depth_velocity_compare.py`, both drawing with `Blit.use`) **(seen)**: the window manager makes a window's context
current with the window, as it switches GL contexts per window now **(seen)**, and each script makes the one for its
window. A `Blit` per window, up to four, is a small shader compiled at the start **(inferred)**. The 33 `Blit.use(x)` calls become `x.draw()`, in smile, the flow, the render and
those scripts, in one commit; the `utils.py` classes go. `LayerCompositor`, which kept its own `shaders.Blit` for the
opacity **(seen)**, draws with `draw(opacity=...)` too, so `shaders.Blit` is the context's alone.

The blend state goes into the context too, as Cinder keeps its state stacks there. `Style` keeps one cache of the
blend state for the whole program and skips a `glBlendFunc` it holds already set **(seen)**, while GL keeps that state
per context; and blends set past `Style`, raw, leave its cache wrong as well: `Text`, `PanoramaCompositor`'s static
blit, `ScreenCapture`, white_space's two light-simulation layers, and the 3D fluid's particle density (per-buffer
blends, `glBlendFunci`, that `Style` cannot express) **(seen)**. The apps make up for it with ten `Style.reset_state()`
calls where they start drawing (deep_flow, hd_trio, smile) **(seen)**. Each `gl.Context` holds its own blend stack and
cache; `Style`'s functions keep their names and work on `gl.context()`'s, so its 145 calls in 30 files stay; the raw
blends go through `Style`, and `Style` learns the per-buffer blend (`glBlendFunci`, `glBlendEquationi`) so the
particle density's go through it too; the ten `reset_state()` calls go.

`Image` uploads into a source texture and draws it flipped into its FBO, once per upload **(seen)**, with `src`'s
negative height after C. The flip stays a pass: the time series and the feature window hand their `Image` to their
shaders as data (`TimeSeries`, `Window`), which sample it directly, not through `draw` **(seen)**, so a row order that
only `draw` knows would turn their data upside down. Its users are small data images and white_space's light
simulation, not the camera's upload **(seen)**.

With C, the GL gets its document, `modules/gl/GL.md`: what the GL holds, the context and what it owns, how a texture
draws, and where a shader sits.

## D. Time

`FpsCounter` and `lfo` (`utils.py`), `NoiseSimplex`'s animation and the shader hot-reload's debounce (`shader.py`)
read `time.time()` **(seen)**.
- `FpsCounter` (white_space's light rate) counts what `modules/utils`' `FPS` counts, on `perf_counter` already
  **(seen)**: it goes, and white_space's conductor counts with `FPS(120)` (`processed()` for `tick()`,
  `get_rate_average()` for `get_fps()`), 120 frames as `FpsCounter` averaged, where `FPS` takes 60 unless told.
  When the plan is built, what the light rate is and over how many frames it averages goes into white_space's
  document or plan.
- `NoiseSimplex` takes its time from the tick: its `use()` takes the frame's time. It has no caller **(seen)**.
- `lfo` and the debounce read `time.perf_counter()`.

## E. Names

`text_shader` loses its `Shader` ending and is named after what it draws, `Glyphs`. `box_shader` draws a solid
rectangle in pixels, what `DrawColoredRectangle` draws in other units **(seen)**: it is in G's check. `WindowSettings`'
docstring names "RenderManager", which is `WindowManager`.

## F. `Text` asks GL nothing

`Text.draw_text` lays the string out glyph by glyph, uploads its vertices and asks GL for the bound vertex array
(`glGetIntegerv`) to bind it again after, on every draw **(seen)**. As Cinder's context keeps the bound vertex array
and a draw binds its own scoped (`ScopedVao`), `gl.Context` (C) keeps which vertex array is bound, and `Text` binds
its own through it and gives the one before back, asking GL nothing. The layout stays per draw: static text is a layer's to
keep (`rendering.md`, redraw a layer only when what it draws has changed; the panorama's seams and grid, the flow's
`Legend` **(seen)**), and the text drawn every frame mostly changes every frame (camera readings, fps, tracklet and
pose labels **(seen)**), so a cache of strings would miss.

## G. The shaders on one scheme, none twice

After A to F, for `modules/gl`'s shaders. The 45 shaders follow no shared scheme **(seen)**: textures named by slot (`tex0` 19 times, `tex1`,
`tex2`), `tex` (8) or role (`src`, `dst`, `mask`); uniforms mostly camelCase, `box_color`, `text_color`,
`num_circles` and `aspect_ratio` snake_case; a color as four floats or a tuple, a rectangle as four floats or a
`Rect`; device coordinates (−1..1), texture coordinates (0..1) or texture space with an aspect ratio; the texel size
worked out by the shader, given (`texel_size`) or as a `resolution`; commented-out code in 20 `.frag` files.

As the textbook has it, each shader then has: its inputs named by role (`src`, `dst`, `mask`), in the `.frag` and in
`use()` alike; its uniforms in camelCase; colors as `Color` and rectangles as `Rect`; texture coordinates 0..1 with
the origin top-left wherever a `Rect` is passed, as the frameworks' 2D drawing, our `Rect` and images have it (the
grid: "Rect y is top-down" **(seen)**), GL's bottom-left only inside the shaders; its texel size from its input; no
dead code.

The origin is turned in one place. A texture is held in GL's own row order, a CPU image flipped once at its upload
(`Image`, as the frameworks load an image); the 2D API is top-left, and one projection the context holds turns it
into GL's coordinates, as Cinder's `setMatricesWindow(w, h, originUpperLeft = true)` and openFrameworks' screen setup
do: `gl.Context` holds the 2D projection, and every vertex stage applies it, `_generic.vert` and the shaders with a
`.vert` of their own (the rectangles, the circles, the glyphs, and the render's `PosePointLines`, which follows with
its callers). No shader flips on its own: today `ViewPoseLayer` flips its points before `DrawCircles` because
"`PosePointLines` flips Y internally" and "`DrawCircles` takes GL coords" **(seen)**; both flips go.

None twice: where two do the same, one stays and the callers move to it; where they differ, both stay. To check:
`MaskBlur` (one shader, a direction) against `BlurH` and `BlurV`; `Blend`, `MaskBlend` and `Curtain`, all
`(tex0, tex1, fade)`; `Hsl` against `Hsv`; `Noise` against `NoiseSimplex`; `NoiseBlend` against `NoiseSimplexBlend`;
`BoxShader` against `DrawColoredRectangle`.

A change of form: nothing draws otherwise, every caller changes with it. Three commits, each checked on its own:
1. names (inputs by role, uniforms in camelCase) and no dead code;
2. values (`Color`, `Rect`) and coordinates (0..1, the top-left origin through the context's projection, the texel
   size from the input);
3. none twice.

Checked by the harness (0) after each commit; then Matthias's look at the apps.

## Rules it touches

- `gl.md`, `modules/gl/shaders` holds the shaders useful to any app; an app's own shader sits in that app: A
- `CLAUDE.md` Packages, a package is one thing: A, B
- `CLAUDE.md` Naming, one term per concept: C, E, G
- `CLAUDE.md` API evolution, remove obsolete pathways instead of leaving parallel ones: C, G
- `gl.md`, pair every creation with its destruction; create shaders at setup: C
- `gl.md`, set the GL state a draw needs where it draws: C (the blend through `Style`)
- `CLAUDE.md` Composition and wiring, fix a problem where it is made, not where it shows: C (the resets)
- `CLAUDE.md` Time, `perf_counter` for every stamp and duration; downstream of the camera, time from the tick: D
- `CLAUDE.md` Running code, run a GPU harness only on the user's go, and only what the plan names: 0, C, F, G
- `gl.md`, a shader is named after the operation it does: E, G
- `rendering.md`, prefer to redraw a layer only when what it draws has changed: F
- `docs.md`, a component is described in its own document: C
