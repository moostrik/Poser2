# GL

`modules/gl` is the apps' OpenGL framework: the window, the textures and FBOs, the shaders and their loader, text,
the blend state, and the shaders that are useful to any app. A layer that draws a module's data sits in
`modules/render`, an app's own shader in that app.

## What it serves

The GL serves a pipeline of 2D shader steps; a volumetric step is 3D only within its own pass. Geometry, such as
the poses and the rectangles, is worked out on the CPU and handed to a shader ready to place, so a draw needs no
matrix and no state per target. The pipeline's latency counts, and each GL call and state switch costs CPU time on
the render thread, so the GL keeps their number low: it holds GL's state on the application side and asks no
`glGet*`.

## The context

Each GL context has one `gl.Context`, as Cinder's `gl::Context`: the window manager makes one per window when the
window's context is first made current, and frees it before the window closes; a script that opens its own GL window
makes one for it (`Context().allocate()`). A context holds what its GL context does not share with another:

| It holds           | What for                                                                 |
|--------------------|--------------------------------------------------------------------------|
| the quad           | the vertex array `draw_quad` draws, bound while the context is current   |
| the bound vertex array | the one bound, kept so that no `glGet*` is asked                     |
| `Blit`             | the shader a texture draws itself with                                   |
| the blend state    | `Style`'s state and saved states, this context's alone                   |

`gl.context()` is the one made current (`Context.make_current`, which the window manager calls with each window's
GL context). Each context keeps its own blend state because GL keeps it per context: a state kept for the whole
program goes wrong after a switch of window. Every blend change goes through `Style`, so the state it holds is the
one GL has; `set_buffer_blend_mode` sets one draw buffer's blend alone.

## A texture draws itself

`texture.draw(dst, src, opacity)` draws a texture, an `Fbo`, a `SwapFbo` or an `Image` into the bound target, as
openFrameworks' `ofTexture::draw` and Cinder's `gl::draw(texture, srcArea, dstRect)`: `dst` is where it lands in the
target, `src` the part of the texture drawn, both 0..1 with the origin top-left, all of it over all of the target
without them. A `src` `Rect` with a negative width or height draws it mirrored on that axis (`Image` flips its upload
this way); a `src` `Quad`, four corners worked out on the CPU (`Rect.turned`), draws a turned part. It draws with the
blend state as set.

## Coordinates

Every 2D position a shader takes is in 0..1 of the viewport with (0, 0) at the top left: a rectangle, a point, a
texture's `dst` and `src`, a region, a part of an input sampled (`DepthUpscale`'s `roi`). The turn
into OpenGL's own coordinates (clip space −1..1 and texture coordinates, both from the bottom left) is written once,
in `shaders/position.glsl`: `clipPosition` for a vertex, `topLeft` for a texture coordinate. A shader includes it
(`#include "position.glsl"`; the loader finds it in `shaders/` from any package). A pass that fills its target
places nothing and draws the quad in −1..1 through `_generic.vert`.

A texture is held in GL's row order; a CPU image is flipped once, at its upload (`Image`, `Tensor`). Text is laid
out in pixels on the CPU (`text_layout`), from the size its caller gives, and handed to the shader in 0..1.

## Shaders

`shaders/` groups its shaders by kind, one subpackage each: `draw`, `color`, `filter`, `mask`, `composite`,
`transition`, `generate`, `video`. The package exports every one, so a caller writes `shaders.FillRect`; the kind
orders the files, the operation names the shader. The shared GLSL includes sit at the top of `shaders/`:
`position.glsl` (below) and `colormap.glsl`, turbo, viridis and jet by the number `gl.Colormap` gives them.

A shader's `.frag`, `.vert` or `.comp` source sits beside the `.py` that holds its class, named after it; the loader
resolves it by that module name and falls back to `_generic.vert` for the vertex stage. A shader takes a color as a
`Color` and a rectangle as a `Rect`; a pass reads its input's texel size from the texture itself (`textureSize`). `scripts/shader_check.py`
draws every shader of `modules/gl` once on known input and holds it to a reference image, within 1/255 per channel.
