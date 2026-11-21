# GL

`modules/gl` serves a low-latency 2D pipeline first: each frame goes from camera to screen in as few passes, copies
and waits as possible. A design choice here is judged by what it costs that path.

`modules/gl` is the apps' OpenGL framework: the textures and FBOs, the shaders and their loader, text, the blend
state, and the shaders that are useful to any app. The window loop and the windows are `modules/window`
([WINDOW.md](../window/WINDOW.md)), which draws with `gl`. A layer that draws a module's data sits in
`modules/render`, an app's own shader in that app.

Its files sit in folders by job: `texture/` (the texture kinds on `TextureBase`, their `Swap` pairs, `Fbo`,
`LayeredFbo`, the uploads `Image` and `Tensor` with its CUDA side), `shader/` (`Shader`, `ComputeShader`, the file
watcher, `StorageBuffer`), `state/` (`Context`, the blend state, the clear), `debug/` (GL's debug output, `gl_state`,
`GpuTimer`), `text/` and `shaders/`. Every name is used as `gl.<Name>`, from the package.

## What it serves

The GL serves a pipeline of 2D shader steps; a volumetric step is 3D only within its own pass. Geometry, such as
the poses and the rectangles, is worked out on the CPU and handed to a shader ready to place, so a draw needs no
matrix and no state per target. Each GL call and state switch costs CPU time on the main thread, so the GL keeps
their number low: it holds GL's state on the application side and asks no
`glGet*`. One PyOpenGL call costs 0.4 to 0.7 µs on the main thread (measured 2026-11-20).

GL calls a frame, -sim, counted over 20 s from 10 s in. Before: 2026-11-20, a sampler's unit set by `glUniform1i` and
a texture bound by `glActiveTexture` + `glBindTexture`. Now: 2026-11-21 at 839c2623, `layout(binding)` and
`glBindTextureUnit`, GL debug output on (which changes the time, not the count). white_space's before ran from the
recording's start, its now from 0.08, where several players are in view.

| App         | Recording                  | Before | Now  | Texture binds now |
|-------------|----------------------------|--------|------|-------------------|
| deep_flow   | `20261112-125832_measure`  | 2338   | 1713 | 293               |
| white_space | `20261024-164009_ICK`      | 1000   | 812  | 69                |
| low_flow    | `20261112-125832_measure`  |        | 1312 | 182               |
| hd_trio     | `intro_1`                  |        | 2230 | 316               |
| smile       | none: no camera, the image |        | 643  | 77                |

## The context

Each GL context has one `gl.Context`, as Cinder's `gl::Context`: the window manager makes one per window when the
window's context is first made current, and frees it before the window closes; a script that opens its own GL window
makes one for it (`Context().allocate()`). A context holds what its GL context does not share with another:

| It holds           | What for                                                                 |
|--------------------|--------------------------------------------------------------------------|
| the quad           | the vertex array `draw_quad` draws, bound while the context is current   |
| the bound vertex array | the one bound, kept so that no `glGet*` is asked                     |
| `Blit`             | the shader a texture draws itself with                                   |
| the blend state    | `BlendState`'s state and saved states, this context's alone              |

`gl.context()` is the one made current (`Context.make_current`, which the window manager calls with each window's
GL context). Each context keeps its own blend state because GL keeps it per context: a state kept for the whole
program goes wrong after a switch of window. Every blend change goes through `BlendState`, so the state it holds is the
one GL has; `set_buffer_blend_mode` sets one draw buffer's blend alone.

## The textures

`Texture`, `Texture3D` and `Texture2DArray` share `TextureBase`: the GL name, size, format, wrap and filter, `bind`,
`clear` and the read back. CPU data goes into a texture through `upload(array)`, the array of its size in its upload
type, made contiguous and byte-aligned there; a texture made with `bgr` takes it in BGR order, an OpenCV image's.
A volume is a `Texture2DArray` where the 3D texture's tiling is slow at its size, layers of 2D tiles; it filters
within a layer, not between layers, so a shader that reads between layers mixes them itself **(site fact)**.

An `Fbo` is a framebuffer that owns its color targets, one or more, one size, and is its target 0: the texture a
pass writes, so a pass hands the `Fbo` on as that texture. A draw writes its output `layout(location = i)` into
target i, `targets[i]`, made at a format each (`allocate(width, height, formats)`), so one draw fills several
textures in one pass. A ping-pong pair of them is a `SwapFbo`, swapped as one and used as its current target 0; its
`target(i)` is a `SwapTarget`, a texture that is always the current framebuffer's target i, so a port or a pass
keeps the pair or the target, never one buffer of the pair. `LayeredFbo` is the framebuffer over all layers of a
`Texture2DArray`, for the 3D steps. `Swap` is the pair itself, `TextureSwap` the pair of textures
`SwapFbo`, `SwapTexture3D` and `SwapTexture2DArray` stand on.

A readback or an upload that must not wait goes through a `PixelBufferPool`: pixel buffers of one size on the pack
or the unpack target, mapped once for good or copied out with `read`, each with the fence of the transfer queued into
it last (`fence`, then `passed` on a later tick, without waiting). The caller keeps which buffers are free and does
the transfer itself, the buffer bound: `ScreenCapture`, `ScreenPreview` and the bridge's `VelocityPeak`.

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

A texture is held in GL's row order; a CPU image is flipped once, at its upload (`Image`, `Tensor`). An image
still to be turned or flipped, a camera's frame as delivered, is turned in that same draw: `Image.set_image` takes
the `src` `Quad` it is drawn from, and a quarter turn swaps the texture's width and height. Text is laid
out in pixels on the CPU (`text_layout`), from the size its caller gives, and handed to the shader in 0..1. Its
default font, Roboto Mono, sits beside it in `text/`, with its licence (SIL OFL 1.1, `RobotoMono-OFL.txt`).

## Shaders

`shaders/` groups its shaders by kind, one subpackage each: `draw` (noise among them), `color`, `filter`, `mask`,
`composite` (the wipes among them). The package exports every one, so a caller writes `shaders.FillRect`; the kind
orders the files, the operation names the shader. The shared GLSL includes sit at the top of `shaders/`:
`position.glsl` (below), `color.glsl` (Rec. 709 luminance, the HSV conversions), `blend_modes.glsl` (the textbook
blend modes, which the `Blend` shader draws by `BlendShaderMode`; GL's own blending, `BlendState.BlendMode`, does normal,
add, multiply and screen in the draw itself, without a pass) and `colormap.glsl`, turbo, viridis and jet by the number `gl.Colormap` gives them. An effect that runs
over a chain of passes owns its targets and sits in the kind of the shader it chains: `shaders.Bloom` in `filter`
(a dual-filter chain) and `shaders.DistanceField` in `mask` (jump flooding).

A shader's `.frag`, `.vert` or `.comp` source sits beside the `.py` that holds its class, named after it; the loader
resolves it by that module name, and a shader without its own `.vert` takes the shared `shaders/_generic.vert`.
Every sampler names its texture unit in the source, `layout(binding = N)`, and the Python binds its texture to that
unit with `texture.bind(N)`, one `glBindTextureUnit`; no pass sets a sampler's unit with `glUniform1i` or changes
the active unit. An upload writes a texture by its name (`glTextureSubImage2D`), without binding it. A sampler in a shared include has one unit in every shader
that includes it.
`ComputeShader` is a `Shader` with one stage, its `.comp`. A shader object is its own; its class's program is compiled
once, at the class's first `allocate`, shared by every object of the class (`Shader._programs`), and freed at its last
`deallocate`. A uniform's location is read once after linking; a name the program does not have warns once.

Hot reload, where an app turns it on (`Shader.enable_hot_reload`): the watcher's thread marks each program made from a
changed file, an include or `_generic.vert` included (changed: its write time or size differs from when the program
read it, since Windows reports a read as a change too), and the window loop compiles the marked programs again at the
top of the next frame (`Shader.reload_marked`), once their files have been quiet for 0.1 s; a failed compile keeps the
program it had, and a file that reads empty, a save under way, is tried again next frame. A compile error names the
file and line it is in: each include is framed by `#line` directives that number it, and the driver's log is named by
that table (`name_error_lines`). A shader takes a color as a
`Color` and a rectangle as a `Rect`; a pass reads its input's texel size from the texture itself (`textureSize`). `scripts/shader_check.py`
draws every shader of `modules/gl` once on known input and holds it to a reference image, within 1/255 per channel.
