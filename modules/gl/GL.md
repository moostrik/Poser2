# GL

`modules/gl` serves a low-latency 2D pipeline first: each frame goes from camera to screen in as few passes, copies
and waits as possible. A design choice here is judged by what it costs that path.

`modules/gl` is the apps' OpenGL framework: the window, the textures and FBOs, the shaders and their loader, text,
the blend state, and the shaders that are useful to any app. A layer that draws a module's data sits in
`modules/render`, an app's own shader in that app.

## What it serves

The GL serves a pipeline of 2D shader steps; a volumetric step is 3D only within its own pass. Geometry, such as
the poses and the rectangles, is worked out on the CPU and handed to a shader ready to place, so a draw needs no
matrix and no state per target. Each GL call and state switch costs CPU time on the main thread, so the GL keeps
their number low: it holds GL's state on the application side and asks no
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

## The windows

The window loop runs on the main thread: GLFW puts window creation, monitors and event processing there, and all its
callbacks run there. The launcher calls `RenderBase.run()` after `app.start()`; the loop draws on the same thread, so
nothing crosses a thread between the events and the draw. `stop()` works from any thread: it sets a close request,
GLFW's should-close and an empty event, and from another thread waits up to 2 s for the loop to end. The structure
follows Cinder (`window_systems_research.md`).

There is one kind of window (`Window`, `window.py`), set by its own `WindowSettings`: `enabled`, its `monitor`,
`fullscreen`, and its windowed place on that monitor (`x`, `y`, `width`, `height`). An app's root group `windows`
holds the loop's `WindowManagerSettings` (`loop`) and `window_0`, `window_1`, …; their number is fixed at start.
Every window is made at start, window 0 first, the others sharing its objects. A window that is off is hidden, not
destroyed, and is neither drawn nor swapped. Fullscreen is a borderless window the size of its monitor, so the other
monitors stay on. A window whose monitor is missing waits small on the primary monitor with its settings untouched,
and returns when the monitor does. F toggles the fullscreen of the window with focus; Esc, or closing any window,
ends the app. The monitors are `Monitor`s (`monitor.py`) in `MonitorId` order: the primary first, then by x, then y.

The loop, each frame, in window 0's context:

| Step                | What it does                                                                          |
|---------------------|---------------------------------------------------------------------------------------|
| monitors            | read again after a hot-plug (the monitor callback only marks it)                       |
| settings            | each window's settings read and only what differs applied; a user's move or resize     |
|                     | written to the settings once, as applied, and not applied back                         |
| sizes               | `on_window_resize(window, width, height)` for each framebuffer size that changed       |
| update              | `renderer.update()` once                                                               |
| hold                | a wait on the fence of the paced swap before last: one frame in flight                 |
| draw                | each shown window: its context current, `draw(window, width, height)`                 |
| swap                | the shown windows, the `v_sync_window` last, with v-sync, and a fence after it         |
| pace                | the wait for `fps_limit`, when v-sync does not pace the loop                          |
| events              | `glfw.poll_events()`: the input callbacks                                              |

No callback calls GL: GLFW's callbacks store (a move, a size, a hot-plug) or hand over (`WindowInput`, which tells
each key, mouse and scroll callback its window, and logs a raising callback and runs the next). The renderer's
objects live in window 0's context: the loop makes it current at the top of each frame, around `allocate`,
`on_window_resize` and `deallocate`, and makes a window's own context current only for its draw and swap. A window's
size is its framebuffer's, known from `allocate` on (`RenderBase.window_size`). `default_placement` puts every window
at its default place on its own monitor, windowed; so does the start, for a window under half on the monitors.

### Pacing

With v-sync on and the `v_sync_window` shown, its swap paces the loop. Otherwise `fps_limit` does (0 is off; 60 while
every window is hidden): an absolute deadline on `perf_counter`, a period on from the last, waited with the shared
`utils.Waker`, a high-resolution waitable timer, without a spin. A frame that ends after its deadline goes on at once;
one more than a period behind moves the deadline to now.

The Waker's lateness after its deadline, v-sync off (white_space -sim on 20261024-164009_ICK, the launcher's priority
and switch interval, 30 s from 10 s, 2026-11-20), ms:

| Where                         | Mean | p95  | p99  | Max  |
|-------------------------------|------|------|------|------|
| alone, 300 waits at 60 Hz     | 0.34 | 0.56 | 0.79 | 0.91 |
| in the app, `fps_limit` 60    | 0.98 | 3.72 | 5.23 | 6.82 |
| in the app, `fps_limit` 120   | 1.13 | 4.43 | 5.26 | 10.8 |

The timer is on time; inside the app the main thread waits for the GIL behind the app's busy Python threads after it
fires (**(deduction)**, agreed with the Waker's owner). A higher priority was tried in white_space's clock and made the
app stutter. Less Python on the hot paths is what shortens it. The frame time in those runs is the app's own:
17.7 ms mean at 60 fps (p95 29.8, p99 38.2), its update 9.8 ms mean (p95 26.1).

### One frame in flight

After the paced swap the loop puts a fence; before the next frame draws, it waits on the fence of the swap before
last (`_hold_frame_in_flight`), so one frame stays in flight and the driver queues no more. A GPU wait after the swap
(`glFinish`) holds the queue too, but stalls the CPU until the GPU is done, so the CPU's work and the GPU's no longer
overlap; it is not used.

v-sync on window 1 (fullscreen on a 3840 × 2160 monitor), deep_flow -sim on 20261112-125832_measure, 25 s from 10 s,
2026-11-20. "Unfinished" is the paced swaps the GPU has not passed at a frame's start, a fence polled once; ms:

| After the paced swap | Frame mean | p95  | p99  | Blocks in swap | Blocks in hold | Unfinished 0 / 1 / 2 |
|----------------------|------------|------|------|----------------|----------------|----------------------|
| nothing              | 16.9       | 30.5 | 36.2 | 0.41           |                | 9 / 1336 / 131       |
| `glFinish`           | 24.1       | 30.6 | 35.9 | 13.8           |                | 177 / 863 / 0        |
| the fence            | 16.9       | 17.6 | 33.3 | 8.65           | 0.02           | 191 / 1288 / 0       |

Measured: the fence holds the queue at one frame at 59 fps; `glFinish` holds it at 42 fps. That `glFinish`'s cost is the
lost overlap is a **(deduction)**. So is the time moving into the swap with the fence, and its p95 falling from 30.5 to
17.6 ms (one run each): the driver, with one frame queued, waits for the vblank in the swap.

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
`position.glsl` (below), `color.glsl` (Rec. 709 luminance, the HSV conversions), `blend_modes.glsl` (the textbook
blend modes, which the `Blend` shader draws by `BlendShaderMode`; GL's own blending, `Style.BlendMode`, does normal,
add, multiply and screen in the draw itself, without a pass) and `colormap.glsl`, turbo, viridis and jet by the number `gl.Colormap` gives them. An effect that runs
over a chain of passes owns its targets beside the shaders: `Bloom` (a dual-filter chain) and `DistanceField` (jump
flooding).

A shader's `.frag`, `.vert` or `.comp` source sits beside the `.py` that holds its class, named after it; the loader
resolves it by that module name and falls back to `_generic.vert` for the vertex stage. A shader takes a color as a
`Color` and a rectangle as a `Rect`; a pass reads its input's texel size from the texture itself (`textureSize`). `scripts/shader_check.py`
draws every shader of `modules/gl` once on known input and holds it to a reference image, within 1/255 per channel.
