# GL plan

Shaders that reload and report errors well, fewer GL calls on the main thread, and the package held to the rules.
F and G come from the Cinder research (2026-11-20), argued from the code and not measured, so G starts with a
measurement; E and H from the package scan of the same day. The lines are in work order.

## F. Shaders

Shaders reload at one point in the window loop, and each class's program is compiled once and shared by its
objects (`Shader._programs`).

- F4. The uniforms are read once after linking; setting an unknown name warns once. Test: a misspelled uniform warns
  once.
- F5. A save that arrives in two events or half-written is tried again on the next frame, not reported as a compile
  error; the debounce is per file, not per folder. Test: a half-written file, then the whole one; two files saved
  together both reload.
- F6. `Dither` and `Lut` hold their own state per object, made in `allocate` and freed in `deallocate`; `Lut`'s
  `hasattr` guard goes. Test: with the noise mocked, a `Dither` makes its noise once and frees it once.
- F7. GL.md's Shaders section describes hot reload, `ComputeShader`, the program shared by a class's objects, the
  reload point, F2 to F5, and the `_generic.vert` fallback as it is.

## G. Fewer GL calls

Standard GL 4.5, not from Cinder. Each PyOpenGL call holds the GIL on the main thread; how much that costs is not
known yet.

- G0. Measure: a counter of GL calls per frame and the time of one PyOpenGL call, on deep_flow and white_space -sim.
  G1 and G2 go ahead only if the numbers show a gain worth their size; rectangles drawn one quad each are counted too.
- G1. `layout(binding = N)` on samplers: no `glUniform1i` per texture, per pass, per frame. The one shader below
  `#version 420` (hd_trio's `temperature_bridge_shader.frag`, 410) is raised; `bind_texture*` of `Shader` and
  `ComputeShader` lose their `uniform_name`, and every caller with them. Check: `scripts/shader_check.py` and G0's
  count.
- G2. Direct state access: `glBindTextureUnit` for a texture's bind and `glTextureSubImage2D` for its upload; the CUDA
  upload in `tensor.py` from 6 calls to about 3, its `glGetError` behind the debug flag. Every caller of
  `Texture.bind`/`unbind` moves with it. Check: `scripts/shader_check.py` and G0's count.
- G3. A debug context: GLFW's debug hint at window creation and a synchronous `glDebugMessageCallback` that logs the
  failing call, behind a flag that is off by default (it stalls each call); the ctypes callback is kept so it is not
  collected.

## H. The package held to the rules

- H1. One format table (`texture_format.py`) replaces the three in `texture.py`, `image.py` and `tensor.py`; 16F stays
  `GL_FLOAT` for `Texture` and `GL_HALF_FLOAT` for `Tensor`, said where it differs. Test: for each format, the table
  gives what the three gave.
- H2. The default placement is read from `WindowSettings`' defaults: `WindowManager._DEFAULT_*` go, the button's
  description drops the numbers.
- H3. Tests for `Style`'s state (a repeated call skipped, buffer modes, push and pop), `clear_value`, and `fit` and
  `fill`.
- H4. `Profiler` goes: hd_trio's `fluid_2d` times its passes with `GpuTimer`, which does not wait on the GPU. Check:
  hd_trio's tests and pyright.
- H5. `frame_time` becomes `frame_timestamp` (glossary: a stamp is `timestamp`): `render_base.py`,
  `window_manager.py`, the low_flow and deep_flow renders, and `shaders.SimplexNoise.use` with its callers.
  `artnet_bars.py`'s `frame_time`, a duration, stays.
- H6. `Image`'s `channel_order` becomes an `IntEnum`, every caller with it.
- H7. `__init__.py` drops the exports only gl's own tests use (`Window`, `WindowInput`, `WindowManager`, `Monitor`,
  `MonitorId`); the tests import them from their files.
- H8. `Style` gets the blend equation (min, max), and flow's `particle_density.py` sets it through `Style`;
  modules/flow's owner is told when it is built. Check: flow's GPU check of the particle density.
- H9. GL.md: the `video` shader kind goes; `window_systems_research.md` leaves the repo for the scratchpad, and The
  windows section's link to it becomes one line on Cinder as the model; the narrative traces in the pacing section
  go; "frame time" for a duration stays apart from `frame_timestamp`.

## Rules this plan touches

- gl.md: Prefer as few GL calls and state switches as possible
- gl.md: Prefer not to read back from the GPU or wait on it inside a frame
- gl.md: All OpenGL calls execute on the main thread, in the window loop
- gl.md: Pair every creation with its destruction
- gl.md: Name a shader's `.frag`/`.vert`/`.comp` source after the `.py` file that holds its class, beside it; the loader resolves it by that module name
- gl.md: Shaders get checked in a GPU harness
- glossary.md: A time stamp: `timestamp`; not `time`, `time_stamp`
- docs.md: Leave out how a conclusion was reached, what the code used to do, and what changed; that belongs in commit messages
- CLAUDE.md: Fix a problem where it is made, not where it shows
- CLAUDE.md: Prefer the design with the lower latency, measured in the context of the app
- CLAUDE.md: Its files each hold one class or one subject, named after it
- CLAUDE.md: Its `__init__.py` holds what other packages use
- CLAUDE.md: Inside a package, its subpackages included, import relatively; tests may import absolutely
- CLAUDE.md: Put all imports from one package on a single `from` line
- CLAUDE.md: Define module loggers as `logger = logging.getLogger(__name__)`
- CLAUDE.md: Give public methods return type hints
- CLAUDE.md: Prefer `IntEnum` over string keys for dict lookups and identifiers
- CLAUDE.md: Log exceptions with context; never fail silently
- CLAUDE.md: Rely on running the real-time system for integration; unit-test infrastructure (settings, frame/features, serialization, utility primitives, geometry)
- CLAUDE.md: Update all affected call sites in the same change
- CLAUDE.md: Remove obsolete pathways instead of leaving parallel legacy ones
- CLAUDE.md: Run a test of the app or a GPU harness only on the user's go, and only what the plan names
- CLAUDE.md: A commit leaves no pyright errors
