# The GL's plan

The GL held up to the rules (2026-11-15, the apps' plan F). Built: the shader check, smile's shaders out of the GL,
one shaders folder, a texture that draws itself and `gl.Context` (described in [GL.md](GL.md)), time from
`perf_counter`, `FPS` over a window in seconds, `Text` asking GL nothing, the shaders' names (G: inputs by role,
camelCase uniforms, no dead code), the shaders' coordinates (H: every 2D position 0..1 from the top left, one
conversion in `position.glsl`), the capture asking GL nothing (I), the inputs by type (J: `Color`, `Rect`, the texel
size from the texture), the duplicate shaders merged (K: `FillRect`, `StrokeRect`, `FillCircles`, `BarnDoorWipe`, one
`Blur` with `sigma` in pixels, `Blend` for the mask, `texture.draw` for every copy and every part without a turn, one
window manager test).

The lines below come from the review of every shader in `modules/gl`, `modules/render`, the apps and the flow's
general ones, decided with Matthias item by item (2026-11-15). Names are textbook names he understands at once. The
fluid's own simulation shaders are domain shaders and stay out. Each line is built on Matthias's "go", which covers
the harness runs and the `-sim` runs it names; a line that keeps every look is checked by the harness at max 0, a line
that changes a look by the harness's new reference and the app. What the code shows is marked **(seen)**, what is
inferred **(inferred)**.

| Line                     | What                                                              | Changes a look              |
|--------------------------|-------------------------------------------------------------------|-----------------------------|
| L. Subfolders            | the shaders by kind, first, so every later shader lands in place  | no                          |
| M. Textbook names        | the renames below                                                 | no                          |
| N. Textbook algorithms   | the LUT, corners on the CPU, `MaskAA`, colormaps, HSV, the blur   | the LUT, hd_trio's trail    |
| O. Shared GLSL           | `color.glsl`, `blend_modes.glsl`, one `Blend` shader              | hd_trio's foreground, a bit |
| P. Into the GL           | general shaders out of render and the flow                        | no                          |
| Q. Out of the GL         | `DepthAlign` beside its CUDA twin                                 | no                          |
| R. New                   | the standard shaders the pipeline lacks                           | new                         |

## L. Subfolders

- L1. `modules/gl/shaders/` gets one subpackage per kind, as the visual libraries group theirs: `draw`, `color`,
  `filter`, `mask`, `composite`, `transition`, `generate`, `video`. Each shader moves in with its `.frag`/`.vert`;
  the shared includes (`position.glsl` and those of O) stay at the top, where the loader finds them
- L2. `shaders/__init__.py` exports every subpackage's classes, so callers keep `shaders.FillRect`: gl.md's rule names
  a shader `shaders.<Operation>`, CLAUDE.md's has other packages import from a package's `__init__.py`, and the kind
  adds nothing the operation's name does not already say. No caller changes but the two that import a shader's file
  inside the GL: `text.py` (`.shaders.fill_rect`, `.shaders.glyphs`) and the harness (`modules.gl.shaders.blur`)
  **(seen)**; the harness holds every shader at max 0
- L3. The harness gets a case, its reference taken before anything changes, for every shader the plan touches that
  it lacks **(seen)**: the flow's `HSV`, `GaussianBlur`, `Add`, `Set` and `Trail`; `GuidedFilter`, `MaskMultiply`,
  `MaskAA`, `HueShift`, `CelShade`, `DensityColorize` and `DepthView`. Each later line shows max 0 against them, or
  what changed where a look changes (`CelShade`, the trail)

## M. Textbook names

| Now                              | Becomes                                                       |
|----------------------------------|---------------------------------------------------------------|
| `Contrast`                       | `BrightnessContrast`                                          |
| `MaskHardStep`                   | `Threshold`                                                   |
| `EdgeSketch`                     | `Sobel`                                                       |
| `DetailThreshold`                | `AdaptiveThreshold`                                           |
| `NoiseBlend`, `NoiseSimplexBlend`| one `GradientWipe(src, dst, gradient, progress, softness)`; the per-channel variant goes |
| `Noise`, `NoiseSimplex`          | `WhiteNoise`, `SimplexNoise`                                  |
| `Tint`                           | `ColorFill`                                                   |
| `Blend(src, dst, fade)`          | `Mix(src, dst, amount)`                                       |
| `AddDodgeBlend`                  | `AddMix`                                                      |
| `MSColorMask`                    | `LayerComposite`                                              |
| `MaskDilate`, `MaskErode`        | `Dilate`, `Erode`, in `filter`                                |

`Exposure` (its uniforms are already `exposure`, `offset`, `gamma` **(seen)**), `Lut` and `BarnDoorWipe` keep their
names. The noise blends make no noise: they take it as a texture **(seen)**. The settings `blend_factor` (hd_trio's
mask, the foreground look) name a smoothing factor, not the shader, and keep their name. `AddMix` and
`LayerComposite` stay in `render/modules/pose`: their one caller is the similarity silhouette, `AddMix`'s curves are
tuned for it.

## N. Textbook algorithms

- N1. `Lut` samples at the LUT's texel centers, `(c * (N - 1) + 0.5) / N` **(seen: it samples at `c` now)**; the
  layer compositor's look shifts slightly toward the LUT's intent
- N2. `texture.draw` takes `src` as a `Rect` or a `Quad` (four `Point2f`, top left, top right, bottom right, bottom
  left, in 0..1 from the top left) worked out on the CPU; a `Rect`, a flipped one included, is the `Quad` of its
  corners, so every caller of today stays. `DrawRoi` and smile's `BlitRotated` go **(seen)**; hd_trio's turned
  centering builds its `Quad` in `CenterGeometry`
- N3. `MaskAA` keeps its one used mode (`adaptiveSmooth`, a 0.5 threshold antialiased by `fwidth`); the five others go
  and its texel uniform goes for `textureSize` **(seen)**
- N4. `colormap.glsl` with `Turbo`, `Viridis` and `Jet`, chosen by an `IntEnum`; `DepthView` takes it as a setting,
  `Jet` its default so it looks as now
- N5. The flow's `HSV` is the textbook one (the hue wraps, saturation and value clamp to 0..1) **(seen)**: it replaces
  the GL's `Hsv`, which no app calls; the bridges draw exactly as now
- N6. The flow's `GaussianBlur` is no Gaussian: hand weights, 40% on the center, at most 6.33 pixels **(seen)**.
  `SmoothTrail` takes the GL's `Blur` at the σ nearest its look, 0.236 × `blur_radius` pixels (1.9 at hd_trio's 8)
  **(deduction from its weights)**; its setting becomes `blur_sigma`, the presets with it; the flow's goes

## O. Shared GLSL

After N, so the flow's `HSV` is gone and the name `Blend` is free.

- O1. `color.glsl` holds `rgb2hsv` and `hsv2rgb`, now copied in gl `Hsv`, `DensityColorize`, `CelShade` and
  `HueShift` **(seen)**; they include it
- O2. `color.glsl` holds `luminance`, Rec. 709 (0.2126, 0.7152, 0.0722), for every shader that weighs brightness:
  `GuidedFilter`, the motion's `Luminance`, `CelShade`, `Sobel`, `AdaptiveThreshold` (the last three on Rec. 601 now
  **(seen)**). `CelShade`'s bands and blur, hd_trio's foreground look, shift slightly
- O3. `blend_modes.glsl` holds the blend modes, the full textbook set (darken, multiply, color burn, linear burn,
  lighten, screen, color dodge, add, overlay, soft light, hard light, vivid light, linear light, pin light, hard mix,
  difference, exclusion, subtract, divide, hue, saturation, color, luminosity), and smile's `Omission`'s own (reflect,
  glow) **(seen)**; `Omission` includes it and draws as now
- O4. One blend mode enum, `Style.BlendMode`, lists every mode of O3. `Style` sets the ones GL's blending does in the
  draw itself (normal, add, multiply, screen, and the existing premultiplied, sum, revealage); a `Blend(src, dst, mode,
  amount)` shader does the rest in a pass of its own into a target of its own **(seen: `Style.BlendMode` exists)**.
  One system: a mode is named once, and where it runs follows from what GL can do

## P. Into the GL

- P1. `GuidedFilter` from `render/modules/flow` (hd_trio's mask, the flow's view) **(seen)**
- P2. `MaskMultiply` from `render/modules/flow`, beside `MaskApply` **(seen)**
- P3. `MaskAA`, after N3, from `render/modules/flow`, as `MaskAntialias`
- P4. `HueShift` from `render/modules/inference`, as `Colorize` **(seen)**
- P5. One `WeightedSum` (`a * x + b * y`) replaces the flow's `Add`, `Set` and `Trail` **(seen)**. The bridge's own
  `WeightedSum` stays the bridge's: it sums up to seven sources, or their vectors' lengths for an amount, capped, into
  RGB alone **(seen)**

Kept where they are, as domain shaders: render's `TimeSeries`, `Feature`, `Window`, `PosePointLines`,
`PanoramicStitch`, `DensityColor`, `DensityColorize`, `DepthMask`, `DepthView`, `CelShade`; smile's `FaceColor`,
`FlashIn`, `FlashOut`, `Omission`; white_space's `LightSimulation`; the motion's `Luminance`.

## Q. Out of the GL

- Q1. `DepthAlign` moves beside its CUDA twin, `modules/inference/depth_align.py`, as its reference: no app calls it,
  `scripts/depth_align_check.py` compares the two **(seen)**

## R. New

Standard shaders of the visual libraries, all decided yes:

- R1. Tone mapping, ACES and Reinhard by an `IntEnum`: the fluid renders in RGBA16F **(inferred)**, its display is 8-bit
- R2. Dither, ordered and blue noise: 16-bit gradients band at 8 bits, at 4K most
- R3. Bloom on a dual-filter (Kawase) blur: a wide glow in few passes
- R4. Jump flooding: a mask's distance field, for outlines and glows of any width
- R5. Levels (black, white and gamma, in and out) and a color matrix (saturation that keeps luminance)
- R6. Bilateral and median filters
- R7. Displace, posterize, mosaic, chroma and luma key, vignette

Each new shader gets a harness case and its reference on the first run, which I look at before it is kept. Jump
flooding alone is also held to the exact distance, worked out in numpy: it is an approximation whose known misses a
look does not show.

## Documents and commits

- `GL.md` (the subfolders, the includes, the blend modes), `RENDER.md` (the shaders that leave render) and the flow's
  documents that name a moved shader follow in the line that moves it
- The flow's shaders are changed here; the flow's threads hear of N5, N6 and P5 once each works
- Each line, L to R, is one commit

## Rules it touches

- `CLAUDE.md` Packages, a package is one thing; Imports, import another package from its `__init__.py`: L
- `gl.md`, a shader is named after the operation it does, the `shaders` package in front: L, N
- `gl.md`, simple geometry is worked out on the CPU: N2
- `gl.md`, `modules/gl/shaders` holds the shaders useful to any app; an app's own shader sits in that app: P, Q
- `CLAUDE.md` API evolution, remove obsolete pathways; keep presets in sync: N, O, P5
- `CLAUDE.md` Types, prefer `IntEnum` for identifiers: N4, O4, R1
- `CLAUDE.md` Running code, run a GPU harness only on the user's go: every line
