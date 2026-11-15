# The GL's plan

The GL held up to the rules (2026-11-15, the apps' plan F). Built: the shader check, smile's shaders out of the GL,
one shaders folder, a texture that draws itself and `gl.Context` (described in [GL.md](GL.md)), time from
`perf_counter`, `FPS` over a window in seconds, `Text` asking GL nothing, the shaders' names (G: inputs by role,
camelCase uniforms, no dead code), the shaders' coordinates (H: every 2D position 0..1 from the top left, one
conversion in `position.glsl`), the capture asking GL nothing (I), the inputs by type (J: `Color`, `Rect`, the texel
size from the texture), the duplicate shaders merged (K: `FillRect`, `StrokeRect`, `FillCircles`, `BarnDoorWipe`, one
`Blur` with `sigma` in pixels, `Blend` for the mask, `texture.draw` for every copy and every part without a turn, one
window manager test), the shaders by kind in subpackages with harness references for the twelve outside the GL the
plan changes (L), the textbook names (M: `BrightnessContrast`, `Threshold`, `Sobel`, `AdaptiveThreshold`, one
`GradientWipe`, `WhiteNoise`, `SimplexNoise`, `ColorFill`, `Mix`, `AddMix`, `LayerComposite`, `Dilate`, `Erode`),
the textbook algorithms (N: the LUT at its texel centers, within 0.8/255 of a numpy lookup; `texture.draw` of a
`Quad`, `DrawRoi` and smile's `BlitRotated` gone; `MaskAA` one mode; `colormap.glsl` with `gl.Colormap`; the flow's
HSV as the GL's; the 2D trail on the GL's `Blur` at σ 1.7, measured nearest the old blur, 5/255 apart on average and
4-channel since), the shared GLSL (O1 to O3: `color.glsl` with Rec. 709 `luminance` and the HSV conversions in seven
shaders, `blend_modes.glsl` with the textbook set), the general shaders into the GL (P: `GuidedFilter`,
`MaskMultiply`, `MaskAntialias`, `Colorize`, and `WeightedSum` for the flow's `Add`, `Set` and `Trail`; the bridge's
own `WeightedSum` stays the bridge's), `DepthAlign` out of the GL (Q: beside its CUDA twin in
`modules/inference/shaders`), the standard shaders the pipeline lacked (R: `ToneMap`, `Levels`, `ColorMatrix`,
`Posterize`, `Dither` with an ordered and a blue-noise map, `Vignette`, `Mosaic`, `Displace`, `Median`, `Bilateral`,
`ChromaKey`, `LumaKey`, `DualFilter`, and the effects `Bloom` and `DistanceField`, the last held to the exact
distance), the `Blend` shader (O4: `blend_modes.glsl`'s 26 modes by `BlendShaderMode`, its numbers held to the
GLSL's by a test; `Style.BlendMode` keeps GL's own, so a layer's dropdown offers what it can do).

No step is left.
