# Render plan

What is left to do in `modules/render`, from the user's naming call and the package scan of 2026-11-21 (after c866306e). Each step gives its why (seen in the code, or a deduction) and its check.

These stay as they are, on the user's word:

- The similarity silhouette's thresholds stay in `SimilaritySilhouetteLayer`.
- The 2D and 3D flow layers stay separate until the flow is finished: their code is not shared and their files are not regrouped.
- `CropLayer` stays in `modules/inference`, because it cuts the image to an inference crop.

The pixel buffer and fence code of `general/screen_recording` is not in this plan: `gl.PixelBufferPool` holds it, and `ScreenCapture` and `ScreenPreview` use it (GL plan D1, built in d1be9840). The pack and unpack alignment constants and their set-back in those two files go in GL's alignment item, where every transfer sets its own alignment.

## D. Settings and work

- **D2** `FeaturePlot` draws a layer's labels into its label Fbo only while `draw_labels` is on; `FeatureFrameLayer` keeps `draw_labels` off by default.
  - Why: the frame layer draws its labels whenever they change and `render_labels` is on, but never shows them, because its `draw_labels` is False and nothing sets it (seen; the same before A3). That is text drawn every change for nothing. The user said yes.
  - Check: pyright, the render tests, and white_space -sim with the data layers in frame and window mode (window labels still shown).

## Rules it touches

- Performance: "Prefer the design with the lower latency, measured in the context of the app" (D2)
