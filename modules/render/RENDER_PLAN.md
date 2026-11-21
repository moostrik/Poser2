# Render plan

What is left to do in `modules/render`, from the user's naming call and the package scan of 2026-11-21 (after c866306e). Each step gives its why (seen in the code, or a deduction) and its check.

These stay as they are, on the user's word:

- The similarity silhouette's thresholds stay in `SimilaritySilhouetteLayer`.
- The 2D and 3D flow layers stay separate until the flow is finished: their code is not shared and their files are not regrouped.
- `CropLayer` stays in `modules/inference`, because it cuts the image to an inference crop.

The pixel buffer and fence code of `general/screen_recording` is not in this plan: `gl.PixelBufferPool` holds it, and `ScreenCapture` and `ScreenPreview` use it (GL plan D1, built in d1be9840). The pack and unpack alignment constants and their set-back in those two files go in GL's alignment item, where every transfer sets its own alignment.

## D. Settings and work

- **D1** hd_trio gets a `data_time` group in its render settings: like white_space's, a `_MotionTimeDrawSettings` subclass whose `stage` is a `Field[Stage]` of hd_trio's own `Stage`, defaulting to `Stage.RAW` (0, what it draws now), and passes it to its `MotionTimeDraw`s. `MotionTimeDraw` then takes its settings as required, and its `settings or …()` fallback goes. hd_trio's settings files (`studio.json`, `umu.json`) get the `data_time` key at stage 0, so they stay in step with the settings classes.
  - Why: hd_trio builds `MotionTimeDraw(i, self.board)` without settings (`hd_trio_render.py`), so the panel cannot set its stage and the class keeps a fallback for that one caller, against "Components take their settings object (or a `Group` of it) in the constructor" (seen). The user said yes.
  - Check: pyright; both settings files load into the new group with no warning (as `test_flow_systems` loads the flow's); and one hd_trio -sim with `data_time` shown, which draws the RAW stage's motion time as before. The settings files show no `data_time` layer (seen), so a run without it would show nothing of D1.
- **D2** `FeaturePlot` draws a layer's labels into its label Fbo only while `draw_labels` is on; `FeatureFrameLayer` keeps `draw_labels` off by default.
  - Why: the frame layer draws its labels whenever they change and `render_labels` is on, but never shows them, because its `draw_labels` is False and nothing sets it (seen; the same before A3). That is text drawn every change for nothing. The user said yes.
  - Check: pyright, the render tests, and white_space -sim with the data layers in frame and window mode (window labels still shown).

## Rules it touches

- Settings: "Components take their settings object (or a `Group` of it) in the constructor, never as a global" (D1)
- Performance: "Prefer the design with the lower latency, measured in the context of the app" (D2)
