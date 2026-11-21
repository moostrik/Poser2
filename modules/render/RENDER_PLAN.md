# Render plan

What is left to do in `modules/render`, from the user's naming call and the package scan of 2026-11-21 (after c866306e). Each step gives its why (seen in the code, or a deduction) and its check.

These stay as they are, on the user's word:

- The similarity silhouette's thresholds stay in `SimilaritySilhouetteLayer`.
- The 2D and 3D flow layers stay separate until the flow is finished: their code is not shared and their files are not regrouped.
- `CropLayer` stays in `modules/inference`, because it cuts the image to an inference crop.

The pixel buffer and fence code of `general/screen_recording` is not in this plan: `gl.PixelBufferPool` holds it, and `ScreenCapture` and `ScreenPreview` use it (GL plan D1, built in d1be9840). The pack and unpack alignment constants and their set-back in those two files go in GL's alignment item, where every transfer sets its own alignment.

## B. Panel text

- **B1** The render's settings fields and panels shown to the user get their one-line description, with the unit where there is one (seen: 24 fields and 17 panels with none; a hidden one, `visible=False`, is left out):
  - Fields: `DataLayerSettings` (8), `LayerCompositorSettings.blend_mode`, `LutSettings.strength`, `SimilaritySilhouetteLayerSettings.blend_mode` and `max_players`, and the 12 colors of `ColorSettings`.
  - Panels: `LayerCompositorSettings.lut`, and in `modules/flow`'s layers (the flow threads' files) `monitor_main` and `monitor_debug` of the motion, color, bridge, 2D fluid and 3D fluid settings, the 2D fluid's `lut`, the 3D fluid's `composite` and its composite's `lut`, the 2D and 3D monitors' `advanced` and the 3D monitor's `orbit`.
  - Why: "Prefer a description for every field and panel shown to the user", and "Keep `Field` descriptions to one short line of panel text: what the value is and its unit"; these show the user no text.
  - Check: pyright, and the render tests.

## C. Docstrings

- **C1** The reasons left in the panorama's docstrings leave them, and the docstrings keep what each thing is: `marks.py`'s module docstring (27 lines, 14 bold), `GridDraw`, `LabelDraw`, `MarkDraw`, `SeamDraw`, `StitchDraw`, and `strip.elevation_window` (seen). A reason about reading the strip goes to white_space's `CALIBRATION.md`, where most of them are already stated; a reason about the code (such as a label never moving sideways, and flipping its side at the right edge because `Text.draw_box_text` does not clamp) goes to a panorama section in `RENDER.md`.
  - Why: "A docstring says what the thing is", "Reasons sit in the package's document", and docs.md's "Give each document one audience and purpose": `CALIBRATION.md` is the operator's procedure; c866306e moved only `PanoramaCompositor`'s, `PanoramaBlend`'s and `PanoramaCompositorSettings`'s.
  - Check: pyright, and the panoramic tests.

## D. Settings and work

- **D1** hd_trio gets a `data_time` group in its render settings: like white_space's, a `_MotionTimeDrawSettings` subclass whose `stage` is a `Field[Stage]` of hd_trio's own `Stage`, defaulting to `Stage.RAW` (0, what it draws now), and passes it to its `MotionTimeDraw`s. `MotionTimeDraw` then takes its settings as required, and its `settings or …()` fallback goes. hd_trio's settings files (`studio.json`, `umu.json`) get the `data_time` key at stage 0, so they stay in step with the settings classes.
  - Why: hd_trio builds `MotionTimeDraw(i, self.board)` without settings (`hd_trio_render.py`), so the panel cannot set its stage and the class keeps a fallback for that one caller, against "Components take their settings object (or a `Group` of it) in the constructor" (seen). The user said yes.
  - Check: pyright; both settings files load into the new group with no warning (as `test_flow_systems` loads the flow's); and one hd_trio -sim with `data_time` shown, which draws the RAW stage's motion time as before. The settings files show no `data_time` layer (seen), so a run without it would show nothing of D1.
- **D2** `FeaturePlot` draws a layer's labels into its label Fbo only while `draw_labels` is on; `FeatureFrameLayer` keeps `draw_labels` off by default.
  - Why: the frame layer draws its labels whenever they change and `render_labels` is on, but never shows them, because its `draw_labels` is False and nothing sets it (seen; the same before A3). That is text drawn every change for nothing. The user said yes.
  - Check: pyright, the render tests, and white_space -sim with the data layers in frame and window mode (window labels still shown).

## Rules it touches

- Settings: "Prefer a description for every field and panel shown to the user" (B1)
- Settings: "Keep `Field` descriptions to one short line of panel text: what the value is and its unit" (B1)
- Settings: "Components take their settings object (or a `Group` of it) in the constructor, never as a global" (D1)
- Performance: "Prefer the design with the lower latency, measured in the context of the app" (D2)
- Docstrings: "A docstring says what the thing is" (C1)
- Docstrings: "Reasons sit in the package's document, history in the commit" (C1)
- docs.md: "Give each document one audience and purpose" (C1)
