# Render plan

What is left to do in `modules/render`, from the package scan of 2026-11-21. Each step gives its why (the finding it
rests on, seen in the code or a deduction) and its check. Callers are from the deep-flow copy of 2026-11-20.

These stay as they are, on the user's word:

- The similarity silhouette's thresholds stay in `SimilaritySilhouetteLayer`.
- The 2D and 3D flow layers stay separate until the flow is finished: their code is not shared and their files are not regrouped.
- `CropLayer` stays in `modules/inference`, because it cuts the image to an inference crop.

The pixel buffer and fence code of `general/screen_recording` is not in this plan: `gl.PixelBufferPool` holds it, and `ScreenCapture` and `ScreenPreview` use it (GL plan D1, built in d1be9840). The pack and unpack alignment constants and their set-back in those two files go in GL's alignment item, where every transfer sets its own alignment.

## A. Code that does the same, written once

- **A1** `ScreenCapture` hands its frames out through a `utils.Broadcast`, which replaces `_callbacks`, `_callback_lock` and the copy in `poll`. With no callback registered, the frame is released at once, as now.
  - Why: a hand-rolled callback list, where one failing callback stops the ones after it (seen).
  - Check: `modules/render/tests` and a short screen recording take in deep_flow -sim.
- **A2** One layer for an image taken from the board replaces the copied update in `CameraImageLayer`, `CropImageLayer`, `SegmentationMaskLayer` and `SegmentationForegroundLayer`. It is built with the board read it shows (a callable) and its texture's wrap and border. It caches the read, clears on lost, keeps the stream (`record_stream`), uploads, and keeps the `timestamp` where the read has one. The four class names go: one class takes the read, and four snake_case builder functions in render's `modules/camera` and `modules/inference` (`camera_image_layer(cam_id, board)`, `crop_image_layer`, `segmentation_mask_layer`, `segmentation_foreground_layer`) build it, so the call sites stay readable. `InferenceFrameImageLayer` and `InferenceFrameDepthLayer` keep their own shape: they are handed a frame and tell it by its timestamp.
  - Why: the four repeat one update, and the mask and foreground layers differ only in the field they read and their border (seen).
  - Call sites: the four apps' renders (all use the camera image and crops; hd_trio also uses the mask and foreground) and `PoseCompositor`.
  - Check: hd_trio -sim (must look the same), white_space -sim and deep_flow -sim.
- **A3** `FeatureFrameLayer` and `FeatureWindowLayer` share their mode check, colors, clear, draw and label drawing once, by composition.
  - Why: `_is_active`, `clear`, `draw` and `_resolve_colors` are the same word for word, and the label drawing differs only in where a label sits (seen).
  - Call sites: the four apps.
  - Check: white_space -sim, hd_trio -sim and deep_flow -sim, with the data layers in frame and window mode.
- **A4** The pose box-to-image transform becomes one method on `Points2D` (`modules/pose/features/points_2d.py`). It takes the box (`Rect`) and gives the points in image coordinates, keeping their scores. `PoseRenderer` and `PoseLineLayer` use it, and their `_transform_to_image_space` goes.
  - Why: `_transform_to_image_space` is written twice, and gl.md says "Simple geometry, as from poses, is worked out on the CPU, in plain modules" (seen).
  - Check: a new unit test in `modules/pose/tests`, and white_space -sim.
- **A5** `PanoramaCompositor` computes the camera's sector (`target_fov`) once and hands it to `GridRenderer` and `SeamRenderer`.
  - Why: `360 / num_cams` is written three times, while the compositor says it owns the strip's geometry (seen).
  - Check: the panoramic tests, and white_space -sim with the panorama shown.

## B. The tree

- **B1** `modules/pose` groups its files by job:
  - `features/` holds the data layer settings and the feature frame and window layers.
  - `lines/` holds the pose line layer, the pose renderer and the view pose layer.
  - The box renderer, the compositor, the motion time renderer and the silhouette stay at the root.
  - Why: ten loose files (seen); "A package with many loose files groups them in folders by job".
- **B2** `DataCache` gets its own file and a unit test.
  - Why: `layer_base.py` holds three subjects, and `DataCache`, a utility primitive, has no test (seen).
- **B3** `general/composite/layer_compositor.py` and `general/grade/lut_grade.py` move directly into `general/`, and their one-file packages go. Their importers follow: `render/__init__.py` and the 2D and 3D fluid layers.
  - Why: "Prefer no folder for one or two files", and rendering.md asks for packages only in `modules` (seen).
- **B4** On the PC only (git does not track them): delete the folders `modules/render/layers/` and `modules/render/shaders/`, which hold only stale `__pycache__`.
- Check for B1 to B3, one commit: the render tests, the guard tests (import layering), pyright and the new `DataCache` test. They are moves, so no app run.

## C. Names

- **C1** These names change everywhere:
  - `CameraReadingsLayer` becomes `CameraReadingsRenderer`; call sites: hd_trio's and white_space's renders. Why: it owns no texture, which rendering.md calls a `Renderer` (seen).
  - `PanoramaLayerSettings` becomes `PanoramaCompositorSettings`, in `panoramic/panorama_compositor_settings.py`; call sites: white_space's settings and render. Why: the glossary says `<ClassName>Settings`, and `settings.py` is named after no class (seen).
  - grey becomes gray, including `DepthDrawMode.GRAY`, which nothing outside render uses; centre becomes center, neighbour becomes neighbor, labelled becomes labeled. Why: the glossary's American spelling (seen).
  - The settings files keep their group keys (`panorama`), so they need no migration (deduction: confirm in white_space's settings files).
  - Check: pyright and the guard tests (glossary).

## D. Time

- **D1** `PanoramaCompositor` takes the frame's time from white_space's render each tick (`set_frame_time`, from the window's `frame_time`) and hands it to `MarkContext`; `time.perf_counter()` goes.
  - Why: the marks' fade reads a clock in the render, against "Downstream of the camera, take time from the frame or the tick instead of reading a clock" (seen).
  - Check: white_space -sim, where a lost mark fades as before.

## E. Leftovers and documents

- **E1** Leftovers go:
  - stale docstrings: DataHub, GPUFrame, `CenterMaskLayer`, FeatureBand, the `Window` shader's base-class text and uniform list, `ColorSettings`'s history, `MotionTimeRenderer`'s bottom-left
  - commented-out code and prints in the silhouette layer
  - unused imports
  - `TrackletRenderer`'s second `REMOVED` branch
  - the `settings or …Settings()` fallbacks on required settings
  - `LayerCompositor`'s `hasattr` filter
  - the `LayerCompositor` "Usage:" block, and the reasons in `CameraReadingsRenderer`'s docstring (to `RENDER.md`)
  - Why: "Remove obsolete pathways", and "A docstring says what the thing is" (seen).
  - Check: pyright, and hd_trio -sim for the silhouette and the compositor (must look the same).
- **E2** The documents:
  - `RENDER.md` names `general/` as it is after B3, with `screen_recording/`.
  - The panorama's reasons leave the docstrings of `PanoramaCompositor`, `PanoramaBlend` and `PanoramaCompositorSettings` for `apps/white_space/docs/CALIBRATION.md`, the one user's calibration document.
  - Why: `RENDER.md` leaves out `screen_recording`, and those docstrings hold reasons and a table (seen).

After the last step: all four apps start once.

## Rules it touches

- Packages: "Its files each hold one class or one subject, named after it"
- Packages: "A package with many loose files groups them in folders by job"
- Packages: "Prefer no folder for one or two files"
- Packages: "Inside a package, code that does the same is written once"
- Concurrency: "A callback list is a `utils.Broadcast`"
- Concurrency: "A failing callback does not stop the others"
- Composition and wiring: "Prefer composition over inheritance for runtime assembly"
- Time: "Downstream of the camera, take time from the frame or the tick instead of reading a clock"
- Docstrings: "A docstring says what the thing is"
- Docstrings: "Reasons sit in the package's document, history in the commit"
- API evolution: "Remove obsolete pathways instead of leaving parallel legacy ones"
- API evolution: "Update all affected call sites in the same change"
- gl.md: "Simple geometry, as from poses, is worked out on the CPU, in plain modules"
- rendering.md: "`modules` holds the layers that draw a module's data, in a package named after the module"
- rendering.md: "`Renderer`: draws into a target it is given and owns no texture"
- glossary.md: "A settings class: `<ClassName>Settings`, after the class it configures"
- glossary.md: "American spelling: `center`, `visualization`, `color`; not `centre`, `visualisation`, `colour`"
