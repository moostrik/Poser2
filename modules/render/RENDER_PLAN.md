# The render's plan

The render package held up to the rules (2026-11-14, the apps' plan F). `modules/render` is 8,000 lines: layers in
seven packages named by kind or by app idea (`cam`, `center`, `data`, `flow`, `generic`, `panorama`, `source`) and
shaders in four (`cam`, `data`, `generic`, `mask`). It is rebuilt as `rendering.md`'s What the render holds has it
(9232b6be, a3da2095): three categories, `modules`, `general` and `shared`, each holding packages named after their
subject, each package its layers and, in `shaders/`, those layers' shaders; a shader no layer owns goes to
`modules/gl/shaders/`; `modules/render/__init__.py` is what apps import from. Who uses what is from the imports
(seen in code). One step per commit, each checked by the render, flow and app tests, the guard tests and every app's
start (-sim, smile on its image input, 10 s), so the apps keep working after each.

Two decisions of this plan, beside the rules:
- One system per idea: hd_trio's center does what the flow's `ViewLayer` does in deep_flow, so it is hd_trio's until
  hd_trio moves onto the flow's view.
- No module imports `modules/render` (true today, seen in code): a guard test keeps it so.

## The tree

```
modules/render/
  modules/
    pose/       PoseCompositor, PoseRenderer, PoseBoxRenderer, FeatureFrameLayer, FeatureWindowLayer,
                MotionTimeRenderer, PoseLineLayer, ViewPoseLayer, SimilaritySilhouetteLayer
    camera/     CameraImageLayer, CameraReadingsLayer
    tracker/    TrackerCompositor, TrackletRenderer, the panoramic tracker's strip (panorama's compositor and renderers)
    inference/  CropLayer, CropImageLayer, SegmentationMaskLayer, SegmentationForegroundLayer, ForegroundLookLayer,
                CaptureImageLayer, CaptureDepthLayer
    flow/       the flow's stage layers, as now
    board/      TimeSeriesLayer
  general/
    composite/  LayerCompositor (now CompositeLayer: it draws other layers into its FBO)
  shared/       created with its first layer: every layer two apps use today draws a module's data or is general
apps/hd_trio/render/                 hd_trio_render.py (now apps/hd_trio/render.py), layers/center/: CenterGeometry,
                                     CenterCamLayer, CenterMaskLayer, CenterForegroundLayer
modules/gl/shaders/                  the render's shaders no single layer package owns (below)
```

## Where each layer comes from

| Now                                   | Used by                    | Goes to                                               |
|---------------------------------------|----------------------------|-------------------------------------------------------|
| `cam/pose_compositor`, `pose_renderer`, `bbox_renderer` | all three; the renderers inside | `modules/pose/` (`PoseBoxRenderer` for `BBoxRenderer`) |
| `cam/tracker_compositor`, `tracklet_renderer` | all three          | `modules/tracker/`                                    |
| `cam/crop_layer`                      | the pose compositor        | `modules/inference/`                                  |
| `cam/camera_readings_layer`           | hd_trio, white_space       | `modules/camera/`                                     |
| `data/*` but the time series          | all three                  | `modules/pose/`                                       |
| `data/time_series_layer`              | deep_flow                  | `modules/board/`                                      |
| `source/image_source_layer`           | all three                  | `modules/camera/` (`CameraImageLayer`)                |
| `source/crop_source_layer`            | all three                  | `modules/inference/` (`CropImageLayer`)               |
| `source/mask_source_layer`, `foreground_source_layer` | hd_trio    | `modules/inference/` (`SegmentationMaskLayer`, `SegmentationForegroundLayer`) |
| `source/capture_image_layer`, `capture_depth_layer` | deep_flow    | `modules/inference/`                                  |
| `center/` geometry, cam, mask, foreground | hd_trio                | `apps/hd_trio/render/layers/center/` (one system per idea) |
| `center/center_pose_layer`            | deep_flow, hd_trio         | `modules/pose/` (`ViewPoseLayer`: the pose at a view's placement) |
| `panorama/*`                          | white_space                | `modules/tracker/`                                    |
| `flow/*`                              | deep_flow, hd_trio         | `modules/flow/`                                       |
| `generic/composite_layer`             | deep_flow, hd_trio         | `general/composite/` (`LayerCompositor`)              |
| `generic/ms_color_mask_layer`         | deep_flow, hd_trio         | `modules/pose/` (`SimilaritySilhouetteLayer`: silhouettes by pose similarity) |
| `generic/foreground_look_layer`       | deep_flow; its chain in hd_trio's center | `modules/inference/` (the segmentation's foreground) |
| `layers/_deprecated/`                 | empty                      | goes                                                  |

## Where each shader goes

A shader goes with the one package whose layers use it. A shader that does a generic operation (a blend, a tint, a
mask applied) is a GL tool however few use it now, and goes to `modules/gl/shaders/` as it is, as does one used by
several packages, by apps only, or by none (its order there is parked in `modules/gl/GL_PLAN.md`). `FeatureShader`,
`WindowShader` and `TimeSeriesShader` lose their `Shader` ending. Owners, seen in code: the foreground look uses
`CelShade`, `HueShift`, `Blend` and `MaskApply`; the silhouettes use `Tint`, `AddDodgeBlend` and `MSColorMask`.

| Shader                                                  | Used by (after the moves)              | Goes to                     |
|---------------------------------------------------------|----------------------------------------|-----------------------------|
| `Feature`, `Window`, `PosePointLines`, `MSColorMask`, `AddDodgeBlend` | pose's layers (and apps) | `modules/pose/shaders/` |
| `TimeSeries`                                            | the time series                        | `modules/board/shaders/`    |
| `PanoramicStitch`                                       | the panoramic strip                    | `modules/tracker/shaders/`  |
| `CelShade`, `HueShift`                                  | the foreground look                    | `modules/inference/shaders/` |
| `DensityColor`, `DensityColorize`, `DepthMask`, `DepthView`, `GuidedFilter`, `MaskAA`, `MaskMultiply` | the flow's layers (and hd_trio's center) | `modules/flow/shaders/` |
| `DrawCircles`, `MaskBlend`, `MaskBlur`, `MaskDilate`, `MaskErode` | hd_trio's center only        | `apps/hd_trio/render/shaders/` |
| `DrawRoi`, `MaskApply`, `Blend`, `Tint`                 | generic operations                     | `modules/gl/shaders/`       |
| `DrawColoredRectangle`, `DrawRectangleOutline`          | apps and several packages              | `modules/gl/shaders/`       |
| `DepthAlign`                                            | no layer (a check script)              | `modules/gl/shaders/`       |
| `EdgeSketch`, `DetailThreshold`, `MaskHardStep`         | none (no import)                       | `modules/gl/shaders/`       |

## Steps

1. **hd_trio's center.** hd_trio's `render.py` becomes the package `apps/hd_trio/render/` (`hd_trio_render.py`, as
   smile and white_space have theirs), so a `render/` folder does not shadow it; then `CenterGeometry`,
   `CenterCamLayer`, `CenterMaskLayer`, `CenterForegroundLayer`, the center's own shaders and its test
   (`test_center_geometry`) move into it, and `modules.render` exports what the center then imports from it
   (`ViewLayer`, `ForegroundLook`, the mask shaders it shares with the flow); `CenterPoseLayer` becomes `ViewPoseLayer`, with deep_flow's and
   hd_trio's call sites. Checked by: the render and app tests, the guard tests, every app's start.
2. **The tree.** The layers and shaders move as the tables have them, the `layers/` and `shaders/` trees go, and
   `modules/render/__init__.py` exports every layer, so an app's `ls.X` becomes `render.X` in the same commit, with
   the renames; every app's renderer sits in `apps/<app>/render/<app>_render.py`, as smile's does (deep_flow's
   `render.py` and white_space's `render/render.py` move, the classes keep their names); and the apps' shader
   imports follow their shaders (deep_flow, hd_trio, smile and white_space:
   `DrawRectangleOutline` and `DrawColoredRectangle` to `modules/gl`, `PosePointLines` to the pose package;
   `scripts/depth_align_check.py`: `DepthAlign` to `modules/gl`), as do the scripts that import render layers
   (`bridge_compare`, `bridge_shader_check`, the fluid thread's `view_check`, `expansion_check`,
   `velocity_peak_check`, the monitor harness and the rest, to `render.X`). The render's tests move to their packages'
   `tests/`: `test_strip` and `test_marks` to `modules/tracker`, `test_flow_systems`, `test_stage_layer`,
   `test_frame_interval`, `test_monitor_mouse` and `test_layer_names` to `modules/flow`, `test_angle_offset` to
   `modules/pose`; `test_best_grid`, `test_fit` and `test_window` stay at the root with `subdivision`. With the move,
   in the same commit, also `modules/tracker/panoramic/projection.py`'s docstring that names the strip's path, and: `board.md`'s `paths` (the board's render package), `flow.md`'s
   `paths` and `tests/test_glossary.py`'s `FLOW_ROOTS` (`modules/render/modules/flow`), and the documents that name
   the old paths: white_space's `LAYERS.md`, the flow's `FLOW.md`, `FLOW_PLAN.md`, `FLUID_3D.md`, `BRIDGE_3D.md`, `TIMING.md`, `MOTION.md` and
   `VIEW.md` (the fluid thread's, told first). A guard test: no `modules/` package imports `modules.render`. The
   package's document, `modules/render/RENDER.md`, is written in the same commit from the rules thread's proposed
   text (what it holds, the three categories, packages, kinds of layer, names), with the packages `general/` holds
   as the code shows them. Checked by: as above.
3. **The two-app layers' choices.** `LayerCompositor`, `SimilaritySilhouetteLayer` and `ForegroundLookLayer`: a
   constant one holds becomes a setting, at today's value. Checked by: as above, and your look at deep_flow and
   hd_trio.

## Open

- smile keeps a copy of `Blend` (`apps/smile/render/shaders/blend.py`, the same `mix`); once render's is in
  `modules/gl`, smile can use that one. For smile's turn, not this plan.

- hd_trio's center onto the flow's `ViewLayer`: one system for a view placed at the player; a change of behavior,
  after this plan.
- `PoseCompositor`'s and `TrackerCompositor`'s stages and widths per stage are set in their constructors (stage 1, 2,
  3; line width × 2): the module's choice, or settings the apps set?

## Rules it touches

- `rendering.md`, What the render holds: 1, 2.
- A package is one thing, and its name is that thing: 1, 2.
- Name a concept after what it is in the domain; similar things share the noun: 1, 2, 3.
- `rendering.md`'s `Layer`, `Renderer`, `Compositor`: 2, 3.
- When renaming or removing a section, state or setting, search the documents for references to it: 2.
