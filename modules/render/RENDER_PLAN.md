# The render's plan

The render package held up to the rules (2026-11-14, the apps' plan F). Its tree is built as `rendering.md` has it
(the three categories, packages by subject, shaders with their layers or in `modules/gl/shaders/`); the package's
document is [RENDER.md](RENDER.md). One step per commit, checked by the render, flow and app tests, the guard tests
and every app's start (-sim, smile on its image input, 10 s).

## Steps

1. **The two-app layers' choices.** `LayerCompositor`, `SimilaritySilhouetteLayer` and `ForegroundLookLayer`: a
   constant one holds becomes a setting, at today's value. Checked by: as above, and your look at deep_flow and
   hd_trio.

## Open

- smile keeps a copy of `Blend` (`apps/smile/render/shaders/blend.py`, the same `mix`); now that render's is in
  `modules/gl`, smile can use that one. For smile's turn, not this plan.
- hd_trio's center onto the flow's `ViewLayer`: one system for a view placed at the player; a change of behavior,
  after this plan.
- `PoseCompositor`'s and `TrackerCompositor`'s stages and widths per stage are set in their constructors (stage 1, 2,
  3; line width × 2): the module's choice, or settings the apps set?

## Rules it touches

- Name a concept after what it is in the domain; similar things share the noun: 1.
- `rendering.md`'s `Layer`, `Renderer`, `Compositor`: 1.
