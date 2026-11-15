# The render's plan

The render package held up to the rules (2026-11-14, the apps' plan F). Its tree is built as `rendering.md` has it
(the three categories, packages by subject, shaders with their layers or in `modules/gl/shaders/`); the package's
document is [RENDER.md](RENDER.md). Its steps are built; what is left is open.

## Open

- smile keeps a copy of `Blend` (`apps/smile/render/shaders/blend.py`, the same `mix`); now that render's is in
  `modules/gl`, smile can use that one. For smile's turn, not this plan.
- hd_trio's center onto the flow's `ViewLayer`: one system for a view placed at the player; a change of behavior,
  after this plan.
- `PoseCompositor`'s and `TrackerCompositor`'s stages and widths per stage are set in their constructors (stage 1, 2,
  3; line width × 2): the module's choice, or settings the apps set?
- `SimilaritySilhouetteLayer`'s two easing curves (`easeInOutSine` on the motion, `easeInSine` on the foreground's
  blend) are fixed functions: a choice of curve, not a value, so they are not settings.
- The panoramic strip's `Compositor` does not say what it draws (`rendering.md`: what it draws, then its kind).

## Rules it touches

- Name a concept after what it is in the domain; similar things share the noun: the strip's `Compositor`.
