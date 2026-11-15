# The render's plan

The render package held up to the rules (2026-11-14, the apps' plan F). Its tree is built as `rendering.md` has it
(the three categories, packages by subject, shaders with their layers or in `modules/gl/shaders/`); the package's
document is [RENDER.md](RENDER.md). Its steps are built; what is left is open.

## Open

- `SimilaritySilhouetteLayer`'s two easing curves (`easeInOutSine` on the motion, `easeInSine` on the foreground's
  blend) are fixed functions: a choice of curve, not a value, so they are not settings.
- The panoramic strip's `Compositor` does not say what it draws (`rendering.md`: what it draws, then its kind).

## Rules it touches

- Name a concept after what it is in the domain; similar things share the noun: the strip's `Compositor`.
