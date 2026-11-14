# The rules' plan

What is still to do on the rules in `CLAUDE.md` and `.claude/rules/`. A built step leaves this plan.

## 1. What belongs in the render

Decided with Matthias. `modules/render` holds three categories, each a folder:

- `modules`: a layer that draws a module's data, one package per module
- `general`: a layer that draws no module's data and is useful to any app
- `shared`: a layer that is not general and that more than one app uses, one package per shared thing

A layer that is not general and that one app uses stays in `apps/<app>/render/layers/`.

The folders are ordered by subject first and by kind inside. A package holds its layers and, in a `shaders/` folder, the shaders only they use; the shader operations many layers use sit in `general/shaders/`. The separate `layers/` and `shaders/` trees go.

```
modules/render/
  modules/
    pose/        its layers
      shaders/   the shaders only pose layers use
    flow/ ...
  general/       the compositor, the base layer, subdivision, color
    shaders/     the operations many layers use (mask, draw, blend)
  shared/
    <thing>/     its layers
      shaders/
```

Open:

- `modules/render/modules/` reads badly; keep it, or another name for the first category's folder.

When the plan is clear and on Matthias's "go", in one commit: `rendering.md`'s section "What the render holds" is rewritten to these categories and this order, and the class line becomes "A layer's class is named after what it draws, then its kind", so it no longer reads on shader classes. The apps thread's render plan (`modules/render/RENDER_PLAN.md`) orders the code by these lines; it is told when they are in.

## 2. Split rendering.md

After step 1. `rendering.md` mixes three subjects under one set of paths:

- OpenGL itself (the render thread, resources, the `glWaitSync` line): its own file, loaded for any code that touches GL
- the render package (what it holds, its order and names, `Layer`, `Renderer`, `Compositor`): its own file
- advice that cannot be checked ("Keep layer interfaces narrow and explicit", "Batch work where possible"): each line looked at for what it is good for

## Rules it touches

- `rendering.md`: 1, 2
- `CLAUDE.md` "Put a rule where it loads when the code it governs is written": 2
