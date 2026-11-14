# The rules' plan

What is still to do on the rules in `CLAUDE.md` and `.claude/rules/`. A built step leaves this plan.

## 1. What belongs in the render

Matthias's three main categories of what `modules/render` holds:

- `modules`: a layer that draws a module's data
- `general`: a layer that draws no module's data and is useful to any app
- `shared`: a layer that is not general and that more than one app uses

A layer that is not general and that one app uses stays in `apps/<app>/render/layers/`.

Open, to decide before any rule changes:

- Are the three categories folders (`layers/modules/<module>/`, `layers/general/`, `layers/shared/`) or kinds only?
- Do the shaders follow the same three, or stay ordered by the kind of operation?

When decided, in one commit: the first two lines of `rendering.md`'s "What the render holds" are replaced by the lines for the three categories, and the class line becomes "A layer's class is named after what it draws, then its kind", so it no longer reads on shader classes. The apps thread's render plan (`modules/render/RENDER_PLAN.md`) orders the code by these lines; it is told when they are in.

## 2. Split rendering.md

After step 1. `rendering.md` mixes three subjects under one set of paths:

- OpenGL itself (the render thread, resources, the `glWaitSync` line): its own file, loaded for any code that touches GL
- the render package (what it holds, its order and names, `Layer`, `Renderer`, `Compositor`): its own file
- advice that cannot be checked ("Keep layer interfaces narrow and explicit", "Batch work where possible"): each line looked at for what it is good for

## Rules it touches

- `rendering.md`: 1, 2
- `CLAUDE.md` "Put a rule where it loads when the code it governs is written": 2
