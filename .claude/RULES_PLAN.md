# The rules' plan

What is still to do on the rules in `CLAUDE.md` and `.claude/rules/`. A built step leaves this plan.

## 1. What belongs in the render

Being decided with Matthias. A layer is placed by whether it is general, not by who uses it or what it draws:

- general: `modules/render` (a layer that draws a module's data in `layers/<module>/`; a general layer that draws no module's data in a package named after what it is)
- not general, one app: `apps/<app>/render/layers/`
- not general, more than one app: proposed `apps/shared/render/layers/`, not agreed

When decided: the first line of `rendering.md`'s "What the render holds" is replaced by it, and the class line becomes "A layer's class is named after what it draws, then its kind", so it no longer reads on shader classes.

## 2. Split rendering.md

After step 1. `rendering.md` mixes three subjects under one set of paths:

- OpenGL itself (the render thread, resources, the `glWaitSync` line): its own file, loaded for any code that touches GL
- the render package (what it holds, its order and names, `Layer`, `Renderer`, `Compositor`): its own file
- advice that cannot be checked ("Keep layer interfaces narrow and explicit", "Batch work where possible"): each line looked at for what it is good for

## Rules it touches

- `rendering.md`: 1, 2
- `CLAUDE.md` "Put a rule where it loads when the code it governs is written": 2
