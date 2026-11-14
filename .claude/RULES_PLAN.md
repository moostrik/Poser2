# The rules' plan

What is still to do on the rules in `CLAUDE.md` and `.claude/rules/`. A built step leaves this plan.

## 1. What belongs in the render

Decided with Matthias. `modules/render` holds three categories, each a folder:

- `modules`: the layers that draw a module's data
- `general`: the layers that draw no module's data and are useful to any app
- `shared`: the layers that are not general and that more than one app uses

A layer that is not general and that one app uses stays in `apps/<app>/render/layers/`.

All three have the same shape: a category holds packages, and a package is named after its subject. A package holds its layers and, in a `shaders/` folder, the shaders only they use. The separate `layers/` and `shaders/` trees go.

```
modules/render/
  modules/   pose/  camera/  tracker/  inference/  flow/  board/
  general/   composite/  mask/  color/  draw/
  shared/    <thing>/
```

The general packages are a first reading of the file names (`composite`: the compositor, the base layer, subdivision; `mask`: blur, dilate, erode, apply; `color`: tint, hue shift, blends; `draw`: circles, rectangles, the ROI); the apps thread's render plan settles them from the code.

On Matthias's "go", in one commit, `rendering.md`'s section "What the render holds" becomes:

- `modules/render` holds three categories, each a folder: `modules`, `general` and `shared`
- `modules` holds the layers that draw a module's data
- `general` holds the layers that draw no module's data and are useful to any app
- `shared` holds the layers that are not general and that more than one app uses
- A layer that is not general and that one app uses sits in `apps/<app>/render/layers/`
- A category holds packages, and a package is named after its subject: a module (`modules/pose/`), a kind of work (`general/mask/`), a shared thing
- A package holds its layers, and in `shaders/` the shaders only they use
- A layer's class is named after what it draws, then its kind (`TrackletRenderer`, `PoseCompositor`)
- A shader is named after the operation it does

The apps thread's render plan (`modules/render/RENDER_PLAN.md`) orders the code by these lines; it is told when they are in.

## 2. Split rendering.md

After step 1. `rendering.md` mixes three subjects under one set of paths:

- OpenGL itself (the render thread, resources, the `glWaitSync` line): its own file, loaded for any code that touches GL
- the render package (what it holds, its order and names, `Layer`, `Renderer`, `Compositor`): its own file
- advice that cannot be checked ("Keep layer interfaces narrow and explicit", "Batch work where possible"): each line looked at for what it is good for

## Rules it touches

- `rendering.md`: 1, 2
- `CLAUDE.md` "Put a rule where it loads when the code it governs is written": 2
