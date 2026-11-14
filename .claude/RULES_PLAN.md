# The rules' plan

What is still to do on the rules in `CLAUDE.md` and `.claude/rules/`. A built step leaves this plan. Each step is one commit, built on Matthias's "go".

## 1. What the render holds

Decided with Matthias. `modules/render` holds three categories, each a folder, and all three have the same shape: a category holds packages, a package is named after its subject, and a package holds its layers and, in `shaders/`, its shaders. The separate `layers/` and `shaders/` trees go.

```
modules/render/
  modules/   pose/  camera/  tracker/  inference/  flow/  board/
  general/   composite/  mask/  color/  draw/
  shared/    <thing>/
```

The general packages are a first reading of the file names; the apps thread's render plan (`modules/render/RENDER_PLAN.md`) settles them from the code and orders the code by the lines below. It is told when they are in.

`rendering.md`'s section "What the render holds" becomes:

- `modules/render` holds three categories, each a folder: `modules`, `general` and `shared`
- `modules` holds the layers that draw a module's data
- `general` holds the layers that draw no module's data and are useful to any app
- `shared` holds the layers that are not general and that more than one app uses
- A layer that is not general and that one app uses sits in `apps/<app>/render/layers/`
- A category holds packages, and a package is named after its subject: a module (`modules/pose/`), a kind of work (`general/mask/`), a shared thing
- A package holds its layers, and its shaders in `shaders/`
- A layer's class is named after what it draws, then its kind (`TrackletRenderer`, `PoseCompositor`)
- A shader is named after the operation it does; the `shaders` package in front says what it is (`shaders.MaskBlur`)

The layer class line repeats CLAUDE.md's Naming and stays as emphasis. A shader carries no `Shader` ending; the three that have it today (`FeatureShader`, `WindowShader`, `TimeSeriesShader`) lose it in the render plan.

## 2. Split rendering.md

`rendering.md` mixes OpenGL itself, the render package, and advice that cannot be checked. It stays the render package's file; a new `gl.md` takes OpenGL itself.

### gl.md

Description "OpenGL: the render thread, resources, state, shaders". Paths as `rendering.md` has now: `modules/gl/**`, `modules/render/**`, `modules/flow/**`, `apps/*/render/**`.

- All OpenGL calls execute on the render thread
- Create a texture, FBO or shader once, not per frame
- Pair every creation with its destruction
- Set the GL state a draw needs where it draws (blend, depth, viewport)
- Prefer not to read back from the GPU or wait on it inside a frame
- Name a shader's `.frag`/`.vert`/`.comp` source after the `.py` file that holds its class, beside it; the loader resolves it by that module name
- Import OpenGL names by name, never with `from OpenGL.GL import *`
- Treat hot-reload as a development aid, not runtime control flow
- The `glWaitSync` line, unchanged
- The line on text drawn into a cached FBO, unchanged

### rendering.md

Description "The render package: what it holds, its layers and their names". Paths: `modules/render/**`, `apps/*/render/**`.

- Section "What the render holds": the nine lines of step 1
- Section "Layers": the `Layer`, `Renderer` and `Compositor` lines and the swap buffer's `back` line, unchanged; "A layer hands out its Fbo or SwapFbo, not the texture inside it"; "Prefer composition of layers over monolithic render passes"

### Lines that leave, and what says it instead

| Line now                                                                      | Said by                                                        |
|-------------------------------------------------------------------------------|----------------------------------------------------------------|
| Do not perform GL resource creation, binding, or draw calls from worker threads | "All OpenGL calls execute on the render thread"                |
| Keep texture/FBO/shader lifecycle explicit                                    | "Pair every creation with its destruction"                     |
| Avoid transient resource churn in per-frame hot paths                         | "Create a texture, FBO or shader once, not per frame"          |
| Validate render-state assumptions locally (blend, depth, viewport)            | "Set the GL state a draw needs where it draws"                 |
| Minimize CPU-GPU synchronization points                                       | "Prefer not to read back from the GPU or wait on it inside a frame" |
| Avoid unnecessary memory transfers between CPU and GPU                        | CLAUDE.md "avoid unnecessary copies", and the read-back line   |
| Keep shader contracts stable and documented by uniform/buffer naming          | CLAUDE.md's Naming                                             |
| Preserve layer boundaries and data ownership                                  | the `Layer`, `Renderer`, `Compositor` lines (who owns what)    |
| Keep layer interfaces narrow and explicit                                     | "A layer hands out its Fbo or SwapFbo, not the texture inside it" |
| Batch work where possible                                                     | nothing; no case behind it is known                            |

To check when building: the read-back line against the screen recorder and any other read-back in the code, so the line describes what the code should do.

## Rules it touches

- `rendering.md`: 1, 2
- `gl.md` (new): 2
- `CLAUDE.md` "Put a rule where it loads when the code it governs is written": 2
- `CLAUDE.md` "One rule per bullet" and "A rule describes the codebase as it should look": 2
