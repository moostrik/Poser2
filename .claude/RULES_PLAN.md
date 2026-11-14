# The rules' plan

What is still to do on the rules in `CLAUDE.md` and `.claude/rules/`. A built step leaves this plan. Each step is one commit, built on Matthias's "go".

## 1. What the render holds

Decided with Matthias. `modules/render` holds layers, in three categories. A category is a folder that holds packages; a package is named after its subject and holds its layers and, in `shaders/`, the shaders of those layers. A shader that no layer owns is a GL tool and sits in `modules/gl/shaders/`, which already holds such operations (blit, contrast, hsv, sharpen, blur). The separate `layers/` and `shaders/` trees go.

```
modules/render/
  modules/   pose/  camera/  tracker/  inference/  flow/  board/
  general/   composite/  ...
  shared/    <thing>/

modules/gl/
  shaders/   blit, contrast, hsv, ... and the render's shaders without a layer (mask, color, draw)
```

The packages under `general` are a first reading of the file names; the apps thread's render plan (`modules/render/RENDER_PLAN.md`) settles them from the code.

`rendering.md`'s section "What the render holds" becomes:

- `modules/render` holds three categories, each a folder: `modules`, `general` and `shared`
- `modules` holds the layers that draw a module's data
- `general` holds the layers that draw no module's data and are useful to any app
- `shared` holds the layers that are not general and that more than one app uses
- A layer that is not general and that one app uses sits in `apps/<app>/render/layers/`
- A category holds packages
- A package is named after its subject: a module (`modules/pose/`), a kind of work (`general/composite/`), a shared thing
- A package holds its layers, and the shaders of those layers in `shaders/`
- A shader that no layer owns sits in `modules/gl/shaders/`
- A layer's class is named after what it draws, then its kind (`TrackletRenderer`, `PoseCompositor`)
- A shader is named after the operation it does; the `shaders` package in front says what it is (`shaders.MaskBlur`)

Notes on these lines:

- The layer class line repeats CLAUDE.md's Naming and stays as emphasis.
- A shader carries no `Shader` ending; the three in the render that have it today (`FeatureShader`, `WindowShader`, `TimeSeriesShader`) lose it in the render plan.
- A category is a folder, not a package: CLAUDE.md's package lines hold for each package under it.
- `shared` is decided by how many apps use a layer, so a layer moves from an app to `shared` when a second app takes it. Accepted.
- A file sees at most two `shaders`: its own package's and the GL's. To check when building: how `modules/gl` hands out its shaders now, so the example in the last line is right for both.

What follows outside the rule files, for the render plan's step that moves the code (same commit as the move, so no path goes stale):

- `board.md` `paths`: `modules/render/layers/**` becomes the board's render package
- `flow.md` `paths` and `tests/test_glossary.py` `FLOW_ROOTS`: `modules/render/layers/flow` becomes `modules/render/modules/flow`
- The render plan's own five rules and naming scheme are replaced by the lines above; that plan is redone from them. The apps thread is told the final lines when step 1 is in.

## 2. Split rendering.md

`rendering.md` mixes OpenGL itself, the render package, and advice that cannot be checked. It stays the render package's file; a new `gl.md` takes OpenGL itself.

### gl.md

Description "OpenGL: the render thread, resources, state, shaders". Paths as `rendering.md` has now: `modules/gl/**`, `modules/render/**`, `modules/flow/**`, `apps/*/render/**`.

- All OpenGL calls execute on the render thread
- Create textures, FBOs and shaders at setup or on a resize, not per frame
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

- Section "What the render holds": the lines of step 1
- Section "Layers": the `Layer`, `Renderer` and `Compositor` lines and the swap buffer's `back` line, unchanged; "A layer hands out its Fbo or SwapFbo, not the texture inside it"; "Prefer composition of layers over monolithic render passes"

### Lines that leave, and what says it instead

| Line now                                                                      | Said by                                                        |
|-------------------------------------------------------------------------------|----------------------------------------------------------------|
| Do not perform GL resource creation, binding, or draw calls from worker threads | "All OpenGL calls execute on the render thread"                |
| Keep texture/FBO/shader lifecycle explicit                                    | "Pair every creation with its destruction"                     |
| Ensure creation and destruction paths are paired                              | "Pair every creation with its destruction"                     |
| Avoid transient resource churn in per-frame hot paths                         | "Create textures, FBOs and shaders at setup or on a resize, not per frame" |
| Validate render-state assumptions locally (blend, depth, viewport)            | "Set the GL state a draw needs where it draws"                 |
| Minimize CPU-GPU synchronization points                                       | "Prefer not to read back from the GPU or wait on it inside a frame" |
| Avoid unnecessary memory transfers between CPU and GPU                        | CLAUDE.md "avoid unnecessary copies", and the read-back line   |
| Keep shader contracts stable and documented by uniform/buffer naming          | CLAUDE.md's Naming                                             |
| Preserve layer boundaries and data ownership                                  | the `Layer`, `Renderer`, `Compositor` lines (who owns what)    |
| Keep layer interfaces narrow and explicit                                     | "A layer hands out its Fbo or SwapFbo, not the texture inside it" |
| Batch work where possible                                                     | nothing; no case behind it is known                            |

To check when building:

- The read-back line against the read-backs in the code (screen recorder, velocity peak, finite guard, smile's screen grab), so the line describes what the code should do
- That `gl.md`'s paths cover all code that calls OpenGL: search for OpenGL imports outside them

## 3. The order of `modules/gl/shaders`

Open, not yet discussed. With the render's layerless shaders coming in, the folder needs an order of its own. To decide:

- Its sub-folders: what `convenience/` is for, and whether the shaders are grouped by kind of work (mask, color, draw, blit)
- The names that carry the ending (`box_shader`, `text_shader`), against "a shader carries no `Shader` ending"

## Rules it touches

- `rendering.md`: 1, 2
- `gl.md` (new): 2, 3
- `board.md` and `flow.md` (`paths` only): 1, with the code move
- `CLAUDE.md` "Put a rule where it loads when the code it governs is written": 2
- `CLAUDE.md` "One rule per bullet" and "A rule describes the codebase as it should look": 1, 2
