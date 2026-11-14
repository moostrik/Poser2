# The rules' plan

What is still to do on the rules in `CLAUDE.md` and `.claude/rules/`. This plan changes rule files only; what it found for the code is handed on in steps 3 and 4. A built step leaves this plan. Each step is one commit, built on Matthias's "go".

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
- A shader is named after the operation it does; the `shaders` package in front says what it is (`shaders.PosePointLines`)

Notes on these lines:

- The layer class line repeats CLAUDE.md's Naming and stays as emphasis.
- A shader carries no `Shader` ending; the three in the render that have it today (`FeatureShader`, `WindowShader`, `TimeSeriesShader`) lose it when the code is moved.
- `modules/render` is the package, and its `__init__.py` is what apps import from; the categories and their packages order it inside.
- `shared` is decided by how many apps use a layer, so a layer moves from an app to `shared` when a second app takes it. Accepted.
- The render's shaders without a layer move into `modules/gl/shaders/` as they are, with the render's code move; the folder's order comes later, in the GL's plan.

## 2. Split rendering.md

`rendering.md` mixes OpenGL itself, the render package, and advice that cannot be checked. It stays the render package's file; a new `gl.md` takes OpenGL itself.

### gl.md

Description "OpenGL: the render thread, resources, state, shaders". Paths: `modules/gl/**`, `modules/render/**`, `modules/flow/**`, `apps/*/render/**`, `apps/*/render.py`, and what else a search for OpenGL imports on `deep-flow` finds when building (smile's `player/`).

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

Description "The render package: what it holds, its layers and their names". Paths: `modules/render/**`, `apps/*/render/**`, `apps/*/render.py`.

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

To check when building: the read-back line against the read-backs in the code (screen recorder, velocity peak, finite guard, smile's screen grab), so the line describes what the code should do.

## 3. Park the GL's findings

A new `modules/gl/GL_PLAN.md` holds what this work found for the GL. Nobody builds it until Matthias takes it up. Its content:

- The order of `modules/gl/shaders`, fuller once the render's shaders without a layer are in: what `convenience/` is for, and whether the shaders are grouped by kind of work (mask, color, draw, blit)
- The names that carry the ending (`box_shader`, `text_shader`), against "a shader carries no `Shader` ending"
- How the GL hands out its shaders: by bare name today (`from modules.gl import Blit`); whether that still reads well with many more

It ends with the rules it touches: `gl.md`, CLAUDE.md's Naming and Imports.

## 4. Send the render's text to the apps thread

After steps 1 and 2 are in. The apps thread's render plan (`modules/render/RENDER_PLAN.md`) is redone from the new rule lines. It gets:

- The rule lines of step 1 as they stand in `rendering.md`, and the tree
- The notes of step 1
- What changes with the code move, in the same commit as the move: `board.md` `paths` (`modules/render/layers/**` becomes the board's render package); `flow.md` `paths` and `tests/test_glossary.py` `FLOW_ROOTS` (`modules/render/layers/flow` becomes `modules/render/modules/flow`)
- The shaders without a layer move into `modules/gl/shaders/` as they are, in the same move; which shaders have no layer is settled from the code, and the folder's order is not its work
- Until the move, the commit check reports the old `layers/` and `shaders/` trees against the new lines; that is expected

## 5. This plan is deleted

On Matthias's word, when 1 to 4 are done.

## Rules it touches

- `rendering.md`: 1, 2
- `gl.md` (new): 2
- `CLAUDE.md` "Put a rule where it loads when the code it governs is written": 2
- `CLAUDE.md` "One rule per bullet" and "A rule describes the codebase as it should look": 1, 2
- `CLAUDE.md` "A plan is named after the package, app or folder it is for, and sits in it": 3
- `CLAUDE.md` "A document describes what is built; what is not built is in the plan": 3, 4
