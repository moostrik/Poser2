---
description: The render package: what it holds, its layers and their names
paths:
  - "modules/render/**"
  - "apps/*/render/**"
---
# Rendering

## What the render holds

- `modules/render` holds three categories, each a folder: `modules`, `general` and `shared`
- `modules` holds the layers that draw a module's data, in a package named after the module
- `general` holds the layers that draw no module's data and are useful to any app
- `shared` holds the layers that are not general and that more than one app uses
- A layer that is not general and that one app uses sits in `apps/<app>/render/layers/`
- A package is named after its subject: a module (`modules/pose/`), a kind of work (`general/screen_recording/`), a shared thing
- A package holds its layers, and the shaders of those layers in `shaders/`
- A layer's class is named after what it draws, then its kind (`TrackletDraw`, `PoseCompositor`)

## Layers

- `Layer`: owns its output, an Fbo or texture, and draws into it
- `Draw`: has no output of its own; it draws into the target it is given
- `Compositor`: owns its output Fbo and draws other layers into it
- A layer hands out its Fbo, SwapFbo or SwapTarget, never one buffer of a pair
- Prefer to redraw a layer only when what it draws has changed
- Prefer composition of layers over monolithic render passes
- A swap buffer's other buffer, the previous one: `back` (`SwapFbo.back`, `SwapTexture3D.back`, `SwapTexture2DArray.back`); not `back_texture`. A texture object is passed as itself, never as a `.texture` of it; a layer's `texture` is the layer's output
