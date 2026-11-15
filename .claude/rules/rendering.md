---
description: The render package: what it holds, its layers and their names
paths:
  - "modules/render/**"
  - "apps/*/render/**"
---
# Rendering

## What the render holds

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

## Layers

- `Layer`: owns its FBO or texture
- `Renderer`: draws into a target it is given and owns no texture
- `Compositor`: owns an FBO and draws other layers into it
- A layer hands out its Fbo or SwapFbo, not the texture inside it
- Prefer composition of layers over monolithic render passes
- A swap buffer's other buffer, the previous one: `back` (`SwapFbo.back`, `SwapTexture3D.back`, `SwapTexture2DArray.back`); not `back_texture`. A texture object is passed as itself, never as a `.texture` of it; a layer's `texture` is the layer's output
