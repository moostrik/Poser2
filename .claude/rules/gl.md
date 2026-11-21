---
description: OpenGL: the main thread, resources, state, shaders
paths:
  - "modules/gl/**"
  - "modules/window/**"
  - "modules/render/**"
  - "modules/flow/**"
  - "apps/*/render/**"
  - "apps/smile/player/**"
---
# OpenGL

## What the GL does

- Judge a GL design by the latency it adds to the 2D path, camera to screen
- A 3D step stays inside its own pass
- Drawing is done in shaders
- Simple geometry, as from poses, is worked out on the CPU, in plain modules
- Volumetrics and any other 3D are done inside a shader
- Prefer as few GL calls and state switches as possible
- A texture is held in GL's row order
- A CPU image is flipped once, at its upload
- All shaders take 2D positions from 0 to 1, with (0, 0) at the top left
- The conversion to OpenGL's own coordinates is written once, and every shader uses it

## Working with OpenGL

- All OpenGL calls execute on the main thread, in the window loop
- Create textures, FBOs and shaders at setup or on a resize, not per frame
- Pair every creation with its destruction
- Set the GL state a draw needs where it draws (blend, depth, viewport)
- Change the blend state only through `BlendState`, which keeps it without `glGet*`
- Prefer not to read back from the GPU or wait on it inside a frame
- Name a shader's `.frag`/`.vert`/`.comp` source after the `.py` file that holds its class, beside it; the loader resolves it by that module name
- A shader is named after the operation it does; the `shaders` package in front says what it is (`shaders.PosePointLines`)
- Import OpenGL names by name, never with `from OpenGL.GL import *`
- `modules/gl/shaders` holds the shaders that are useful to any app
- An app's own shader sits in that app
- Shaders get checked in a GPU harness
