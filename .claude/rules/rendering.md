---
description: OpenGL, render layers, shaders, and flow rendering
paths:
  - "modules/gl/**"
  - "modules/render/**"
  - "modules/flow/**"
  - "apps/*/render/**"
---
# GL & Rendering Guidelines

## Threading rule

- All OpenGL calls must execute on the render thread
- Do not perform GL resource creation, binding, or draw calls from worker threads

## Resource lifecycle

- Keep texture/FBO/shader lifecycle explicit
- Ensure creation and destruction paths are paired
- Avoid transient resource churn in per-frame hot paths

## Shader and pipeline behavior

- Keep shader contracts stable and documented by uniform/buffer naming
- Treat hot-reload as a development aid, not runtime control flow
- Validate render-state assumptions locally (blend, depth, viewport)
- Name a shader's `.frag`/`.vert`/`.comp` source after the `.py` file that holds its class, beside it; the loader resolves it by that module name
- Import OpenGL names by name, never with `from OpenGL.GL import *`

## Layered rendering

- Preserve layer boundaries and data ownership
- Keep layer interfaces narrow and explicit
- Prefer composition of layers over monolithic render passes

## Performance focus

- Minimize CPU-GPU synchronization points
- Batch work where possible
- Avoid unnecessary memory transfers between CPU and GPU
