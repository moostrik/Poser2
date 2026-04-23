---
description: "Use when editing OpenGL, render layers, shaders, or flow rendering modules."
applyTo: "modules/gl/**, modules/render/**, modules/flow/**, apps/*/render/**"
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
- Shader `.frag`/`.vert` files are resolved by `ClassName.lower()` with no separator — `MyShader` → `myshader.frag`. Name the class and file as one unbroken lowercase word (e.g. `LightSimulation` + `lightsimulation.frag`), or use a style where lowercasing matches the filename (e.g. `WS_Lines` → `ws_lines.frag`).

## Layered rendering

- Preserve layer boundaries and data ownership
- Keep layer interfaces narrow and explicit
- Prefer composition of layers over monolithic render passes

## Performance focus

- Minimize CPU-GPU synchronization points
- Batch work where possible
- Avoid unnecessary memory transfers between CPU and GPU
