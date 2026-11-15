---
description: OpenGL: the render thread, resources, state, shaders
paths:
  - "modules/gl/**"
  - "modules/render/**"
  - "modules/flow/**"
  - "apps/*/render/**"
  - "apps/smile/player/**"
---
# OpenGL

- All OpenGL calls execute on the render thread
- Create textures, FBOs and shaders at setup or on a resize, not per frame
- Pair every creation with its destruction
- Set the GL state a draw needs where it draws (blend, depth, viewport)
- Prefer not to read back from the GPU or wait on it inside a frame
- Name a shader's `.frag`/`.vert`/`.comp` source after the `.py` file that holds its class, beside it; the loader resolves it by that module name
- Import OpenGL names by name, never with `from OpenGL.GL import *`
- Pass `0xFFFFFFFFFFFFFFFF` as `glWaitSync`'s timeout, never PyOpenGL's `GL_TIMEOUT_IGNORED`: in PyOpenGL 3.1.10 it is a signed value the driver rejects (GL_INVALID_VALUE), and with error checking off the error surfaces at a later `glGetError`, in unrelated code
- Draw text that changes less often than every frame into a cached FBO, redrawn only when its content changes; `Text` builds and uploads each string's glyphs on every draw
- `modules/gl/shaders` holds the shaders that are useful to any app
- An app's own shader sits in that app
