# Coordinate System Convention

## Overview

This application uses a **top-left origin, pixel-based coordinate system** throughout.

The coordinate system is implemented via a **GPU flip on upload** approach where all textures are normalized to have the same V orientation during upload:

| Layer | Responsibility | Implementation |
|-------|----------------|----------------|
| **Image Upload** | Normalize CPU→GPU textures | Upload to staging, blit with V-flip to FBO |
| **Tensor Upload** | Normalize CUDA→GL textures | `torch.flip(tensor, dims=[0])` before PBO copy |
| **Vertex Shader** | Pixels → NDC with Y-flip | `ndc.y = -ndc.y` for top-left screen origin |
| **Fragment Shader** | Match `gl_FragCoord` to system | `layout(origin_upper_left)` when used |

**Key insight**: By flipping textures once during upload, ALL textures (Image, Tensor, FBO) have uniform V orientation. No per-draw flip decisions needed.

## Screen/Window Space

```
(0, 0) ──────────────► X
  │
  │
  │
  ▼
  Y                (width, height)
```

- **Origin (0, 0)**: Top-left corner of screen/window/FBO
- **X-axis**: Increases rightward (standard)
- **Y-axis**: Increases **downward** (screen/UI convention)

This matches:
- Window coordinate systems (GLFW, Win32, etc.)
- NumPy/image array indexing `image[y, x]`
- Most UI frameworks
- Fragment shader `gl_FragCoord.xy` with `origin_upper_left`

## Implementation

### Layer 1: GPU Flip on Upload (Image.py, Tensor.py)

**Problem**: CPU-uploaded textures and FBO-rendered content have different V orientations:
- CPU upload: NumPy row 0 → OpenGL V=0 (bottom of GL texture)
- FBO render: Top of content → V=1 (content rendered top-down)

**Solution**: Flip CPU uploads during GPU upload so all textures have top at V=1.

**Image class** (NumPy → OpenGL):
```python
def set_from_image(self, image: np.ndarray) -> None:
    # Step 1: Upload to staging texture (no flip)
    self._staging.bind()
    glTexImage2D(..., image)

    # Step 2: Blit with V-flip to self (FBO)
    FboBlit().use(self, self._staging, flip_v=True)
```

**Tensor class** (CUDA → OpenGL):
```python
def _update_with_pbo(self, tensor: torch.Tensor) -> None:
    # Flip on GPU before PBO copy
    tensor = torch.flip(tensor, dims=[0])
    # ... CUDA-GL interop copy ...
```

### Layer 2: Vertex Shader (_generic.vert)

All rendering uses **pixel coordinates** (0 to width/height), not NDC coordinates (-1 to 1).

```glsl
void main() {
    vec2 position = gl_Vertex.xy;  // Pixel coords (0 to width/height)

    // Texture coordinates - all textures have uniform orientation
    // flipV is only used during upload blit step
    texCoord.x = gl_MultiTexCoord0.x;
    texCoord.y = flipV ? (1.0 - gl_MultiTexCoord0.y) : gl_MultiTexCoord0.y;

    // Convert pixels to NDC with Y-flip for top-left screen origin
    vec2 ndc = (position / resolution) * 2.0 - 1.0;
    ndc.y = -ndc.y;  // Flip Y: pixel (0,0) -> NDC (-1, 1) = top-left

    gl_Position = vec4(ndc, 0.0, 1.0);
}
```

**Key points**:
- NDC Y-flip makes pixel (0,0) render at top-left of screen
- `flipV` uniform is only used during upload blit (FboBlit with `flip_v=True`)
- Normal drawing uses `flipV=false` since all textures have uniform orientation

### Layer 3: Fragment Shaders

Fragment shaders use our top-left coordinate system automatically via `texCoord`.

**Texture Sampling** (most common):
```glsl
uniform sampler2D tex0;
in vec2 texCoord;

void main() {
    vec4 color = texture(tex0, texCoord);
    // texCoord (0, 0) = top-left of texture
    // texCoord (1, 1) = bottom-right of texture
}
```

**Using gl_FragCoord** (when needed):
```glsl
layout(origin_upper_left) in vec4 gl_FragCoord;

void main() {
    vec2 pixelPos = gl_FragCoord.xy;
    // pixelPos (0, 0) = top-left of framebuffer
    // pixelPos (width, height) = bottom-right of framebuffer
}
```

**Important**: Always add `layout(origin_upper_left)` to fragment shaders that use `gl_FragCoord.xy`. Without this qualifier, `gl_FragCoord` uses OpenGL's default bottom-left origin.

**Calculating pixel position from texCoord**:
```glsl
uniform vec2 resolution;
in vec2 texCoord;

void main() {
    vec2 pixelPos = texCoord * resolution;
    // pixelPos is in [0, width] x [0, height] with (0,0) at top-left
}
```

### Shader Usage Pattern

Every shader that renders to an FBO follows this pattern:

```python
def use(self, fbo: Fbo, texture: Texture, ...) -> None:
    glBindFramebuffer(GL_FRAMEBUFFER, fbo.fbo_id)
    glViewport(0, 0, fbo.width, fbo.height)

    glUseProgram(self.shader_program)

    # Pass resolution to vertex shader for pixel→NDC conversion
    glUniform2f(glGetUniformLocation(self.shader_program, "resolution"),
                float(fbo.width), float(fbo.height))

    # Bind textures, set other uniforms...

    # Draw fullscreen quad in pixel space
    draw_quad_pixels(fbo.width, fbo.height)

    glBindFramebuffer(GL_FRAMEBUFFER, 0)
```

### No Projection Matrix

**This system does NOT use projection matrices** (`glOrtho`, `glMatrixMode`, etc.). The vertex shader handles all coordinate conversion via NDC Y-flip.

- No `setView()` or `setOrthoView()` calls needed
- No legacy fixed-function pipeline calls

### Why GPU Flip on Upload?

OpenGL stores texture data bottom-up by convention:
- Row 0 of `glTexImage2D` data → bottom of texture (V=0)
- NumPy/images have row 0 at top

FBO-rendered content is different:
- Content rendered top-down ends up with top at V=1

**Solution**: Flip CPU-uploaded textures during upload so ALL textures have top at V=1. This is:
- **GPU-only**: No CPU overhead (Image uses blit, Tensor uses `torch.flip`)
- **One-time cost**: Flip happens once during upload, not every draw
- **Uniform**: All texture types (Image, Tensor, FBO) behave identically
- **Simple**: No per-draw `flip_v` decisions needed

## Advantages

1. **Intuitive**: Work in pixels, not abstract NDC space
2. **Consistent**: All textures have uniform V orientation
3. **Simple**: No per-draw flip flags needed
4. **GPU-only**: Flip happens on GPU during upload
5. **Modern**: Uses modern OpenGL with shader-based rendering

## Compatibility

- **NumPy/Image Arrays**: Row 0 (top) → upload → V=1 (top) after GPU flip
- **CUDA Tensors**: Row 0 (top) → `torch.flip` → V=1 (top) in OpenGL
- **FBOs**: Top of rendered content at V=1 (native)
- **UI Coordinates**: Direct 1:1 mapping with screen/window coordinates
- **gl_FragCoord**: Use `layout(origin_upper_left)` qualifier
