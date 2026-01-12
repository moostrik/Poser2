# Coordinate System Convention

## Overview

This application uses a **top-left origin, pixel-based coordinate system** throughout.

All rendering is shader-based. Vertices are specified in **pixel coordinates** and the vertex shader converts them to normalized device coordinates (NDC) with Y-axis inverted.

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
- **Y-axis**: Increases **downward** (screen/UI convention, not OpenGL default)

This matches:
- Window coordinate systems (GLFW, Win32, etc.)
- NumPy/image array indexing `image[y, x]`
- Most UI frameworks
- Fragment shader `gl_FragCoord.xy` when Y is inverted

## Implementation

### Pixel-Based Rendering

All rendering uses **pixel coordinates** (0 to width/height), not NDC coordinates (-1 to 1).

**Vertex Input**: `position` attribute receives pixel coordinates
```python
# Draw a quad covering a 1920x1080 framebuffer
draw_quad_pixels(1920.0, 1080.0)  # Vertices: (0,0) to (1920,1080)
```

**Vertex Shader** (_generic.vert):
```glsl
in vec2 position;              // Pixel coordinates (e.g., 0 to 1920)
uniform vec2 resolution;       // FBO/window dimensions
out vec2 texCoord;

void main() {
    // Normalize to [0, 1]
    vec2 normalized = position / resolution;

    // Texture coordinates
    texCoord = normalized;
    if (flipX) texCoord.x = 1.0 - texCoord.x;
    if (flipY) texCoord.y = 1.0 - texCoord.y;

    // Convert to NDC with Y-flip for top-left origin
    vec2 ndc;
    ndc.x = normalized.x * 2.0 - 1.0;    // 0→-1, width→1
    ndc.y = 1.0 - normalized.y * 2.0;    // 0→1, height→-1 (FLIPPED)

    gl_Position = vec4(ndc, 0.0, 1.0);
}
```

**Key transformation**: Pixel (0, 0) → NDC (-1, 1) = top-left

### Shader Usage Pattern

Every shader that renders to an FBO follows this pattern:

```python
def use(self, fbo: Fbo, texture: Texture, ...) -> None:
    glBindFramebuffer(GL_FRAMEBUFFER, fbo.fbo_id)
    glViewport(0, 0, fbo.width, fbo.height)

    glUseProgram(self.shader_program)

    # Pass resolution to vertex shader
    glUniform2f(glGetUniformLocation(self.shader_program, "resolution"),
                float(fbo.width), float(fbo.height))

    # Optional: texture flipping
    glUniform1i(glGetUniformLocation(self.shader_program, "flipX"), 0)
    glUniform1i(glGetUniformLocation(self.shader_program, "flipY"), 0)

    # Other uniforms...

    # Draw fullscreen quad in pixel space
    draw_quad_pixels(fbo.width, fbo.height)

    glBindFramebuffer(GL_FRAMEBUFFER, 0)
```

### No Projection Matrix

**This system does NOT use projection matrices** (`glOrtho`, `glMatrixMode`, etc.). The vertex shader handles all coordinate conversion directly.

- `Fbo.begin()` only binds framebuffer and sets viewport
- `SwapFbo.begin()` only binds framebuffer and sets viewport
- No `setView()` calls needed for shader rendering

### Texture Coordinates

Texture coordinates are automatically normalized by the vertex shader:
- Pixel position → Normalized [0, 1] → TexCoord
- Optional flipping via `flipX`/`flipY` uniforms
- Matches NumPy array indexing (row 0 = V=0 at top)

### Fragment Shaders

Fragment shaders automatically use our top-left coordinate system:

**1. Texture Sampling**
```glsl
uniform sampler2D tex0;
in vec2 texCoord;

void main() {
    vec4 color = texture(tex0, texCoord);
    // texCoord (0, 0) = top-left of texture
    // texCoord (1, 1) = bottom-right of texture
}
```

**2. Pixel Coordinates (gl_FragCoord)**
```glsl
layout(origin_upper_left) in vec4 gl_FragCoord;

void main() {
    vec2 pixelPos = gl_FragCoord.xy;
    // pixelPos (0, 0) = top-left of framebuffer
    // pixelPos (width, height) = bottom-right of framebuffer
}
```

**Important**: Add `layout(origin_upper_left)` to fragment shaders that use `gl_FragCoord.xy` to ensure pixel (0,0) is at top-left. Without this qualifier, `gl_FragCoord` uses OpenGL's default bottom-left origin.

**3. Custom Calculations**
If your fragment shader calculates positions or patterns, work in pixel space:
```glsl
uniform vec2 resolution;
in vec2 texCoord;

void main() {
    // Get pixel position from texCoord
    vec2 pixelPos = texCoord * resolution;
    // pixelPos is now in [0, width] x [0, height] with (0,0) at top-left
}
```
**Rule of thumb**: If you're drawing fullscreen and care about exact pixels, use gl_FragCoord. If you're working with texture regions or normalized coordinates, calculate from texCoord.

## Advantages

1. **Intuitive**: Work in pixels, not abstract NDC space
2. **Consistent**: Fragment shader `gl_FragCoord.xy` matches vertex input
3. **Simple**: No projection matrix setup or management
4. **Flexible**: Per-shader resolution, supports any FBO/window size
5. **Modern**: Shader-based approach, not legacy fixed-function pipeline

## Compatibility

- **NumPy/Image Arrays**: Row 0 (top) maps to Y=0 (top) - direct match
- **OpenGL Textures**: Handled via `flipY` uniform when needed
- **UI Coordinates**: Direct 1:1 mapping with screen/window coordinates
- **ofxFlowTools (C++)**: Matches the pixel-based rendering approach
