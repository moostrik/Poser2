# Coordinate System Quick Reference

## Overview
This project uses a **top-left origin** coordinate system with **Y-axis pointing down**, matching common windowing systems and image coordinate conventions.

---

## Screen/Window Coordinates

```
(0,0) ──────────► X
 │
 │
 │
 ▼ Y

Origin: Top-left corner
X-axis: Left → Right (positive)
Y-axis: Top → Bottom (positive)
```

### Key Points
- **Origin**: `(0, 0)` is at the **top-left** corner
- **X increases**: left to right
- **Y increases**: top to bottom
- **Window dimensions**: `(width, height)` is at the bottom-right corner

### Implementation
Set via `View.set_view(width, height)`:
```python
glOrtho(0, width, height, 0, -1, 1)
```
This inverts the Y-axis compared to OpenGL's default bottom-left origin.

---

## Texture Coordinates

### IMPORTANT: Two Different Coordinate Spaces

**Texture Coordinate Space** (Internal - OpenGL Standard):
```
 │ V
 │
 │
(0,0) ──────────► U

OpenGL texture coordinate space:
(0, 0) = bottom-left of texture
(1, 1) = top-right of texture
```

**Rendered Output Space** (Visual - Screen Mapping):
```
(0,0) ──────────► U
 │
 │
 │
 ▼ V

After draw_quad() mapping:
(0, 0) = top-left of rendered output
(1, 1) = bottom-right of rendered output
```

### Key Points
- **Internal texture coordinates**: Follow OpenGL convention (bottom-left origin)
- **Rendered output**: Flipped to match screen coordinates (top-left origin)
- **U/S**: Horizontal coordinate (0.0 - 1.0)
- **V/T**: Vertical coordinate (0.0 - 1.0)

### When to Use Each Space

**Use Texture Space (bottom-left origin) for:**
- Geometric calculations (CentreGeometry anchor points)
- Shader uniform coordinates
- Direct texture coordinate manipulation
- ROI calculations in texture space

**Use Screen Space (top-left origin) for:**
- Drawing to screen/FBO
- Mouse/UI coordinates
- Image pixel coordinates

### Drawing Functions

**View.draw_quad(x, y, w, h)** - Screen & FBO rendering:
```python
from modules.gl.View import draw_quad
draw_quad(x, y, width, height)  # ALWAYS flips Y
```
- Texture `(0, 1)` → Top-left of destination
- Texture `(1, 0)` → Bottom-right of destination
- Use for all rendering to screen or FBO

**Shader.draw_quad()** - Fullscreen shader passes:
```python
from modules.gl.Shader import draw_quad
draw_quad()  # NO flip, NDC space
```
- Texture `(0, 0)` → NDC `(-1, -1)`
- Use only for fullscreen shader effects

### Converting Between Spaces
```python
# Image space (top-left) → Texture space (bottom-left)
tex_y = 1.0 - img_y
tex_rotation = -img_rotation

# Example from CentreGeometry
anchor_tex = Point2f(anchor_img.x, 1.0 - anchor_img.y)
```

---

## FBO (Framebuffer Object) Rendering

### Coordinate System
FBOs use the **same top-left origin** as the main window:
```python
fbo.begin()  # Sets up top-left origin viewport
# Draw here - (0,0) is top-left
fbo.end()    # Restores previous state
```

### Important Behavior
- `Fbo.begin()` calls `set_view(width, height)` internally (top-left origin)
- `View.draw_quad()` flips when rendering INTO FBO → stores bottom-left texture
- FBO texture is standard OpenGL (0,0) = bottom-left
- `View.draw_quad()` flips again when drawing FBO to screen → displays correctly
- **Double-flip system** makes FBOs "just work"

### State Management
```python
fbo.begin()      # Saves current view, sets FBO viewport
push_view()      # Manual view state save
set_view(w, h)   # Apply top-left coordinate system
pop_view()       # Restore previous view
fbo.end()        # Restores saved view state
```

---

## Mouse/Input Coordinates

### Mouse Position
```python
mouse_x, mouse_y  # (0,0) at top-left, increases down-right
```
- Directly matches screen coordinates
- No transformation needed

### Click Regions
When testing if a click is inside a rectangle:
```python
if x <= mouse_x <= x + width and y <= mouse_y <= y + height:
    # Inside rectangle at (x, y) with size (width, height)
```

---

## OpenGL Default vs. Our System

| Aspect | OpenGL Default | Our System |
|--------|----------------|------------|
| **Screen Origin** | Bottom-left `(0, 0)` | Top-left `(0, 0)` |
| **Screen Y-axis** | Points UP | Points DOWN |
| **Texture Coords (Internal)** | Bottom-left `(0, 0)` | Bottom-left `(0, 0)` ✓ |
| **Texture Rendering** | No flip | Y-flipped via `draw_quad()` |
| **glOrtho** | `(0, W, 0, H, -1, 1)` | `(0, W, H, 0, -1, 1)` |

---

## Common Patterns

### Drawing Textures
```python
from modules.gl.View import draw_quad

# To screen/FBO
texture.bind()
draw_quad(x, y, width, height)  # Auto Y-flip
texture.unbind()
```

### Geometric Calculations (CentreGeometry Pattern)
```python
# Calculate in image space (intuitive)
anchor_img = Point2f(x_img, y_img)

# Convert to texture space (for shaders)
anchor_tex = Point2f(x_img, 1.0 - y_img)
rotation_tex = -rotation_img
roi_tex_y = 1.0 - (roi_img.y + roi_img.height)
```

### Drawing to FBO then to Screen
```python
from modules.gl.View import draw_quad

# Render to FBO (View.draw_quad flips → stores bottom-left texture)
fbo.begin()
texture.bind()
draw_quad(0, 0, fbo.width, fbo.height)
texture.unbind()
fbo.end()

# Draw FBO to screen (View.draw_quad flips again → correct display)
fbo.bind()
draw_quad(x, y, width, height)
fbo.unbind()
```

### SwapFbo Double Buffering
```python
swap_fbo.begin()          # Draw to current buffer (top-left origin)
swap_fbo.back_texture     # Read from previous buffer
# ... render ...
swap_fbo.end()
swap_fbo.swap()           # Swap buffers for next frame
```

---

## Utility Functions

### Fit/Fill Calculations
```python
from modules.gl.Utils import fit, fill

# Fit: Maintain aspect ratio, letterbox if needed
x, y, w, h = fit(src_w, src_h, dst_w, dst_h)

# Fill: Maintain aspect ratio, crop if needed
x, y, w, h = fill(src_w, src_h, dst_w, dst_h)
```
These return positions in top-left coordinate system.

---

## Advanced: Multi-Space Geometry (CentreGeometry Pattern)

Some layers like **CentreGeometry** work across multiple coordinate spaces for geometric accuracy:

### Coordinate Spaces Used

1. **Bbox-relative space** `[0, 1]` - Normalized within bounding box
2. **Image space** - Full image coordinates (top-left origin)
3. **Texture space** - OpenGL coordinates (bottom-left origin)
4. **Crop space** `[0, 1]` - Normalized within crop region

### Why Use Texture Space?

**Texture space is used for geometric calculations** because:
- ROI rectangles map directly to shader uniforms
- Rotation calculations align with OpenGL rendering
- No coordinate flip needed when passing to shaders

### Conversion Pattern

```python
# 1. Calculate in image space (top-left, intuitive)
anchor_img = shoulder_pos * bbox.size + bbox.position

# 2. Convert to texture space (bottom-left, for OpenGL)
anchor_tex = Point2f(anchor_img.x, 1.0 - anchor_img.y)

# 3. Calculate ROI in image space
roi_img = Rect(x, y, width, height)

# 4. Convert ROI to texture space (flip Y)
roi_tex = Rect(
    roi_img.x,
    1.0 - (roi_img.y + roi_img.height),  # Flip: bottom edge becomes top
    roi_img.width,
    roi_img.height
)

# 5. Negate rotation for texture space
rotation_tex = -rotation_img
```

### When to Use This Pattern

Use multi-space calculations when:
- Computing ROIs for shader-based cropping
- Calculating anchor points for geometric transformations
- Passing coordinates directly to shader uniforms
- Need precise alignment between CPU calculations and GPU rendering

**Example:** [CentreGeometry.py](modules/render/layers/centre/CentreGeometry.py) calculates pose-centered crop regions by working in image space then converting to texture space for shader consumption.

---

## Image/Tensor Data

### NumPy Arrays (from OpenCV, PIL, etc.)
- **OpenCV images**: Already in top-left origin (row 0 = top)
- **Shape**: `(height, width, channels)` or `(height, width)`
- **Directly compatible** with our coordinate system

### PyTorch Tensors
- **Shape**: `(C, H, W)` or `(H, W, C)` or `(H, W)`
- **First row**: Top of image
- **Directly compatible** with our coordinate system

### Upload to Texture
```python
image = Image()  # or Tensor()
image.set_image(numpy_array)  # or set_tensor(torch_tensor)
image.update()
# Texture (0,0) will be top-left of array
```

---

## Troubleshooting

### Image Appears Upside Down
- **Cause**: Using wrong `draw_quad()` - use `View.draw_quad()` for rendering
- **Cause**: Using `Shader.draw_quad()` outside shader context

### FBO Rendering Looks Flipped
- **Check**: `fbo.begin()` is called (sets top-left view)
- **Check**: Using `View.draw_quad()` not `Shader.draw_quad()`
- **Check**: Not mixing manual `glOrtho()` with `set_view()`

### Mouse Clicks Not Registering
- **Check**: Using screen coordinates directly (no Y inversion)
- **Check**: Rectangle test uses `y + height`, not `y - height`

---

## Quick Cheat Sheet

```python
# SCREEN SPACE
(0, 0)           # Top-left corner
(width, height)  # Bottom-right corner

# TEXTURE SPACE (Internal - OpenGL)
(0.0, 0.0)       # Bottom-left of texture coordinate system
(1.0, 1.0)       # Top-right of texture coordinate system

# RENDERED OUTPUT (After draw_quad flip)
(0.0, 0.0)       # Appears at top-left of screen
(1.0, 1.0)       # Appears at bottom-right of screen

# DRAWING
from modules.gl.View import draw_quad  # Use this for rendering
draw_quad(x, y, w, h)  # Always Y-flipped (screen/FBO)

from modules.gl.Shader import draw_quad  # Only for shaders
draw_quad()  # No flip (NDC space)

# FBO
fbo.begin()      # Auto-applies top-left coordinate system
fbo.end()        # Restores previous state

# VIEW MANAGEMENT
set_view(w, h)   # Apply top-left coordinate system
push_view()      # Save current state
pop_view()       # Restore state

# COORDINATE CONVERSION
image_y_to_tex_y = 1.0 - image_y     # Image (top-left) → Texture (bottom-left)
image_rot_to_tex = -image_rotation   # Negate rotation for texture space
```

---

## References
- [View.py](View.py) - `set_view()` implementation
- [Fbo.py](Fbo.py) - FBO coordinate system handling
- [Texture.py](Texture.py) - `draw_quad()` texture mapping
- [WindowManager.py](WindowManager.py) - Window coordinate callbacks
- [CentreGeometry.py](modules/render/layers/centre/CentreGeometry.py) - Multi-space geometric calculations
