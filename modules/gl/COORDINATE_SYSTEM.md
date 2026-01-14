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

```
(0,0) ──────────► U
 │
 │
 │
 ▼ V

Standard texture mapping:
(0, 0) = top-left of texture
(1, 1) = bottom-right of texture
```

### Key Points
- **Standard OpenGL convention**: `(0, 0)` = bottom-left
- **Our convention**: Textures are **flipped** to match screen coordinates
- **U/S**: Horizontal coordinate (0.0 - 1.0)
- **V/T**: Vertical coordinate (0.0 - 1.0)

### Texture Flipping
The `draw_quad()` function handles texture flipping:
- **`flipV=False` (default)**: Matches our top-left screen coordinates
  - Maps texture `(0,1)` to screen top-left
  - Maps texture `(1,0)` to screen bottom-right
- **`flipV=True`**: Standard OpenGL (bottom-left origin)
  - Use when source texture is already in bottom-left coordinates

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
- `Fbo.begin()` calls `set_view(width, height)` internally
- Rendering to FBO uses top-left origin
- Reading from FBO texture uses standard texture coordinates
- **No additional flipping needed** when drawing FBO to screen

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
| **Origin** | Bottom-left `(0, 0)` | Top-left `(0, 0)` |
| **Y-axis** | Points UP | Points DOWN |
| **Texture (0,0)** | Bottom-left | Top-left |
| **glOrtho** | `(0, W, 0, H, -1, 1)` | `(0, W, H, 0, -1, 1)` |

---

## Common Patterns

### Drawing a Texture
```python
texture.bind()
draw_quad(x, y, width, height, flipV=False)
texture.unbind()
# Draws texture at (x, y) with top-left origin
```

### Drawing to FBO then to Screen
```python
# Render to FBO
fbo.begin()  # Sets top-left origin for FBO
draw_something()
fbo.end()

# Draw FBO to screen
fbo.draw(x, y, width, height)  # No flip needed
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
- **Cause**: Source texture is in OpenGL bottom-left coordinates
- **Solution**: Use `draw_quad(..., flipV=True)`

### FBO Rendering Looks Flipped
- **Check**: Verify `fbo.begin()` is called (it sets up proper view)
- **Check**: Not mixing manual `glOrtho()` calls with `set_view()`

### Mouse Clicks Not Registering
- **Check**: Using screen coordinates directly, not inverting Y
- **Check**: Rectangle test uses `y + height`, not `y - height`

---

## Quick Cheat Sheet

```python
# SCREEN SPACE
(0, 0)           # Top-left corner
(width, height)  # Bottom-right corner

# TEXTURE SPACE
(0.0, 0.0)       # Top-left of texture (after our mapping)
(1.0, 1.0)       # Bottom-right of texture

# DRAWING
draw_quad(x, y, w, h, flipV=False)  # Default: top-left origin
draw_quad(x, y, w, h, flipV=True)   # OpenGL bottom-left origin

# FBO
fbo.begin()      # Auto-applies top-left coordinate system
fbo.end()        # Restores previous state

# VIEW MANAGEMENT
set_view(w, h)   # Apply top-left coordinate system
push_view()      # Save current state
pop_view()       # Restore state
```

---

## References
- [View.py](View.py) - `set_view()` implementation
- [Fbo.py](Fbo.py) - FBO coordinate system handling
- [Texture.py](Texture.py) - `draw_quad()` texture mapping
- [WindowManager.py](WindowManager.py) - Window coordinate callbacks
