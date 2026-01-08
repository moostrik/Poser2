# Coordinate System Convention

## Overview

This application uses a **top-left origin coordinate system** throughout, NOT the standard OpenGL bottom-left origin.

## Screen/Window Space

```
(0, 0) ──────────────► X
  │
  │
  │
  ▼
  Y                (width, height)
```

- **Origin (0, 0)**: Top-left corner of screen/window
- **X-axis**: Increases rightward (standard)
- **Y-axis**: Increases **downward** (non-standard for OpenGL)

This matches:
- Window coordinate systems (GLFW, Win32, etc.)
- NumPy/image array indexing `image[y, x]`
- Most UI frameworks

## Implementation

### Projection Matrix
All rendering uses `glOrtho(0, width, height, 0, -1, 1)`:
- `Fbo.begin()` - FBO rendering
- `SwapFbo.begin()` - Swap FBO rendering
- `RenderBase.setView()` - Main window rendering

The key is the **inverted Y range**: `(height, 0)` instead of `(0, height)`.

### Texture Coordinates

**draw_quad()** default behavior (`flipV=False`):
- V = 1.0 at top (vertex Y = 0)
- V = 0.0 at bottom (vertex Y = height)
- This flips textures to match our top-down Y-axis

**Image/Tensor Upload Strategy**:
1. NumPy arrays already match our coordinate system (row 0 = top)
2. OpenGL textures use bottom-left origin (V=0.0 at bottom)
3. During upload, render to FBO with `flipV=True` to convert NumPy → OpenGL texture space
4. During draw, use default `flipV=False` to convert texture space → our screen space

### Why This Works

```
NumPy Array          OpenGL Texture         Screen Space
(Top-left origin)    (Bottom-left origin)   (Top-left origin)

Row 0 (top)          ┌─────────┐ V=1.0      ┌─────────┐ Y=0
  ...                │         │             │         │
Row N (bottom)       └─────────┘ V=0.0      └─────────┘ Y=height

                Upload with          Draw with
                flipV=True          flipV=False
                (flip once)         (flip back)
```

## Implications

1. **Vertex Drawing**: Draw from top to bottom (Y: 0 → height)
2. **Texture Sampling**: Flipped V coordinates compensate for coordinate system mismatch
3. **FBO Rendering**: Renders "upside down" relative to standard OpenGL, but correct for our space
4. **Compatibility**: Shaders work normally - they see standard texture coordinates

## Alternative: Standard OpenGL

To use standard OpenGL (bottom-left origin):
- Change all `glOrtho` to `(0, width, 0, height, -1, 1)`
- Remove texture coordinate flipping from `draw_quad`
- Flip NumPy arrays during upload (CPU or GPU)
- Update all Y-coordinate logic throughout the codebase

**Not recommended** - too many systems depend on top-left convention.
