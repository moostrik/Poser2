#version 460 compatibility

// Coordinate System: Top-left origin, pixel-based
// All textures have uniform V orientation (top at V=1) after GPU flip on upload.
// See COORDINATE_SYSTEM.md for details

out vec2 texCoord;

uniform vec2 resolution;
uniform bool flipV = true;  // Flip V to sample normalized textures (top at V=1)

void main() {
    // Get position from immediate mode (glVertex2f) - pixel coordinates
    vec2 position = gl_Vertex.xy;

    // Texture coordinates
    // flipV=true (default): flip V to sample top of texture (V=1) at top of quad
    // flipV=false: no flip, for staging textures with top at V=0
    texCoord.x = gl_MultiTexCoord0.x;
    texCoord.y = flipV ? (1.0 - gl_MultiTexCoord0.y) : gl_MultiTexCoord0.y;

    // Convert pixels to NDC with Y-flip for top-left origin
    vec2 ndc = (position / resolution) * 2.0 - 1.0;
    ndc.y = -ndc.y;  // Always flip: pixel (0,0) -> NDC (-1, +1) = top-left

    gl_Position = vec4(ndc, 0.0, 1.0);
}