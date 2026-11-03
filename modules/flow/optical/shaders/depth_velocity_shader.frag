#version 460 core

// Depth Velocity Shader
// Motion toward the camera in view widths per frame, measured like the flow's x/y: over a window of
// depth samples, not one pixel. Each sample's change is its depth now minus its depth where it came
// from, `displacement` (uv per frame) back, over the metres one view width spans there,
// `depth × view_span`. A sample is averaged in when it has depth now and then, within `tolerance` of
// both its depth then (not a limb passing over it) and the pixel's (not the background beside the
// player). With less than `coverage` of the window averaged in, as at a silhouette, the motion is 0,
// and so it is without depth at the pixel or without a span. Read nearest, so an edge never reads as
// a distance in between. The numpy reference is depth_velocity.window_depth_displacement

uniform sampler2D tex0;  // Depth now R (m)
uniform sampler2D tex1;  // Depth the frame before R (m)
uniform sampler2D displacement;     // uv per frame (RG), with warp
uniform bool warp;
uniform vec2 depth_step; // uv between two depth samples
uniform int window;      // Radius in depth samples
uniform float tolerance; // Neighbours further than this in depth are not averaged in (m)
uniform float coverage;  // Share of the window averaged in that the motion needs
uniform float view_span; // Metres one view width spans per metre of depth

in vec2 texCoord;
out vec4 fragColor;

float depth_at(sampler2D depth, vec2 uv) {
    ivec2 size = textureSize(depth, 0);
    ivec2 texel = clamp(ivec2(floor(uv * vec2(size))), ivec2(0), size - 1);
    return texelFetch(depth, texel, 0).r;
}

void main() {
    float center = depth_at(tex0, texCoord);
    if (center <= 0.0 || view_span <= 0.0) {
        fragColor = vec4(0.0, 0.0, 0.0, 1.0);
        return;
    }
    vec2 moved = warp ? texture(displacement, texCoord).xy : vec2(0.0);

    float total = 0.0;
    int kept = 0;
    for (int j = -window; j <= window; j++) {
        for (int i = -window; i <= window; i++) {
            vec2 p = texCoord + vec2(i, j) * depth_step;
            float now = depth_at(tex0, p);
            float then = depth_at(tex1, p - moved);
            float change = now - then;
            if (now > 0.0 && then > 0.0 && abs(change) <= tolerance && abs(now - center) <= tolerance) {
                total -= change / (now * view_span);
                kept++;
            }
        }
    }
    float side = float(2 * window + 1);
    float needed = max(coverage * side * side, 1.0);
    fragColor = vec4(float(kept) >= needed ? total / float(kept) : 0.0, 0.0, 0.0, 1.0);
}
