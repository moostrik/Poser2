#version 460 core

// Depth Velocity Shader
// Velocity toward the camera from two smoothed depths, 0 without depth at both and at a jump. With
// `warp` the previous depth is the one where the pixel came from, `displacement` (uv per frame) back,
// taken nearest: the change is measured along the motion, so a surface sliding sideways over a slope
// reads no motion toward the camera. The numpy reference is kinetics.depth_speed, with kinetics.warped

uniform sampler2D tex0;  // Smoothed depth R
uniform sampler2D tex1;  // Previous smoothed depth R
uniform sampler2D displacement;     // uv per frame (RG), with warp
uniform bool warp;
uniform float dt;        // Seconds between them
uniform float jump;      // More change than this is an edge (m)
uniform float scale;     // Converts m/s to simulation velocity

in vec2 texCoord;
out vec4 fragColor;

float previous_at(vec2 uv) {
    if (!warp) {
        return texture(tex1, uv).r;
    }
    ivec2 size = textureSize(tex1, 0);
    ivec2 from = clamp(ivec2(floor((uv - texture(displacement, uv).xy) * vec2(size))), ivec2(0), size - 1);
    return texelFetch(tex1, from, 0).r;
}

void main() {
    float depth = texture(tex0, texCoord).r;
    float previous = previous_at(texCoord);
    float change = depth - previous;
    bool moving = depth > 0.0 && previous > 0.0 && abs(change) <= jump;
    fragColor = vec4(moving ? -change / dt * scale : 0.0, 0.0, 0.0, 1.0);
}
