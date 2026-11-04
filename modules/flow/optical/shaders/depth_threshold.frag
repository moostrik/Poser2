#version 460 core

// Depth Threshold Shader
// Takes the noise floor off a signed field, continuously; with `gate`, only where the image moves (the
// 3×3 displacement past `motion_floor`), 0 elsewhere. The numpy reference is depth_velocity.depth_threshold

uniform sampler2D tex0;  // Motion toward the camera R
uniform float threshold; // Size taken off, in the field's units
uniform sampler2D displacement;     // uv per frame (RG), with gate
uniform bool gate;
uniform float motion_floor;         // Displacement the image must move by (px per frame)

in vec2 texCoord;
out vec4 fragColor;

bool moving() {
    vec2 size = vec2(textureSize(displacement, 0));
    ivec2 at = ivec2(floor(texCoord * size));
    ivec2 last_texel = ivec2(size) - 1;
    float largest = 0.0;
    for (int j = -1; j <= 1; j++) {
        for (int i = -1; i <= 1; i++) {
            vec2 d = texelFetch(displacement, clamp(at + ivec2(i, j), ivec2(0), last_texel), 0).xy * size;
            largest = max(largest, length(d));
        }
    }
    return largest > motion_floor;
}

void main() {
    float value = texture(tex0, texCoord).r;
    if (gate && !moving()) {
        value = 0.0;
    }
    fragColor = vec4(sign(value) * max(abs(value) - threshold, 0.0), 0.0, 0.0, 1.0);
}
