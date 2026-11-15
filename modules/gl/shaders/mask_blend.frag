#version 460 core

uniform sampler2D src;      // Previous blended mask
uniform sampler2D dst;      // Current mask (low-res)
uniform float fade;         // Temporal blend factor

in vec2 texCoord;
out vec4 fragColor;

void main() {
    float texel0 = texture(src, texCoord).r;
    float texel1 = texture(dst, texCoord).r;
    fragColor = vec4(mix(texel0, texel1, fade), 0.0, 0.0, 0.0);
}
