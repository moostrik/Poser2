#version 460 core

uniform sampler2D tex0;      // Previous blended mask
uniform sampler2D tex1;      // Current mask (low-res)

in vec2 texCoord;
out vec4 fragColor;

void main() {
    float texel0 = texture(tex0, texCoord).r;
    float texel1 = texture(tex1, texCoord).r;
    fragColor = vec4(texel0 * texel1, 0.0, 0.0, 0.0);
}