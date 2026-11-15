#version 460 core

uniform sampler2D src;
uniform sampler2D dst;
uniform float fade;

in vec2 texCoord;
out vec4 fragColor;

void main() {
    vec4 texel0 = texture(src, texCoord);
    vec4 texel1 = texture(dst, texCoord);
    fragColor = mix(texel0, texel1, fade);
}
