#version 460 core

uniform sampler2D tex0;
uniform sampler2D tex1;
uniform float fade;

in vec2 texCoord;
out vec4 fragColor;

void main() {
    vec4 texel0 = texture(tex0, texCoord);
    vec4 texel1 = texture(tex1, texCoord);
    fragColor = mix(texel0, texel1, fade);
}