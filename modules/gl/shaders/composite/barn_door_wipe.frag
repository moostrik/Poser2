#version 460 core

uniform sampler2D src;
uniform sampler2D dst;
uniform float fade;

in vec2 texCoord;
out vec4 fragColor;

void main() {
    vec4 texel0 = texture(src, texCoord);
    vec4 texel1 = texture(dst, texCoord);

    float f = pow(fade, 2.5);
    float fadeLeft = 0.5 - f * 0.5;
    float fadeRight = f * 0.5 + 0.5;


    if (texCoord.x > fadeLeft && texCoord.x < fadeRight) {
        fragColor = texel1;
    } else {
        fragColor = texel0;
    }
}