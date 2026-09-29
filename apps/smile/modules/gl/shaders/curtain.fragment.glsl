#version 460 core

uniform sampler2D tex0;
uniform sampler2D tex1;
uniform float fade;

in vec2 texCoord;
out vec4 fragColor;

void main() {
    vec4 texel0 = texture(tex0, texCoord);
    vec4 texel1 = texture(tex1, texCoord);

    float f = pow(fade, 2.5);
    float fadeLeft = 0.5 - f * 0.5;
    float fadeRight = f * 0.5 + 0.5;


    if (texCoord.x > fadeLeft && texCoord.x < fadeRight) {
        fragColor = texel1;
    } else {
        fragColor = texel0;
    }
}