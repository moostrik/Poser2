#version 460 core

uniform sampler2D tex0;
uniform float scale;
uniform float angle;

in vec2 texCoord;
out vec4 fragColor;

void main() {
    vec2 scaledTexCoord = vec2(texCoord.x * 1.0, 0.0);
    vec2 tc = vec2(1.0, 1.0);

    vec4 texel0 = texture(tex0, texCoord);
    fragColor = texel0;

    // fragColor = vec4(1.0, 1.0, 0.0, 1.0);

    // Apply rotation around the center of the texture
}