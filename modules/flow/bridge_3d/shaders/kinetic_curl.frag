#version 460 core

// Kinetic Curl Shader
// 2D curl of a velocity field, in velocity per texel like the fluid's curl

uniform sampler2D tex0;  // Velocity RG

in vec2 texCoord;
out vec4 fragColor;

void main() {
    vec2 texel = 1.0 / vec2(textureSize(tex0, 0));
    float dvdx = texture(tex0, texCoord + vec2(texel.x, 0.0)).y - texture(tex0, texCoord - vec2(texel.x, 0.0)).y;
    float dudy = texture(tex0, texCoord + vec2(0.0, texel.y)).x - texture(tex0, texCoord - vec2(0.0, texel.y)).x;
    fragColor = vec4(0.5 * (dvdx - dudy), 0.0, 0.0, 1.0);
}
