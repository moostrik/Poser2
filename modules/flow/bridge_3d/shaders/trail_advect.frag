#version 460 core

// Trail Advect Shader
// Moves a field with a motion: each pixel takes the value from where the carrier trail says it came
// from, linear between pixels, so a trail travels with a moving limb instead of staying at a fixed pixel.
// The numpy reference is trails.advect, with the carrier as the displacement

uniform sampler2D tex0;  // Field to move
uniform sampler2D tex1;  // Carrier trail: the velocity that carries it (xy)
uniform vec2 carry;      // uv per frame per unit of trail velocity, times the advect amount

in vec2 texCoord;
out vec4 fragColor;

void main() {
    vec2 from = texCoord - texture(tex1, texCoord).xy * carry;
    fragColor = texture(tex0, from);
}
