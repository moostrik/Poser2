#version 460 core

// Kinetic Advect Shader
// Moves a field with the motion it holds: each pixel takes the value from where the fast trail says it
// came from, linear between pixels. The trails then travel with a moving limb, and the pixels it leaves
// fill from behind it instead of keeping its trail, so only a real stop leaves fast and slow apart.
// The numpy reference is kinetics.advect, with the fast trail as the displacement

uniform sampler2D tex0;  // Field to move
uniform sampler2D tex1;  // Fast trail: the velocity that carries it (xy)
uniform vec2 carry;      // uv per frame per unit of trail velocity, times the advect amount

in vec2 texCoord;
out vec4 fragColor;

void main() {
    vec2 from = texCoord - texture(tex1, texCoord).xy * carry;
    fragColor = texture(tex0, from);
}
