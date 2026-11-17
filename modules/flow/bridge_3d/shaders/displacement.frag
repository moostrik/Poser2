#version 460 core

// Displacement
// The rate at which a moving body fills the water, dchi/dt = -v . grad(chi), chi the body's share of a cell:
// positive where the body moves into water, negative where it leaves, about 0 summed, 0 for a still body.
// Red: the image plane's part, -(v_x dm/dx + v_y dm/dy), 1/s, from the velocity (widths/s) and the mask's
// central differences at the velocity's texels, in widths: x spans one width over u, y H/W widths over v.
// Green: z's part as the fluid needs it, v_z times the mask (widths/s); the fluid takes its slope in depth.

uniform sampler2D velocity;     // the mixed velocity, xyz in widths/s
uniform sampler2D mask;         // the player's mask (red, 0 to 1)

in vec2 texCoord;
out vec4 fragColor;

void main() {
    vec3 v = texture(velocity, texCoord).xyz;
    vec2 size = vec2(textureSize(velocity, 0));
    vec2 texel = 1.0 / size;
    float m = texture(mask, texCoord).r;
    float dm_du = (texture(mask, texCoord + vec2(texel.x, 0.0)).r - texture(mask, texCoord - vec2(texel.x, 0.0)).r) / (2.0 * texel.x);
    float dm_dv = (texture(mask, texCoord + vec2(0.0, texel.y)).r - texture(mask, texCoord - vec2(0.0, texel.y)).r) / (2.0 * texel.y);
    vec2 gradient = vec2(dm_du, dm_dv * size.x / size.y);     // per width
    fragColor = vec4(-dot(v.xy, gradient), v.z * m, 0.0, 0.0);
}
