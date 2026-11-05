#version 460 core

// Compose Velocity Shader
// A flow's 3D velocity: its x/y and its z, each times its strength, zero where it is not finite:
// the bridges feed back, so one such pixel would stay and spread through their blur.
// With has_mask, times the mask

uniform sampler2D tex0;  // x/y RG
uniform sampler2D tex1;  // z R, read only with has_depth
uniform sampler2D tex2;  // mask R (0 to 1), read only with has_mask
uniform bool has_depth;
uniform bool has_mask;
uniform vec3 strength;   // Multiplier on x, y and z

in vec2 texCoord;
out vec4 fragColor;

void main() {
    vec3 velocity = vec3(texture(tex0, texCoord).xy, has_depth ? texture(tex1, texCoord).r : 0.0) * strength;
    if (has_mask) {
        velocity *= texture(tex2, texCoord).r;
    }
    bool finite = !any(isnan(velocity)) && !any(isinf(velocity));
    fragColor = vec4(finite ? velocity : vec3(0.0), 0.0);   // w unused
}
