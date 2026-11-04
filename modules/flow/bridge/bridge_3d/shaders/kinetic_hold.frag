#version 460 core

// Kinetic Hold Shader
// Peak-hold of the release, max(0, -acceleration): a stop reads as one blob, not a flicker.
// A pixel cannot tell a stop from a limb that left it, a patch can: the stop gate lets a release
// through as far as its patch is slowing down. The numpy references are kinetics.stop_gate and hold_release

uniform sampler2D tex0;  // Acceleration R
uniform sampler2D tex1;  // Previous held release R
uniform sampler2D tex2;  // Patch means: fast speed R, slow speed G, by mip level
uniform float decay;     // What remains of the held release this update
uniform float strength;  // Stop gate strength, 0 = every release counts
uniform float lod;       // Mip level of a patch

in vec2 texCoord;
out vec4 fragColor;

const float EPSILON = 1e-4;

void main() {
    float release = max(0.0, -texture(tex0, texCoord).r);
    if (strength > 0.0) {
        vec2 patch_speed = textureLod(tex2, texCoord, lod).rg;
        float slowdown = clamp((patch_speed.g - patch_speed.r) / max(patch_speed.g, EPSILON), 0.0, 1.0);
        release *= mix(1.0, slowdown, strength);
    }
    float held = texture(tex1, texCoord).r * decay;
    fragColor = vec4(max(release, held), 0.0, 0.0, 1.0);
}
