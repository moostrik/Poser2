#version 460 core

uniform sampler2D uBase;
uniform sampler2D uFrg;
uniform sampler2D uMask;
uniform vec3 uTrackColor;
uniform float uStrength;

in vec2 texCoord;
out vec4 fragColor;

vec3 colorDodge(vec3 base, vec3 blend) {
    return base / max(1.0 - blend, 0.001);
}

void main() {
    vec4 base = texture(uBase, texCoord);
    vec4 frg = texture(uFrg, texCoord);
    float mask = texture(uMask, texCoord).r;

// Blend modes
    vec3 added = clamp(frg.rgb + uTrackColor, 0.0, 1.0);  // True additive
    vec3 dodged = colorDodge(frg.rgb, uTrackColor * 0.5);
    dodged = clamp(dodged, 0.0, 1.0);

    // Add weight: stays high until late
    float addWeight = 1.0 - pow(uStrength, 2.0);
    // Dodge weight: kicks in late
    float dodgeWeight = pow(uStrength, 1.5);

    // Combine add and dodge
    vec3 combined = added * addWeight + dodged * dodgeWeight;

    // Foreground opacity ramps up quickly to show bright effects early
    float frgOpacity = pow(uStrength, 0.3);

    // Blend with base using mask and ramped opacity
    vec3 result = mix(base.rgb, combined, mask * frgOpacity);

    // Preserve base alpha
    fragColor = vec4(result, base.a);
}
