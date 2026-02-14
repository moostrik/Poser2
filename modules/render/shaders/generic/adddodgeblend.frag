#version 460 core

uniform sampler2D uBase;    // Own camera's tinted mask (RGBA)
uniform sampler2D uFrg;     // Foreground (cel-shaded + hue-shifted)
uniform float uStrength;    // Blend strength (0 = base only, 1 = full effect)
uniform float uDodgeIntensity;  // Dodge blend intensity (0.0 - 1.0)
uniform float uAddCurve;        // Additive falloff exponent
uniform float uDodgeCurve;      // Dodge ramp exponent
uniform float uOpacityCurve;    // Foreground visibility ramp

in vec2 texCoord;
out vec4 fragColor;

vec3 colorDodge(vec3 base, vec3 blend) {
    return base / max(1.0 - blend, 0.001);
}

void main() {
    vec4 base = texture(uBase, texCoord);   // Tinted own mask
    vec4 frg = texture(uFrg, texCoord);     // Already has track color from HueShift

    // Blend modes: foreground brightens the colored mask
    vec3 added = clamp(base.rgb + frg.rgb, 0.0, 1.0);
    vec3 dodged = colorDodge(base.rgb, frg.rgb * uDodgeIntensity);
    dodged = clamp(dodged, 0.0, 1.0);

    // Crossfade from additive to dodge based on strength
    float addWeight = 1.0 - pow(uStrength, uAddCurve);
    float dodgeWeight = pow(uStrength, uDodgeCurve);

    vec3 combined = added * addWeight + dodged * dodgeWeight;

    // Foreground opacity ramp
    float frgOpacity = pow(uStrength, uOpacityCurve);

    // Blend: base (colored mask) → combined (with foreground energy)
    vec3 result = mix(base.rgb, combined, frg.a * frgOpacity);

    // Alpha: union of mask and foreground
    float alpha = max(base.a, frg.a * frgOpacity);

    fragColor = vec4(result, alpha);
}
