#version 460 core

uniform sampler2D uBase;
uniform sampler2D uFrg;
uniform float uStrength;
uniform float uAddCurve = 0.3;    // Higher = additive peaks later
uniform float uBlendCurve = 1.5;  // Higher = dodge kicks in later

in vec2 texCoord;
out vec4 fragColor;

void main() {
    vec4 base = texture(uBase, texCoord);
    vec4 frg = texture(uFrg, texCoord);

    float strength = uStrength;

    vec3 added = base.rgb + frg.rgb * pow(strength, uAddCurve);
    vec3 blended = mix(added, frg.rgb, pow(strength, uBlendCurve));
    blended = clamp(blended, 0.0, 1.0);

    float alpha = base.a;

    fragColor = vec4(blended, alpha);
}