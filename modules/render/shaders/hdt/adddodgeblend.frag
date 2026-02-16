#version 460 core

uniform sampler2D uTex0;
uniform sampler2D uTex1;
uniform float uBlend;
uniform float uAddCurve = 0.3;    // Higher = additive peaks later
uniform float uBlendCurve = 1.5;  // Higher = dodge kicks in later

in vec2 texCoord;
out vec4 fragColor;

void main() {
    vec4 base = texture(uTex0, texCoord);
    vec4 frg = texture(uTex1, texCoord);

    vec3 added = base.rgb + frg.rgb * pow(uBlend, uAddCurve);
    vec3 blended = mix(added, frg.rgb, pow(uBlend, uBlendCurve));
    blended = clamp(blended, 0.0, 1.0);

    float alpha = base.a;

    fragColor = vec4(blended, alpha);
}