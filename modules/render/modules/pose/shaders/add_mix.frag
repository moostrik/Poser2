#version 460 core

uniform sampler2D dst;              // the base
uniform sampler2D src;              // the foreground
uniform float blend;
uniform float addCurve = 0.3;       // Higher = additive peaks later
uniform float blendCurve = 1.5;     // Higher = the foreground comes in later

in vec2 texCoord;
out vec4 fragColor;

void main() {
    vec4 base = texture(dst, texCoord);
    vec4 foreground = texture(src, texCoord);

    vec3 added = base.rgb + foreground.rgb * pow(blend, addCurve);
    vec3 blended = mix(added, foreground.rgb, pow(blend, blendCurve));
    blended = clamp(blended, 0.0, 1.0);

    fragColor = vec4(blended, base.a);
}
