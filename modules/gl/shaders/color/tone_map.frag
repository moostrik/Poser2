#version 460 core

uniform sampler2D src;
uniform int operator;       // ToneMapOperator: 0 Reinhard, 1 ACES
uniform float exposure;

in vec2 texCoord;
out vec4 fragColor;

// Narkowicz 2015, "ACES Filmic Tone Mapping Curve"
vec3 aces(vec3 x) {
    return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0.0, 1.0);
}

void main() {
    vec4 color = texture(src, texCoord);
    vec3 c = max(color.rgb * exposure, 0.0);
    vec3 mapped = operator == 1 ? aces(c) : c / (1.0 + c);
    fragColor = vec4(mapped, color.a);
}
