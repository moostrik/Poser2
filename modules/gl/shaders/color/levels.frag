#version 460 core

uniform sampler2D src;
uniform float inBlack;
uniform float inWhite;
uniform float gamma;
uniform float outBlack;
uniform float outWhite;

in vec2 texCoord;
out vec4 fragColor;

void main() {
    vec4 color = texture(src, texCoord);
    vec3 c = clamp((color.rgb - inBlack) / max(inWhite - inBlack, 1e-6), 0.0, 1.0);
    c = pow(c, vec3(1.0 / max(gamma, 1e-6)));
    fragColor = vec4(mix(vec3(outBlack), vec3(outWhite), c), color.a);
}
