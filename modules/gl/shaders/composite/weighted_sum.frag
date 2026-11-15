#version 460 core

uniform sampler2D a;
uniform sampler2D b;
uniform float aWeight;
uniform float bWeight;      // 0: b is not read

in vec2 texCoord;
out vec4 fragColor;

void main() {
    vec4 sum = texture(a, texCoord) * aWeight;
    if (bWeight != 0.0) sum += texture(b, texCoord) * bWeight;
    fragColor = sum;
}
