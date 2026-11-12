#version 460 core

// Per pixel, the one of two with the larger amount, as it is: a density's amount is its alpha, a
// temperature's the size of its red (warm or cold alike).

uniform sampler2D held;
uniform sampler2D source;
uniform bool byAlpha;

in vec2 texCoord;
out vec4 fragColor;

void main() {
    vec4 a = texture(held, texCoord);
    vec4 b = texture(source, texCoord);
    bool larger = byAlpha ? b.a > a.a : abs(b.r) > abs(a.r);
    fragColor = larger ? b : a;
}
