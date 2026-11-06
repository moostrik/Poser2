#version 460 core

// Glyph Dots: a round, opaque dot in the colormap's color; a value too small to see draws nothing

in vec3 color;
in float amount;
out vec4 fragColor;

void main() {
    vec2 p = gl_PointCoord * 2.0 - 1.0;
    if (dot(p, p) > 1.0 || abs(amount) < 0.02) discard;
    float shade = 0.75 + 0.25 * sqrt(1.0 - dot(p, p));      // a little round
    fragColor = vec4(color * shade, 1.0);
}
