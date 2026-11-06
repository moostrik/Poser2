#version 460 core

// Glyph Arrows: the arrow's direction color, lit by one light from the front and above, so its shape
// reads from any angle

in vec3 normal;
in vec3 color;
out vec4 fragColor;

const vec3 LIGHT = vec3(0.35, 0.6, 0.72);    // normalized below
const float AMBIENT = 0.35;

void main() {
    float lit = AMBIENT + (1.0 - AMBIENT) * max(dot(normalize(normal), normalize(LIGHT)), 0.0);
    fragColor = vec4(color * lit, 1.0);
}
