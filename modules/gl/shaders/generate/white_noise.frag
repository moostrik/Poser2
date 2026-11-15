#version 460 core

uniform float random;
uniform vec2  resolution;

in vec2 texCoord;
out vec4 fragColor;

float hash(vec2 p) {
    vec3 p3  = fract(vec3(p.xyx) * .1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

void main() {
    float r = random;
    vec2 position = texCoord * resolution;
    vec2 pos = (position * 0.152 + r * 15000. + 50.0);
    float x = hash(pos.xy);
    float y = hash(pos.xy * 0.533);
    float z = hash(pos.xy * 4.22);
    fragColor = vec4(x, y, z, 1.0);
}