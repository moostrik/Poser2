#version 460 core

uniform sampler2D src;
uniform float amount;
uniform float radius;       // where the darkening starts, 0..1 of the half diagonal
uniform float softness;     // over how much of the half diagonal it reaches full

in vec2 texCoord;
out vec4 fragColor;

void main() {
    vec4 color = texture(src, texCoord);
    vec2 size = vec2(textureSize(src, 0));
    vec2 offset = (texCoord - 0.5) * size;
    float distance = length(offset) / (0.5 * length(size));
    float shade = 1.0 - amount * smoothstep(radius, radius + softness, distance);
    fragColor = vec4(color.rgb * shade, color.a);
}
