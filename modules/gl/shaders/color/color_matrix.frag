#version 460 core

layout(binding = 0) uniform sampler2D src;
uniform mat3 matrix;
uniform vec3 offset;

in vec2 texCoord;
out vec4 fragColor;

void main() {
    vec4 color = texture(src, texCoord);
    fragColor = vec4(matrix * color.rgb + offset, color.a);
}
