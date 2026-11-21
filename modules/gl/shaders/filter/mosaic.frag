#version 460 core

layout(binding = 0) uniform sampler2D src;
uniform float block;        // pixels of src

in vec2 texCoord;
out vec4 fragColor;

void main() {
    vec2 size = vec2(textureSize(src, 0));
    vec2 center = (floor(texCoord * size / block) + 0.5) * block / size;
    fragColor = texture(src, center);
}
