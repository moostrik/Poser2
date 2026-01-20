#version 460 core
uniform sampler2D tex;

in vec2 texCoord;
out vec4 fragColor;

#define TINY 0.000001

void main() {
    vec4 v = texture(tex, texCoord);

    // Normalize with TINY trick to avoid division by zero
    fragColor = normalize(v + TINY);
}
