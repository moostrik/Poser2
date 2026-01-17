#version 460 core

uniform sampler2D tex;

in vec2 texCoord;
out vec4 fragColor;

void main() {
    vec4 v = texture(tex, texCoord);
    float mag = length(v);
    // Output to R channel (for R32F textures)
    fragColor = vec4(mag, 0.0, 0.0, 0.0);
}
