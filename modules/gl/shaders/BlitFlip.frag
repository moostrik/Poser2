#version 460 core

uniform sampler2D tex;
uniform bool flipX;
uniform bool flipY;

in vec2 texCoord;
out vec4 fragColor;

void main() {
    vec2 uv = texCoord;
    if (flipX) uv.x = 1.0 - uv.x;
    if (flipY) uv.y = 1.0 - uv.y;
    fragColor = texture(tex, uv);
}
