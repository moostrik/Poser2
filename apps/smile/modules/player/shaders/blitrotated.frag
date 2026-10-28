#version 460 core

uniform sampler2D tex;

in vec2 texCoord;
out vec4 fragColor;

void main() {
    // 90 degrees counterclockwise: the top of the texture ends up at the left
    fragColor = texture(tex, vec2(texCoord.y, 1.0 - texCoord.x));
}
