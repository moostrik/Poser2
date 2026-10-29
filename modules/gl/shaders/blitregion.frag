#version 460 core

uniform sampler2D tex;
uniform vec4 region; // x, y, width, height in normalized texture coordinates [0, 1]

in vec2 texCoord;
out vec4 fragColor;

void main() {
    // Map texCoord to the specified region of the texture
    vec2 regionTexCoord = region.xy + texCoord * region.zw;
    fragColor = texture(tex, regionTexCoord);
}
