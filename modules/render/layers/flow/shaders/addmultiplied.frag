#version 460 core

uniform sampler2D tex0;
uniform sampler2D tex1;
uniform float strength0;
uniform float strength1;

in vec2 texCoord;
out vec4 fragColor;

void main() {
    vec4 color0 = texture(tex0, texCoord) * strength0;
    vec4 color1 = texture(tex1, texCoord) * strength1;

    fragColor = color0 + color1;
}
