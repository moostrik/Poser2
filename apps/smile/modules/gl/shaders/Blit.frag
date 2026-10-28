#version 460 core

uniform sampler2D tex;
uniform float opacity;

in vec2 texCoord;
out vec4 fragColor;

void main() {
    vec4 color = texture(tex, texCoord);
    fragColor = vec4(color.rgb, color.a * opacity);
}
