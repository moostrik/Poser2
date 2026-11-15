#version 460 core

in vec2 texCoord;

uniform sampler2D atlas;
uniform vec4 color;

out vec4 fragColor;

void main() {
    float alpha = texture(atlas, texCoord).r;
    fragColor = vec4(color.rgb, color.a * alpha);
}
