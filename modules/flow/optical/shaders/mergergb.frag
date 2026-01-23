#version 460 core

uniform sampler2D tex0;

in vec2 texCoord;
out vec4 fragColor;

void main() {
    vec4 color = texture(tex0, texCoord);

    float sth = 1.0;

    float r = smoothstep(0.0, sth, color.r);
    float g = smoothstep(0.0, sth, color.g);
    float b = smoothstep(0.0, sth, color.b);

    float merged = (r + g + b);

    fragColor = vec4(merged, 0.0, 0.0, 1.0);
}
