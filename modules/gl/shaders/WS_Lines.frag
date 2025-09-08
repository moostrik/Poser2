#version 460 core

uniform sampler2D tex0;

in vec2 texCoord;
out vec4 fragColor;

void main() {
    vec2 lineTexCoord = vec2(texCoord.x, 0.0);
    float white = texture(tex0, lineTexCoord).b * 0.8;
    float blue = texture(tex0, lineTexCoord).g;
    vec3 color = vec3(white, white, white);
    color += vec3(blue * 0.05, 0.0, blue);

    fragColor = vec4(vec3(clamp(color, 0.0, 1.0)), 1.0);
}