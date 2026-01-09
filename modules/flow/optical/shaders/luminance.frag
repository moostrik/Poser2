#version 460 core

uniform sampler2D tex0;
uniform bool flipV;

in vec2 texCoord;
out vec4 fragColor;

void main() {
    vec2 coord = flipV ? vec2(texCoord.x, 1.0 - texCoord.y) : texCoord;
    vec4 color = texture(tex0, coord);

    // Standard luminance weights (ITU-R BT.709)
    float luminance = dot(color.rgb, vec3(0.2126, 0.7152, 0.0722));

    fragColor = vec4(luminance, 0.0, 0.0, 1.0);
}
