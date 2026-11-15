#version 460 core

uniform sampler2D src;
uniform vec4 tint;

in vec2 texCoord;
out vec4 fragColor;

void main() {
    float mask = texture(src,texCoord).r;
    // Output: tint color where mask exists, mask value as alpha
    fragColor = vec4(tint.rgb, mask);
}
