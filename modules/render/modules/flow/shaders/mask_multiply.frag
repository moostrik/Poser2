#version 460 core

// tex0 times tex1's red: a mask by a mask, or a color image by a mask, premultiplied (RGB and alpha alike).

uniform sampler2D tex0;      // Color or mask
uniform sampler2D tex1;      // Mask, red

in vec2 texCoord;
out vec4 fragColor;

void main() {
    fragColor = texture(tex0, texCoord) * texture(tex1, texCoord).r;
}
