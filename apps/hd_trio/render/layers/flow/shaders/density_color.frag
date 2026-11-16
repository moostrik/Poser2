#version 460 core

// The color a bridge's density carries: the image (or white) times a tint, gated by the mask's red,
// RGB and alpha alike.

uniform sampler2D image;     // The picture whose RGB colors the density
uniform sampler2D mask;      // The player's mask, red
uniform bool has_image;
uniform bool has_mask;
uniform vec3 color;          // The tint; the color alone without an image

in vec2 texCoord;
out vec4 fragColor;

void main() {
    vec4 base = has_image ? texture(image, texCoord) : vec4(1.0);
    float gate = has_mask ? texture(mask, texCoord).r : 1.0;
    fragColor = base * vec4(color, 1.0) * gate;
}
