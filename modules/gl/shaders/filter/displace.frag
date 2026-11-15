#version 460 core

uniform sampler2D src;
uniform sampler2D displacement;     // red across, green down; 0.5 is no move
uniform float amount;               // the move at 0 or 1, in 0..1 of the image

in vec2 texCoord;
out vec4 fragColor;

void main() {
    vec2 move = (texture(displacement, texCoord).rg - 0.5) * 2.0 * amount;
    fragColor = texture(src, texCoord + vec2(move.x, -move.y));     // green down: GL's v runs up
}
