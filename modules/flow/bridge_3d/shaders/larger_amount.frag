#version 460 core

// Per pixel, the one of two with the larger amount, as it is; an amount is the size of its red (warm or cold
// alike), read from the texture itself or from the amount it is paired with.

uniform sampler2D held;
uniform sampler2D source;
uniform sampler2D heldAmount;
uniform sampler2D sourceAmount;

in vec2 texCoord;
out vec4 fragColor;

void main() {
    bool larger = abs(texture(sourceAmount, texCoord).r) > abs(texture(heldAmount, texCoord).r);
    fragColor = larger ? texture(source, texCoord) : texture(held, texCoord);
}
