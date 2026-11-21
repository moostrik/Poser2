#version 460 core

// Per pixel, the one of two with the larger amount, as it is; an amount is the size of its red (warm or cold
// alike), read from the texture itself or from the amount it is paired with.

layout(binding = 0) uniform sampler2D held;
layout(binding = 1) uniform sampler2D source;
layout(binding = 2) uniform sampler2D heldAmount;
layout(binding = 3) uniform sampler2D sourceAmount;

in vec2 texCoord;
out vec4 fragColor;

void main() {
    bool larger = abs(texture(sourceAmount, texCoord).r) > abs(texture(heldAmount, texCoord).r);
    fragColor = larger ? texture(source, texCoord) : texture(held, texCoord);
}
