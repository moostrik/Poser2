#version 460 core

// A mask's edge softened inward: the mask blurred is 0.5 on its edge, so 2 × blurred − 1 is 0 there and outside
// it, and rises to 1 inside over about two sigmas of the blur (erf).

layout(binding = 0) uniform sampler2D blurred;

in vec2 texCoord;
out vec4 fragColor;

void main() {
    float feathered = clamp(2.0 * texture(blurred, texCoord).r - 1.0, 0.0, 1.0);
    fragColor = vec4(feathered, feathered, feathered, 1.0);
}
