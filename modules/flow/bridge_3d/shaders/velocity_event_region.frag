#version 460 core

// Velocity Event Region Shader
// The flow's velocity where a pixel moves, and whether it moves: (velocity * moving, moving). The mip
// levels of this are a region's sums over its moving pixels and its moving share, so the start's region
// weight can take the region's mean velocity over its moving pixels: a limb arriving in the region adds
// moving pixels at its speed and leaves that mean as it was; a limb speeding up raises it

uniform sampler2D tex0;     // The flow's velocity RGB
uniform float floor_speed;  // widths/s: a pixel slower than this does not move

in vec2 texCoord;
out vec4 fragColor;

void main() {
    vec3 velocity = texture(tex0, texCoord).xyz;
    float moving = length(velocity) > floor_speed ? 1.0 : 0.0;
    fragColor = vec4(velocity * moving, moving);
}
