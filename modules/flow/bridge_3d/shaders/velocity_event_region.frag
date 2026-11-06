#version 460 core

// Velocity Event Region Shader
// The flow's velocity where a pixel moves, and whether it moves: (velocity * moving, moving). A pixel
// moves where the motion gives it a velocity; the motion filters the noise within a frame. The mip
// levels of this are a region's sums over its moving pixels and its moving share, so a rise can read
// the region's mean velocity over its moving pixels: a limb arriving in the region adds moving pixels
// at its speed and leaves that mean as it was; a limb speeding up raises it

uniform sampler2D tex0;     // The flow's velocity RGB

in vec2 texCoord;
out vec4 fragColor;

const float EPSILON = 1e-4;     // EPSILON in kinetics: below this a speed is no motion

void main() {
    vec3 velocity = texture(tex0, texCoord).xyz;
    float moving = length(velocity) > EPSILON ? 1.0 : 0.0;
    fragColor = vec4(velocity * moving, moving);
}
