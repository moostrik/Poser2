#version 460 core

// Velocity Bandpass Region Mean 2D Shader
// A region's mean velocity over its moving pixels: the region texture's sums at the region's mip level over its
// moving share; 0 where nothing in the region moves

layout(binding = 0) uniform sampler2D tex0; // Region texture: velocity * moving RG, moving B, with mipmaps
uniform float lod;       // Mip level of a region

in vec2 texCoord;
out vec4 fragColor;

const float EPSILON = 1e-4;

void main() {
    vec3 sums = textureLod(tex0, texCoord, lod).rgb;
    fragColor = vec4(sums.b > EPSILON ? sums.rg / sums.b : vec2(0.0), 0.0, 0.0);     // z and w unused
}
