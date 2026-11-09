#version 460 core

// Velocity Change Region Mean Shader
// A region's mean velocity over its moving pixels: the region texture's sums at the region's mip level
// over its moving share; 0 where nothing in the region moves

uniform sampler2D tex0;  // Region texture: velocity * moving RGB, moving A, with mipmaps
uniform float lod;       // Mip level of a region

in vec2 texCoord;
out vec4 fragColor;

const float EPSILON = 1e-4;

void main() {
    vec4 sums = textureLod(tex0, texCoord, lod);
    fragColor = vec4(sums.a > EPSILON ? sums.rgb / sums.a : vec3(0.0), 0.0);     // w unused
}
