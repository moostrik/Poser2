#version 460 core

// Depth Step Filter Shader
// The depth without its edges: a sample whose 3×3 depth samples, `depth_step` (uv) apart, span a
// step beyond `tolerance` (m) from the nearest to the furthest has no depth (0). At a silhouette
// that is the band where stereo depth jitters between the two surfaces. Read nearest, so an edge
// never reads as a distance in between. The numpy reference is depth_velocity.filter_depth_steps

uniform sampler2D tex0;  // Depth R (m), 0 where there is none
uniform vec2 depth_step; // uv between two depth samples
uniform float tolerance; // A step beyond this is two surfaces (m)

in vec2 texCoord;
out vec4 fragColor;

float depth_at(vec2 uv) {
    ivec2 size = textureSize(tex0, 0);
    ivec2 texel = clamp(ivec2(floor(uv * vec2(size))), ivec2(0), size - 1);
    return texelFetch(tex0, texel, 0).r;
}

void main() {
    float near = 1e30;
    float far = 0.0;
    for (int j = -1; j <= 1; j++) {
        for (int i = -1; i <= 1; i++) {
            float sample_depth = depth_at(texCoord + vec2(i, j) * depth_step);
            if (sample_depth > 0.0) {
                near = min(near, sample_depth);
                far = max(far, sample_depth);
            }
        }
    }
    bool edge = far > 0.0 && far - near > tolerance;
    fragColor = vec4(edge ? 0.0 : depth_at(texCoord), 0.0, 0.0, 0.0);
}
