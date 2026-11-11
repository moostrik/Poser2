#version 460 core

// Z Position Shader
// Where the body is now in the player's depth range, a z position: (depth - near) / (far - near), 0 front to
// 1 back, from the depth track's disparity (1 / depth), times its weight in red, the weight in green. The
// weight is the mask's (1 without one) where the depth lies from near to far, else 0: no z position, not the
// front. The numpy reference is lucas_kanade_flow_3d.z_position

uniform sampler2D tex0;  // the depth track (RGBA32F): its disparity (1/m) in green
uniform sampler2D tex1;  // mask R (0 to 1), read only with has_mask
uniform bool has_mask;
uniform float near;      // m, the range's front
uniform float far;       // m, its back

in vec2 texCoord;
out vec4 fragColor;

void main() {
    float disparity = texture(tex0, texCoord).g;
    float depth = disparity > 1e-6 ? 1.0 / disparity : 0.0;
    float weight = depth >= near && depth <= far && far > near ? 1.0 : 0.0;
    if (has_mask) weight *= texture(tex1, texCoord).r;
    float z = weight > 0.0 ? (depth - near) / (far - near) : 0.0;
    fragColor = vec4(z * weight, weight, 0.0, 0.0);
}
