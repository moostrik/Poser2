#version 460 core

// Kinetic Input Shader
// Copies the optical flow and the velocity toward the camera into the bridge as one vector, zeroing
// what is not finite: the trails feed back, so one such pixel would stay and spread through the blur

uniform sampler2D tex0;  // Optical flow RG
uniform sampler2D tex1;  // Velocity toward the camera R, read only with has_depth
uniform bool has_depth;

in vec2 texCoord;
out vec4 fragColor;

void main() {
    vec3 velocity = vec3(texture(tex0, texCoord).xy, has_depth ? texture(tex1, texCoord).r : 0.0);
    bool finite = !any(isnan(velocity)) && !any(isinf(velocity));
    fragColor = vec4(finite ? velocity : vec3(0.0), 1.0);
}
